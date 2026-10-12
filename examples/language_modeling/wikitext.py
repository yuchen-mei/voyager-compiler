import argparse
import logging
import os

import torch
from accelerate.utils import get_max_memory
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization_configs import QUANTIZATION_CONFIGS, set_qconfig
from voyager_compiler import (
    ShapeProp,
    add_qspec_args,
    get_default_quantizer,
    prepare_pt2e,
    convert_pt2e,
    plot_histogram,
    plot_layer_range,
    with_execution_context,
    print_node_scope_tabular,
    get_aten_graph_module,
    get_device_map,
    dispatch_model,
    insert_align_device_nodes,
    sink_obs_or_fq,
)


logger = logging.getLogger(__name__)


def parse_args(argv=None, **defaults):
    """Parse the shared WikiText CLI, with optional caller-supplied defaults.

    The standalone Llama example uses its historical defaults. External model
    launchers may supply ``argv`` and override model-specific defaults without
    copying the parser or changing quantization and execution settings.
    """
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument(
        '--model_id', required='model_id' not in defaults,
        help='Pretrained model identifier'
    )
    parser.add_argument(
        '--max_length', type=int, default=1024, help='Maximum sequence length'
    )
    parser.add_argument(
        '--stride', type=int, default=512, help='Stride for processing the data'
    )
    parser.add_argument(
        '--output_dir', default=None, help='Output directory for histograms'
    )
    parser.add_argument(
        '--torch_dtype',
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help=(
            "Override the default `torch.dtype` and load the model under this "
            "dtype. If `auto` is passed, the dtype will be automatically "
            "derived from the model's weights."
        )
    )
    parser.add_argument(
        '--qconfig', default=None, help='Quantization scheme for the model'
    )
    parser.add_argument(
        '--reserved_memory',
        type=int,
        default=8,
        help='GPU memory reserved for storing activations'
    )
    parser.add_argument('--revision', default=None, help='Checkpoint revision')
    parser.add_argument(
        '--eager', action='store_true',
        help='Evaluate the loaded model directly, without export or quantization'
    )
    parser.add_argument(
        '--static_shapes', action='store_true',
        help='Export fixed-length inputs instead of a dynamic sequence dimension'
    )
    parser.add_argument(
        '--max_eval_samples', type=int, default=None,
        help='Optional limit on the number of test windows'
    )
    add_qspec_args(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.eager and (
        args.qconfig is not None
        or any(value is not None for value in (
            args.activation, args.output_activation, args.weight, args.bias
        ))
        or args.convert_model
    ):
        parser.error('[ERROR] --eager cannot be combined with quantization settings.')
    return args


def evaluate_wikitext(
    model, input_ids, *, max_length=1024, stride=512, device=None,
    max_eval_samples=None,
):
    """Score token IDs with the shared Llama full-window perplexity protocol.

    The CLI and external-model validation drivers call this with a causal LM
    accepting ``input_ids``, ``labels``, and ``use_cache=False`` and returning
    ``outputs.loss``. ``input_ids`` is the single concatenated WikiText stream;
    ``device`` controls input placement. Optional ``max_eval_samples`` limits
    smoke tests. Window starts, overlap masking, omitted tail, and the mean of
    window losses preserve the original evaluator's behavior. No recurrent or
    attention state is carried between prefills. Returns perplexity, summed
    negative log likelihood, and the scored-token, window, and input counts.
    """
    if max_length < 2 or not 1 <= stride <= max_length:
        raise ValueError('[ERROR] Require max_length >= 2 and 1 <= stride <= max_length.')
    if max_eval_samples is not None and max_eval_samples < 1:
        raise ValueError('[ERROR] max_eval_samples must be positive.')
    seq_len = input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    scored_tokens = 0
    negative_log_likelihood = 0.0
    # Preserve the existing strict bound: the final partial/tail window is omitted.
    for begin_loc in tqdm(range(0, seq_len - max_length, stride)):
        if max_eval_samples is not None and len(nlls) >= max_eval_samples:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        window_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = window_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(window_ids, labels=target_ids, use_cache=False)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        # Count labels after the model's causal shift, including overlap masking.
        num_targets = (target_ids[:, 1:] != -100).sum().item()
        scored_tokens += num_targets
        negative_log_likelihood += neg_log_likelihood.item() * num_targets
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    if not nlls:
        raise ValueError('[ERROR] The token stream must be longer than max_length.')
    ppl = torch.exp(torch.stack(nlls).mean())
    return {
        'perplexity': ppl.item(),
        'negative_log_likelihood': negative_log_likelihood,
        'scored_tokens': scored_tokens,
        'windows': len(nlls),
        'input_tokens': seq_len,
    }


def main(args, *, model_loader=None, model_exporter=None):
    """Run the existing WikiText preparation, calibration, and evaluation flow.

    The standalone CLI calls this through ``with_execution_context``. External
    launchers may bind ``model_loader`` before applying that same decorator.
    The loader receives the checkpoint ID, ``torch_dtype``, ``device``, and
    ``revision`` and returns an HF-compatible causal model and tokenizer.
    Without a loader, the original AutoModel/AutoTokenizer path is used.
    ``--eager`` evaluates the loaded model; otherwise the shared PT2E pipeline
    runs, with dynamic shapes unless ``--static_shapes`` is requested.
    An optional ``model_exporter(model, example_args, example_kwargs,
    dynamic_shapes=...)`` returns a graph with model-specific decompositions
    before the shared quantization preparation; it is unused for eager runs.
    Returns the dictionary produced by ``evaluate_wikitext``.
    """
    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else None
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    if model_loader is None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            device_map=args.gpu,
            attn_implementation="eager", # Turn off flash attention
            revision=args.revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    else:
        model, tokenizer = model_loader(
            args.model_id, torch_dtype=torch_dtype,
            device=device or 'cpu', revision=args.revision,
        )

    if not args.eager:
        quantizer = get_default_quantizer(
            input_activation=args.activation,
            weight=args.weight,
            bias=args.bias,
            record_histogram=args.record_histogram,
            force_scale_power_of_two=args.force_scale_power_of_two,
        )

        if (qconfig := QUANTIZATION_CONFIGS.get(args.qconfig)) is not None:
            set_qconfig(quantizer, qconfig)

        input_ids = torch.randint(
            0, model.config.vocab_size, (1, args.max_length), device=device
        )
        example_args = (input_ids,)
        example_kwargs = {"labels": input_ids.clone(), "use_cache": False}

        dynamic_shapes = None
        if not args.static_shapes:
            seq_len = torch.export.Dim("seq_length", min=3, max=args.max_length)
            dynamic_shapes = {
                "input_ids": {1: seq_len},
                "labels": {1: seq_len},
                "use_cache": None,
            }

        # gm = get_aten_graph_module(model, example_args, example_kwargs, dynamic_shapes)
        # gm.graph.print_tabular()
        # print_node_scope_tabular(gm)

        # LLaMA implementation includes @torch.no_grad() statement, which will turn
        # gradient on if capture_pre_autograd_graph is not called with torch.no_grad().
        with torch.no_grad():
            if model_exporter is not None:
                model = model_exporter(
                    model, example_args, example_kwargs,
                    dynamic_shapes=dynamic_shapes,
                )
            model = prepare_pt2e(
                model, quantizer, example_args, example_kwargs, dynamic_shapes
            )

        sink_obs_or_fq(model)

        if args.gpu is None and torch.cuda.is_available():
            reserved_memory = args.reserved_memory * 1024 ** 3
            max_memory = {
                k: v - reserved_memory for k, v in get_max_memory().items()
                if isinstance(k, int) and v > reserved_memory
            }

            device_map = get_device_map(model, max_memory, verbose=True)
            dispatch_model(model, device_map)

            for node in list(model.graph.nodes):
                if node.op not in ["placeholder", "output"] and not node.users:
                    model.graph.erase_node(node)

            insert_align_device_nodes(model, (input_ids, example_kwargs["labels"]))

        def calibrate(model):
            validation = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            encodings = tokenizer("\n\n".join(validation["text"]), return_tensors="pt")
            seq_len = encodings.input_ids.size(1)
            for i, begin_loc in enumerate(tqdm(range(0, seq_len - args.max_length, args.stride))):
                end_loc = min(begin_loc + args.max_length, seq_len)
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                with torch.no_grad():
                    model(input_ids, labels=target_ids, use_cache=False)

                if i == args.calibration_steps - 1:
                    break

        if args.calibration_steps > 0:
            calibrate(model)
            for module in model.modules():
                if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
                    module.disable_observer()

        if args.convert_model:
            model = convert_pt2e(model)

        model.graph.print_tabular()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    result = evaluate_wikitext(
        model, encodings.input_ids, max_length=args.max_length,
        stride=args.stride, device=device, max_eval_samples=args.max_eval_samples,
    )

    print(f"[INFO] model:      {args.model_id}")
    print(f"[INFO] max length: {args.max_length}")
    print(f"[INFO] stride:     {args.stride}")
    print(f"[INFO] perplexity: {result['perplexity']}")
    print(f"[INFO] scored:     {result['scored_tokens']} tokens, {result['windows']} windows")

    if not args.eager and args.record_histogram and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_histogram(model, args.output_dir)
        plot_layer_range(model, args.output_dir)
    return result


if __name__ == "__main__":
    args = parse_args()
    with_execution_context(main)(args)
