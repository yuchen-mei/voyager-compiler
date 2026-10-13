from google.protobuf import text_format
from torch.utils._pytree import tree_flatten

from .codegen import *
from .decomposed import *
from .fake_quantize import *
from .fp8 import *
from .histogram import *
from .llm_utils import *
from .normal_float import *
from .posit import *
from .pt2e_utils import *
from .pt2e_utils import FAKE_MODE
from .qconfig import *
from .quantize import *
from .quantize_pt2e import *
from .quantizer import *
from .training_args import *
from .utils import *


__all__ = [
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationSpec",
    "TorchExportableModuleWithStaticCache",
    "add_qspec_args",
    "convert_and_export_with_split_cache",
    "convert",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "dispatch_model",
    "dtype_byte_size",
    "export_model",
    "fetch_attr",
    "generate",
    "get_aten_graph_module",
    "get_device_map",
    "get_node_name_to_scope",
    "get_qconfig",
    "get_default_quantizer",
    "insert_align_device_nodes",
    "plot_histogram",
    "plot_layer_range",
    "prepare",
    "prepare_pt2e",
    "print_node_scope_tabular",
    "propagate_config",
    "propagate_shape",
    "quantize",
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "quantize_to_nf",
    "quantize_to_posit",
    "replace_softmax",
    "sink_obs_or_fq",
    "swap_llama_attention",
    "with_execution_context",
]

class qscheme: ...

# Defined in voyager_compiler/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
group_wise_affine: qscheme = QScheme.GROUP_WISE_AFFINE


def _get_op_overload(op_name: str):
    all_overloads = []
    for lib in [torch.ops.aten, torch.ops.quantized_ops]:
        # Also check inplace version of the op (e.g., "add_" for "add")
        for name in [op_name, f"{op_name}_"]:
            if (packet := getattr(lib, name, None)) is None:
                continue
            all_overloads.extend([
                getattr(packet, name) for name in packet.overloads()
            ])
    return all_overloads


class OpMatcher:
    targets: Tuple[torch._ops.OpOverload]
    predicate: Optional[Callable[[Node], bool]] = None

    def __init__(self, *ops, predicate=None):
        self.predicate = predicate

        # Resolve symbolic ops
        targets = []
        for op in ops:
            targets.extend(_get_op_overload(op))

        # Freeze resolved targets
        self.targets = tuple(targets)

    def matches(self, node: Node) -> bool:
        if node.target not in self.targets:
            return False

        return self.predicate(node) if self.predicate else True


def fuse(
    model,
    fusion_pattern,
    example_args,
    example_kwargs=None,
    fuse_reshape=True,
    fake_mode=None
):
    if example_kwargs is None:
        example_kwargs = {}

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    fuse_operator(model, fusion_pattern, fuse_reshape)
    return model


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
    unroll_dims=None,
    transform_layout=False,
    transpose_fc=False,
    cache_size=None,
    num_banks=None,
    fuse_operator=True,
    fuse_reshape=True,
    split_spmm=False,
    use_fake_mode=True,
    pad_terminal_output=True,
):
    """Lower a calibrated graph while retaining real sparse capacity analysis.

    Model frontends pass their example inputs and layout/tiling options here.
    Dense activation shapes use fake tensors; parameters and codebooks remain
    real. Legacy sparse tiling reads CSR contents, so sparse graphs keep their
    existing real propagation even when ``use_fake_mode`` is requested.
    """
    if example_kwargs is None:
        example_kwargs = {}

    has_sparse = any(
        node.target in (
            torch.ops.quantized_ops.quantize_mx_outlier.default,
            torch.ops.quantized_ops.filter_outlier.default,
            torch.ops.quantized_ops.slice_csr_tensor.default,
            torch.ops.quantized_ops.spmm_csr.default,
        ) or node.kwargs.get("A_indptr") is not None
        or node.kwargs.get("A_data") is not None
        for module in model.modules() if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    )
    fake_mode = FAKE_MODE if use_fake_mode and not has_sparse else None

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # -------------------------------------------------------------------------
    # 1. Lowering & Decomposition
    # -------------------------------------------------------------------------
    # Break down complex operators (like MultiHeadAttention) into simpler
    # primitives and handle memory copy/concat operations.

    split_multi_head_attention(model)

    # TODO Disabled for large models. This will be removed in the future once
    # we can handle stack/cat using DMA properly.
    if len(model.graph.nodes) < 10000:
        convert_expand_to_memory_copy(model)
        convert_cat_and_stack_as_stack_on_dim0(model)
        convert_cat_with_mismatched_shapes_to_stack(model)

    fuse_quantize_dequantize_with_previous_op(model)

    # -------------------------------------------------------------------------
    # 2. Hardware Alignment (Padding)
    # -------------------------------------------------------------------------
    # Pad dimensions to align with hardware unrolling constraints (SIMD, systolic
    # array dimensions, etc.) to ensure efficient execution.
    if unroll_dims is not None:
        pad_matrix_op_dimensions(
            model, *unroll_dims, pad_terminal_output=pad_terminal_output
        )
        pad_vector_op_dimensions(model, unroll_dims[1])

    # -------------------------------------------------------------------------
    # 3. Matrix Operation Tiling
    # -------------------------------------------------------------------------
    # Apply L2 tiling logic specifically for matrix operations (GEMM/Conv) to
    # optimize for the specific cache size and memory bank configuration.
    if cache_size is not None:
        run_matrix_op_l2_tiling(model, unroll_dims, cache_size, num_banks)

    # -------------------------------------------------------------------------
    # 4. Data Layout Transformation
    # -------------------------------------------------------------------------
    # Transform GEMM and convolution inputs/weights into layouts friendly
    # for systolic-array based hardware (e.g., transposing weights).

    if transform_layout:
        transpose_conv2d_inputs_and_weights(model)

    transpose_linear_weights(model, transform_layout, transpose_fc)

    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # Remove redundant reshapes that have no effect on tensor semantics
    eliminate_reshape_with_no_effect(model)

    # -------------------------------------------------------------------------
    # 5. Vector Operation Tiling
    # -------------------------------------------------------------------------
    # TODO: Used for unit test. Will be removed in the future
    from .codegen.passes.lowering import split_dense_spmm_node

    if split_spmm:
        split_dense_spmm_node(model)

    # Apply L2 tiling logic for vector-based operations.
    if cache_size is not None:
        run_vector_op_l2_tiling(model, unroll_dims[1], cache_size, num_banks)

    # -------------------------------------------------------------------------
    # 6. Operator Fusion
    # -------------------------------------------------------------------------
    # Perform final operator lowering and fuse sequences of operations (e.g.,
    # Conv+ReLU) into single kernels to reduce memory access overhead.

    if fuse_operator:
        fuse(
            model,
            patterns,
            flatten_args,
            fuse_reshape=fuse_reshape,
            fake_mode=fake_mode,
        )

    rename_nodes_with_param_names(model)
    deduplicate_nodes(model)

    return model


@torch.no_grad()
def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    total_memory=None,
    cache_size=None,
    num_banks=None,
    bank_width=None,
    unroll_dims=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    dump_snapshot=False,
    use_fake_mode=True,
):
    """Emit the legacy protobuf using shape-only dense memory planning.

    Frontends supply a lowered graph and real example inputs. Sparse graphs keep
    real CSR capacity analysis; dense graphs propagate fake activations through
    allocation. Code emission materializes requested tensor files in a bounded
    real execution. Allocator history is retained only for ``dump_snapshot``.
    Return the emitted Model protobuf and write the existing output artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    has_sparse = any(
        node.target in (
            torch.ops.quantized_ops.quantize_mx_outlier.default,
            torch.ops.quantized_ops.filter_outlier.default,
            torch.ops.quantized_ops.slice_csr_tensor.default,
            torch.ops.quantized_ops.spmm_csr.default,
        ) or node.kwargs.get("A_indptr") is not None
        or node.kwargs.get("A_data") is not None
        for module in model.modules() if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    )
    use_fake_mode = use_fake_mode and not has_sparse
    ShapeProp(model, mode=FAKE_MODE if use_fake_mode else None).propagate(*flatten_args)

    allocator = MemoryAllocator(total_memory, bank_width=bank_width)
    run_memory_mapping(
        model, allocator, cache_size, num_banks, bank_width, unroll_dims,
        capture_snapshots=dump_snapshot,
    )

    # Experimental feature
    # from voyager_compiler.codegen.lowering.ir import Module
    # from voyager_compiler.codegen.lowering.codegen import generate_proto

    # top_module = Module.convert(model, name="m")
    # print(top_module.format())

    # params = generate_proto(top_module, model, flatten_args)

    # with open(os.path.join(output_dir, 'module.txt'), "w") as f:
    #     f.write(text_format.MessageToString(params))

    if dump_snapshot:
        allocator.dump_snapshots(os.path.join(output_dir, "memory.png"))

    path = os.path.join(output_dir, "tensor_files")
    params = gen_code(
        model, flatten_args, path if dump_tensors else None,
        use_fake_mode=use_fake_mode,
    )

    with open(os.path.join(output_dir, 'model.txt'), "w") as f:
        f.write(text_format.MessageToString(params))

    operations = [
        op.op.name if op.WhichOneof('op_type') == 'op' else op.fused_op.name
        for op in params.ops if op.op.op != 'nop'
    ]

    with open(os.path.join(output_dir, 'layers.txt'), 'w') as f:
        f.write('\n'.join(operations))

    if len(model.graph.nodes) < 10000:
        gen_compute_graph(model, os.path.join(output_dir, output_file))

    return params
