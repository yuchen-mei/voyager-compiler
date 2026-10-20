import logging
import math
import operator
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch.fx import GraphModule, Node

from .utils import get_arg_value, _pair
from ..banking import require_allocation
from ..mapping import duplicate_shared_nodes
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_fully_connected,
    is_indexing_or_concatenation_op,
    is_linear,
    is_matmul,
    is_nop,
    is_reshape_op,
)
from ...pt2e_utils import deduplicate_nodes, fetch_attr, propagate_shape
from ...quantize_pt2e import create_getattr_from_value

logger = logging.getLogger(__name__)

__all__ = [
    "eliminate_reshape_with_no_effect",
    "transpose_conv2d_inputs_and_weights",
    "transpose_linear_weights",
]

TRANSPOSED_OPERATORS = {
    torch.ops.aten.conv1d.default: torch.ops.quantized_ops.conv1d.default,
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d.default,
    torch.ops.aten.max_pool2d.default: torch.ops.quantized_ops.max_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default: torch.ops.quantized_ops.adaptive_avg_pool2d.default,
    torch.ops.quantized_ops.conv2d_mx.default: torch.ops.quantized_ops.conv2d_mx.default,
}

AXES_ARG_INDEX_MAP = {
    torch.ops.quantized_ops.calculate_mx_qparam.default: 1,
    torch.ops.quantized_ops.dequantize.default: 3,
    torch.ops.quantized_ops.quantize.default: 3,
    torch.ops.quantized_ops.quantize_mx.default: 2,
}

NCL_TO_NLC = (0, 2, 1)
NCHW_TO_NHWC = (0, 2, 3, 1)
NHWC_TO_NCHW = (0, 3, 1, 2)
WEIGHT_NCHW_TO_HWIO = (2, 3, 1, 0)


def conv2d_transposed(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    output = torch.ops.aten.conv2d.default(
        input.permute(0, 3, 1, 2),
        weight.permute(3, 2, 0, 1) if groups == 1 else weight,
        bias,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        groups,
    )
    return output.permute(0, 2, 3, 1)


def extract_conv2d_graph(
    model: GraphModule, start: Node, visited: Set[Node]
) -> List[Node]:
    """
    Depth-first worklist traversal over both consumers (users) and
    producers (input nodes), restricted to reshape/elementwise/transpose
    /indexing/quantization ops that are considered fusable. Conv1D uses the
    same traversal to choose rank-three starting groups. Its separate forward
    pass then tracks rank changes and permutations beyond these groups. Keep
    convolution parameter paths out of the activation-layout traversal.
    """
    channel_last_1d = start.target == torch.ops.aten.conv1d.default
    quantized_lib = torch.ops.quantized_ops

    ALLOW_LIST_OPS = {
        torch.ops.aten.pad.default,
        quantized_lib.calculate_mx_qparam.default,
        quantized_lib.quantize_mx.default,
    }

    def should_traverse(node: Node) -> bool:
        # Keep rank-changing operations and Conv1D layout boundaries outside.
        if channel_last_1d:
            return len(getattr(node, "shape", ())) == 3 and (
                node.target in (
                    torch.ops.aten.conv1d.default,
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.cat.default,
                    torch.ops.aten._to_copy.default,
                    torch.ops.aten.to.dtype,
                    torch.ops.aten.clone.default,
                    torch.ops.aten.contiguous.default,
                    torch.ops.aten.detach.default,
                )
                or node.target in ALLOW_LIST_OPS
                or is_elementwise_op(node)
            )

        # Cannot fuse stack because it creates a new dimension
        if node.target == torch.ops.aten.stack.default:
            return False

        # Only include reshape if the input is a 4D tensor
        if is_reshape_op(node):
            return len(node.shape) == 4

        if node.target == operator.getitem:
            src = node.args[0]
            return src.target == quantized_lib.quantize_mx.default

        return (
            node.target in TRANSPOSED_OPERATORS
            or node.target in ALLOW_LIST_OPS
            or is_elementwise_op(node)
            or is_indexing_or_concatenation_op(node)
        )

    stack = [start]
    nodes_in_graph = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes_in_graph.add(node)

        inputs = node.all_input_nodes
        if channel_last_1d and node.target == torch.ops.aten.conv1d.default:
            inputs = [node.args[0]]
        adjacent_nodes = list(node.users) + list(inputs)

        for n in adjacent_nodes:
            if (
                channel_last_1d and n.target == torch.ops.aten.conv1d.default
                and n.args[0] is not node
            ):
                continue
            if n not in visited and should_traverse(n):
                stack.append(n)

    node_to_idx = {n: i for i, n in enumerate(model.graph.nodes)}
    return sorted(nodes_in_graph, key=lambda n: node_to_idx[n])


def remap_pad_after_permute(
    pad: Tuple[int, ...], dims: Tuple[int, ...], ndim: int
) -> Tuple[int, ...]:
    """
    Remap padding after permuting a tensor.

    Args:
        pad: Original pad tuple as in torch.nn.functional.pad (starts from last dim).
        dims: Permutation dimensions.
        ndim: Number of dimensions in the original tensor.

    Returns:
        Tuple[int, ...]: New pad tuple corresponding to permuted tensor.
    """
    # number of padded dimensions
    k = len(pad) // 2
    assert k <= ndim, "Pad dimensions exceed tensor dimensions"

    # original padded dims (from last to first)
    original_padded_dims = list(range(ndim - k, ndim))

    dim_to_new_index = {d: dims.index(d) for d in range(ndim)}

    new_pad_pairs = {i: (0, 0) for i in range(ndim)}

    # Assign padding for dimensions that were originally padded
    for i, orig_dim in enumerate(reversed(original_padded_dims)):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        new_pad_pairs[dim_to_new_index[orig_dim]] = (left, right)

    # Collect pads in reverse order (last-first)
    new_pad = []
    for i in sorted(new_pad_pairs.keys(), reverse=True):
        new_pad.extend(new_pad_pairs[i])

    return tuple(new_pad)


def _get_path_to_conv2d(node: torch.fx.Node):
    for user in node.users:
        if is_conv2d(user):
            return [node, user]

        if (
            is_nop(user) or is_indexing_or_concatenation_op(user)
            or user.target in [
                torch.ops.quantized_ops.quantize.default,
                torch.ops.aten.pad.default,
            ]
        ):
            path = _get_path_to_conv2d(user)
            if path is not None:
                return [node] + path
    return None


def _process_conv2d_input_nodes(
    node: Node, model: GraphModule, island_set: Set[Node]
):
    """Convert a Conv2D group's external input, leaving other users unchanged.

    The convolution layout pass calls this for incoming parameters and tensors.
    Standard convolution weights use HWIO; eligible activations use NHWC.
    Depthwise parameters retain their existing interpretation.
    """
    graph = model.graph
    path = _get_path_to_conv2d(node)

    # Case A: Input is a weight (Parameter) or weight scale
    if node.op == "get_attr" and path is not None:
        conv2d_node = path[-1]
        if is_depthwise_conv(conv2d_node) or path[-2] not in (
            conv2d_node.args[1], conv2d_node.kwargs.get("weight_scale")
        ):
            return

        logger.debug(f"Permuting parameter {node}")
        param = fetch_attr(model, node.target)
        param.data = param.data.permute(2, 3, 1, 0)

        node.meta["dims"] = WEIGHT_NCHW_TO_HWIO

    # Case B: Input is a node flow from outside the island
    if node.op != "get_attr" and len(node.shape) == 4:
        is_weight_node = (
            path is not None and id(path[-2]) == id(path[-1].args[1])
        )
        dims = WEIGHT_NCHW_TO_HWIO if is_weight_node else NCHW_TO_NHWC

        logger.debug(f"Insert permute after {node} with dims {dims}")
        with graph.inserting_after(node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default, (node, dims),
            )

        permute_node.meta["dims"] = dims
        permute_node.meta["dtype"] = node.meta.get("dtype")

        for user in list(node.users.keys()):
            if user in island_set:
                user.replace_input_with(node, permute_node)


def _rewrite_node_args_for_layout(node: Node, input_dims=None) -> None:
    """Update one island node's axis arguments and operator for its input layout.

    The layout pass calls this in graph order after converting incoming edges.
    Conv1D supplies its per-node input permutation explicitly; Conv2D infers it
    from incoming edges. A scalar first operand has no layout; use the next
    tensor operand in that case. Preserve quantization meaning by remapping its
    axes with slice axes, including default and keyword axis arguments.
    """
    if input_dims is None:
        input_dims = next(
            (n.meta["dims"] for n in node.all_input_nodes if "dims" in n.meta), None
        )
    node.meta["dims"] = input_dims

    args = tuple(node.args)

    if node.target == torch.ops.aten.pad.default:
        pad = remap_pad_after_permute(args[1], input_dims, node.value.ndim)
        node.update_arg(1, pad)

    if is_indexing_or_concatenation_op(node):
        dim = get_arg_value(node, 1, "dim", 0)
        if dim < 0:
            dim = dim + len(input_dims)
        if len(node.args) > 1:
            node.update_arg(1, input_dims.index(dim))
        else:
            node.update_kwarg("dim", input_dims.index(dim))

    if is_reshape_op(node):
        if node.target == torch.ops.aten.transpose.int:
            dims = (args[1], args[2])
        else:
            dims = args[1]
        dims = [d + max(input_dims) + 1 if d < 0 else d for d in dims]
        dims = tuple(input_dims.index(d) for d in dims)
        node.update_arg(1, dims)

    idx = AXES_ARG_INDEX_MAP.get(node.target)
    axes = get_arg_value(node, idx, "axes") if idx is not None else None
    if axes is not None:
        axes = [a + len(input_dims) if a < 0 else a for a in axes]
        axes = tuple(input_dims.index(a) for a in axes)
        if idx < len(args):
            node.update_arg(idx, axes)
        else:
            node.update_kwarg("axes", axes)

    if node.target in TRANSPOSED_OPERATORS:
        node.target = TRANSPOSED_OPERATORS[node.target]
        node.meta["transposed"] = True


def transpose_conv1d_activations(model: GraphModule):
    """Keep Conv1D activations channel-last through supported tensor operations.

    The convolution layout pass calls this after shape propagation. Existing
    rank-three convolution groups seed NCL-to-NLC conversion; downstream nodes
    carry their own physical-to-original axis permutation, even after axes are
    added or removed. Multi-output operators record one permutation per tensor.
    Each input edge is aligned separately, preserving broadcasting, parameters,
    lookup tables, and consumers outside the converted graph. Unsupported
    operations and model outputs receive their original axis order.

    Only transpose/permute nodes become temporary aliases: their complete effect
    is carried in the output permutation and materialized at boundaries when
    needed. Slice, concatenation, casts, and quantization retain their operations.
    Arbitrary reshapes remain boundaries because they can change element order.
    Return the processed nodes, including boundary conversions, so the following
    Conv2D traversal treats their restored outputs as new external inputs.
    """
    graph = model.graph
    nodes = list(graph.nodes)
    seeds = set()
    for node in nodes:
        if node.target == torch.ops.aten.conv1d.default and node not in seeds:
            extract_conv2d_graph(model, node, seeds)
    if not seeds:
        return set()

    aten, quantized = torch.ops.aten, torch.ops.quantized_ops
    copies = (aten._to_copy.default, aten.to.dtype, aten.clone.default,
              aten.contiguous.default, aten.detach.default)
    squeezes = (aten.squeeze.default, aten.squeeze.dim, aten.squeeze.dims)
    splits = (aten.split.Tensor, aten.split_with_sizes.default, aten.chunk.default)
    quantization = (quantized.quantize.default, quantized.dequantize.default)
    data_only = (aten.conv1d.default, quantized.calculate_mx_qparam.default,
                 quantized.quantize_mx.default, quantized.vmap.default)
    layouts, output_layouts = {}, {}
    converted, aliases = set(), []
    layout_required = set()
    for node in reversed(nodes):
        if (node.target == aten.conv1d.default or is_reshape_op(node)
                or any(user in layout_required for user in node.users)):
            layout_required.add(node)

    for node in nodes:
        target = node.target
        if (target == operator.getitem and node.args[0] in output_layouts
                and isinstance(node.args[1], int)):
            dims = output_layouts[node.args[0]][node.args[1]]
            layouts[node] = node.meta["dims"] = dims
            converted.add(node)
            continue

        supported = (
            target in copies + squeezes + splits + data_only
            or target in (aten.select.int, aten.unsqueeze.default, aten.stack.default,
                          aten.cat.default, aten.slice.Tensor, aten.unbind.int)
            or is_reshape_op(node)
            or is_elementwise_op(node) and target != aten.nonzero.default
            or target == aten.pad.default and get_arg_value(node, 2, "mode", "constant") == "constant"
        )
        if not supported or getattr(getattr(target, "_schema", None), "is_mutable", False):
            continue
        # If every tuple element is already required in its original order,
        # restore the input once instead of adding a conversion per output.
        if isinstance(getattr(node, "value", None), (tuple, list)):
            exposed = set()
            for user in node.users:
                if user.op == "output":
                    exposed.update(range(len(node.value)))
                elif user.target == operator.getitem and isinstance(user.args[1], int):
                    if any(consumer.op == "output" for consumer in user.users):
                        exposed.add(user.args[1] % len(node.value))
            if len(exposed) == len(node.value):
                continue
        if target in data_only + quantization:
            # Only data, scale, and zero point share activation axes. Lookup
            # tables and convolution parameters keep their own interpretation.
            count = 3 if target in quantization else 1
            inputs = [
                get_arg_value(node, index, argument.name)
                for index, argument in enumerate(target._schema.arguments[:count])
            ]
            inputs = [n for n in inputs if isinstance(n, Node)]
        else:
            inputs = list(node.all_input_nodes)
        if node not in seeds and not any(n in layouts for n in inputs):
            continue
        # An input already returned in its original layout can also serve this
        # branch. Keep propagation where a later convolution or model permute
        # can still benefit; otherwise reuse that one required restoration.
        if node not in seeds and node not in layout_required and any(
            n in layouts and any(user.op == "output" for user in n.users)
            for n in inputs
        ):
            continue

        # Broadcasting aligns logical axes on the right before any permutation.
        # Choose a full-size activation's layout before a scalar or scale input.
        rank = len(inputs[0].shape)
        if is_elementwise_op(node) and target not in quantization + (quantized.vmap.default,):
            rank = len(node.shape)
        active = [n for n in inputs if n in layouts and len(n.shape) <= rank]
        if target == aten.conv1d.default:
            dims = NCL_TO_NLC
        elif active:
            source = max(active, key=lambda n: len(n.shape))
            leading = rank - len(source.shape)
            dims = tuple(range(leading)) + tuple(d + leading for d in layouts[source])
        else:
            dims = NCL_TO_NLC if rank == 3 else tuple(range(rank))

        # At a multi-input model output, compare moving the inputs with moving
        # the result. This avoids converting an external operand only to convert
        # the combined result back. Size-one axes move no elements.
        if len(inputs) > 1 and all(user.op == "output" for user in node.users):
            costs = {}
            for candidate in (dims, tuple(range(rank))):
                cost = 0
                for operand in inputs:
                    leading = rank - len(operand.shape)
                    shape = (1,) * leading + tuple(operand.shape)
                    current = layouts.get(operand, tuple(range(len(operand.shape))))
                    current = tuple(range(leading)) + tuple(d + leading for d in current)
                    if [d for d in current if shape[d] > 1] != [d for d in candidate if shape[d] > 1]:
                        cost += operand.value.numel() * operand.value.element_size()
                output_dims = list(candidate)
                if target == aten.stack.default:
                    axis = get_arg_value(node, 1, "dim", 0) % (rank + 1)
                    output_dims = [d + (d >= axis) for d in candidate]
                    output_dims.insert(axis, axis)
                if [d for d in output_dims if node.shape[d] > 1] != [d for d in range(len(node.shape)) if node.shape[d] > 1]:
                    cost += node.value.numel() * node.value.element_size()
                costs[candidate] = cost
            dims = min(costs, key=costs.get)

        for operand in inputs:
            if not len(operand.shape):
                continue
            original_dims = layouts.get(operand, tuple(range(len(operand.shape))))
            leading = rank - len(original_dims)
            aligned_dims = tuple(range(leading)) + tuple(d + leading for d in original_dims)
            permutation = tuple(aligned_dims.index(d) for d in dims)
            if permutation == tuple(range(rank)):
                continue
            with graph.inserting_before(node):
                aligned = operand
                if leading:
                    shape = (1,) * leading + tuple(operand.shape[d] for d in original_dims)
                    aligned = graph.call_function(aten.reshape.default, (operand, shape))
                    converted.add(aligned)
                permuted = graph.call_function(aten.permute.default, (aligned, permutation))
            permuted.meta["dtype"] = operand.meta.get("dtype")
            converted.add(permuted)
            node.replace_input_with(operand, permuted)

        result_dims = dims
        if target in (aten.transpose.int, aten.permute.default):
            if target == aten.transpose.int:
                permutation = list(range(rank))
                if rank:
                    first = get_arg_value(node, 1, "dim0") % rank
                    second = get_arg_value(node, 2, "dim1") % rank
                    permutation[first], permutation[second] = permutation[second], permutation[first]
            else:
                permutation = [d % rank for d in get_arg_value(node, 1, "dims")]
            result_dims = tuple(permutation.index(d) for d in dims)
            operand = get_arg_value(node, 0, target._schema.arguments[0].name)
            node.target, node.args, node.kwargs = aten.alias.default, (operand,), {}
            aliases.append(node)
        elif target in (aten.unsqueeze.default, aten.stack.default):
            axis = get_arg_value(node, 1, "dim", 0) % (rank + 1)
            result_dims = [d + (d >= axis) for d in dims]
            result_dims.insert(axis, axis)
            result_dims = tuple(result_dims)
        elif target in squeezes + (aten.select.int, aten.unbind.int):
            if not rank:
                # Squeezing a scalar keeps it scalar, including dim=0 or -1.
                removed = []
            elif target == aten.squeeze.default:
                removed = [d for d, size in enumerate(inputs[0].shape) if size == 1]
            else:
                axis = get_arg_value(node, 1, "dim", 0)
                axes = axis if target == aten.squeeze.dims else [axis]
                axes = [d % rank for d in axes]
                removed = [d for d in axes if target not in squeezes or inputs[0].shape[d] == 1]
                mapped = [dims.index(d) for d in axes]
                mapped = mapped if target == aten.squeeze.dims else mapped[0]
                if len(node.args) > 1:
                    node.update_arg(1, mapped)
                else:
                    node.update_kwarg("dim", mapped)
            result_dims = tuple(d - sum(r < d for r in removed) for d in dims if d not in removed)
        elif target in splits:
            axis = get_arg_value(node, 2, "dim", 0) % rank
            if len(node.args) > 2:
                node.update_arg(2, dims.index(axis))
            else:
                node.update_kwarg("dim", dims.index(axis))
        else:
            _rewrite_node_args_for_layout(node, dims)

        converted.add(node)
        if isinstance(node.value, (tuple, list)):
            output_layouts[node] = tuple(result_dims for _ in node.value)
            node.meta["output_dims"] = output_layouts[node]
        else:
            layouts[node] = node.meta["dims"] = result_dims

    # Restore only edges leaving the converted graph. A tuple must be unpacked
    # first; its data and scales can have different shapes and logical dtypes.
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    for node in nodes:
        if node not in layouts and node not in output_layouts:
            continue
        replacement = None
        for user in list(node.users):
            if user in converted:
                continue
            if replacement is not None:
                user.args = torch.fx.map_arg(user.args, lambda n: replacement if n is node else n)
                user.kwargs = torch.fx.map_arg(user.kwargs, lambda n: replacement if n is node else n)
                continue
            # One restored value can serve every original-layout consumer. Put
            # it before the first consumer in graph order to preserve dominance.
            first_user = min((n for n in node.users if n not in converted), key=node_order.get)
            with graph.inserting_before(first_user):
                restored = []
                permutations = output_layouts[node] if node in output_layouts else (layouts[node],)
                for index, dims in enumerate(permutations):
                    value = node
                    dtype = node.meta.get("dtype")
                    if node in output_layouts:
                        value = graph.call_function(operator.getitem, (node, index))
                        converted.add(value)
                        dtype = dtype[index] if isinstance(dtype, (tuple, list)) else None
                        value.meta["dtype"] = dtype
                    if dims != tuple(range(len(dims))):
                        inverse = tuple(dims.index(d) for d in range(len(dims)))
                        value = graph.call_function(aten.permute.default, (value, inverse))
                        converted.add(value)
                        value.meta["dtype"] = dtype
                    value.shape = node.shape[index] if node in output_layouts else node.shape
                    restored.append(value)
            replacement = restored[0]
            if node in output_layouts:
                # ShapeProp stores lists as tuples; preserve the operator's
                # actual return type when a consumer receives the whole result.
                returns = node.target._schema.returns
                replacement = restored if len(returns) == 1 and isinstance(returns[0].type, torch.ListType) else tuple(restored)
            user.args = torch.fx.map_arg(user.args, lambda n: replacement if n is node else n)
            user.kwargs = torch.fx.map_arg(user.kwargs, lambda n: replacement if n is node else n)

    for node in reversed(aliases):
        node.replace_all_uses_with(node.args[0])
        graph.erase_node(node)
        converted.remove(node)
    # These maps describe the Conv1D rewrite, not the NHWC conversion that a
    # later Conv2D may request. Boundary shapes are back in the original order.
    for node in converted:
        node.meta.pop("dims", None)
    return converted


def transpose_conv2d_inputs_and_weights(model: GraphModule):
    """Convert convolution islands to channel-last layout during compilation.

    Preserve the existing NHWC/HWIO Conv2D path and use NLC activations with
    unchanged weights for Conv1D. Conv1D tracks rank changes and tuple outputs
    separately; Conv2D retains its existing four-dimensional groups. The following
    reshape-elimination pass removes conversions that cancel existing ones.
    """
    graph = model.graph
    visited_nodes = transpose_conv1d_activations(model)

    torch.nn.functional.conv2d = conv2d_transposed

    for node in list(graph.nodes):
        if node in visited_nodes or node.target not in TRANSPOSED_OPERATORS:
            continue

        # Extract the cluster of nodes that can share channel-last activations.
        island_nodes = extract_conv2d_graph(model, node, visited_nodes)
        island_set = set(island_nodes)

        for node_to_treat in island_nodes:
            # Convert incoming activation edges from their original layout.
            for input_node in list(node_to_treat.all_input_nodes):
                if input_node in island_set or "dims" in input_node.meta:
                    continue

                _process_conv2d_input_nodes(input_node, model, island_set)

            for user in list(node_to_treat.users.keys()):
                if user in island_set or "dims" in user.meta:
                    continue

                logger.debug(f"Insert permute before {user} with dims {NHWC_TO_NCHW}")
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default,
                        (node_to_treat, NHWC_TO_NCHW),
                    )
                permute_node.meta["dtype"] = node_to_treat.meta.get("dtype")
                user.replace_input_with(node_to_treat, permute_node)

            _rewrite_node_args_for_layout(node_to_treat)

            def permute(t, dims):
                return tuple(t[i] for i in dims)

            tiled_shapes = node_to_treat.meta.get("tiled_shapes")
            if is_conv2d(node_to_treat) and tiled_shapes is not None:
                for key, arg in [
                    ("input", node_to_treat.args[0]),
                    ("weight", node_to_treat.args[1])
                ]:
                    input_dims = arg.meta["dims"]
                    tiled_shapes[key] = permute(tiled_shapes[key], input_dims)

                    scale_key = f"{key}_scale"
                    if scale_key in tiled_shapes:
                        tiled_shapes[scale_key] = permute(
                            tiled_shapes[scale_key], input_dims
                        )

                tiled_shapes["output"] = permute(tiled_shapes["output"], NCHW_TO_NHWC)

                tiling = node_to_treat.meta["l2_tiling"]
                node_to_treat.meta["l2_tiling"] = permute(tiling, NCHW_TO_NHWC)

                if stride := node_to_treat.meta.get("tile_strides"):
                    stride["input"] = permute(stride["input"], NCHW_TO_NHWC)
                    stride["input_scale"] = permute(stride["input_scale"], NCHW_TO_NHWC)
                    node_to_treat.meta["tile_strides"] = stride

    graph.lint()
    model.recompile()
    return model


def eliminate_reshape_with_no_effect(model: GraphModule):
    """Replace cancelling layout chains with a reshape after shape propagation.

    Preserve conversions that change dtype even when indexed element order is
    unchanged; removing such a cast would alter subsequent numerical operations.
    The compiler invokes this after its convolution and linear layout passes.
    """
    deleted_nodes = set()
    for node in list(model.graph.nodes):
        if not is_reshape_op(node) or node in deleted_nodes:
            continue

        curr_node = node
        input_node = node.all_input_nodes[0]

        group = []
        while (
            len(curr_node.users) == 1
            and (is_reshape_op(curr_node) or is_nop(curr_node))
            # A cast can preserve indexed element order while changing values.
            and curr_node.value.dtype == curr_node.args[0].value.dtype
        ):
            group.append(curr_node)
            curr_node = next(iter(curr_node.users))

        val = input_node.value
        orig_x = torch.arange(val.numel(), dtype=torch.int32)

        x = orig_x.reshape(val.shape)

        last_valid_idx = -1

        for i, gn in enumerate(group):
            args = torch.fx.graph.map_arg(gn.args, lambda n: x)
            x = gn.target(*args)

            if torch.equal(x.reshape(-1), orig_x):
                last_valid_idx = i

        del group[last_valid_idx + 1:]

        if not group:
            continue

        logger.debug(f"Eliminating reshape group: {[n.name for n in group]}")

        output_shape = group[-1].value.shape

        with model.graph.inserting_before(node):
            reshape_node = model.graph.call_function(
                torch.ops.aten.reshape.default,
                (input_node, output_shape),
            )

        propagate_shape(reshape_node)

        group[-1].replace_all_uses_with(reshape_node)

        for n in reversed(group):
            model.graph.erase_node(n)
            deleted_nodes.add(n)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def make_linear_wrapper(transpose=False, skip_fc=False):
    """
    Returns a function that wraps torch.nn.functional.linear with optional
    weight transposition.
    """
    def wrapped_linear(input, weight, bias=None):
        is_fc = all(dim == 1 for dim in input.shape[:-1])
        do_transpose = transpose and not (skip_fc and is_fc)
        return torch.ops.aten.linear.default(
            input, weight.T if do_transpose else weight, bias
        )
    return wrapped_linear


def make_matmul_wrapper(transpose=False, skip_fc=False):
    """
    Returns a function that wraps torch.matmul with optional transposition of
    the second argument.
    """
    def wrapped_matmul(input, other):
        input_shape = input.shape
        other_shape = other.shape

        is_bmm = len(input_shape) > 2 or len(other_shape) > 2
        if is_bmm:
            is_fc = input_shape[-2] == 1
        else:
            is_fc = all(s == 1 for s in input_shape[:-1])

        do_transpose = transpose and not (skip_fc and is_fc)

        return torch.ops.aten.matmul.default(
            input, other if do_transpose else other.transpose(-2, -1)
        )
    return wrapped_matmul


ALLOWED_UPSTREAM_OPS: Set[any] = {
    torch.ops.aten.select.int,
    torch.ops.quantized_ops.calculate_mx_qparam.default,
    torch.ops.quantized_ops.dequantize.default,
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.quantize_mx.default,
}


def is_transpose_2d(node: torch.fx.Node) -> bool:
    """Checks if node is a transpose on the last two dimensions."""
    if node.target != torch.ops.aten.transpose.int:
        return False
    # Check args to ensure it is specifically swapping -2 and -1
    # args format: (input, dim0, dim1)
    rank = len(node.shape)
    dims = set(d if d >= 0 else rank + d for d in node.args[1:])
    return dims == {rank - 2, rank - 1}


def find_upstream_transpose_or_param(
    node: torch.fx.Node,
    *,
    max_depth: int = 16,
) -> Optional[List[torch.fx.Node]]:
    """
    Starting from a transpose node, walks upstream through a list
    of allowed operations until reaching another transpose node
    or a constant param node.
    """
    if not is_transpose_2d(node):
        return None

    def dfs(curr: Node, depth: int) -> Optional[List[Node]]:
        if is_transpose_2d(curr):
            return [curr]

        if curr.op == "get_attr" and require_allocation(curr):
            return [curr]

        if depth > max_depth:
            return None

        path = []
        if curr.target in ALLOWED_UPSTREAM_OPS or is_nop(curr):
            for inp in curr.all_input_nodes:
                path.extend(dfs(inp, depth + 1) or [])

        return [curr] + path if path else None

    if (found_path := dfs(node.args[0], 0)) is None:
        return None

    return list(set([node] + found_path))


def _insert_transposed_input(arg: Node, model: GraphModule):
    with model.graph.inserting_after(arg):
        if arg.op == "get_attr":
            value = fetch_attr(model, arg.target)
            transposed = create_getattr_from_value(
                model, model.graph, arg.name + "_T", value.mT
            )
        else:
            transposed = model.graph.call_function(
                torch.ops.aten.transpose.int, (arg, -2, -1)
            )
    transposed.meta["dtype"] = arg.meta.get("dtype")
    return transposed


def _rank(n: Node) -> int:
    return len(n.shape)


def _fix_axes_after_transpose(node: Node) -> List[int]:
    if (index := AXES_ARG_INDEX_MAP.get(node.target)) is None:
        return

    axes = get_arg_value(node, index, "axes")
    rank = _rank(node)

    # Build forward and inverse permutation for transpose(-2, -1)
    perm = list(range(rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    inv_perm = [perm.index(i) for i in range(rank)]

    # Normalize negative axes first
    norm_axes = [(a + rank) % rank for a in axes]

    # Apply inverse permutation
    new_axes = tuple(inv_perm[a] for a in norm_axes)
    node.args = node.args[:index] + (new_axes,) + node.args[index + 1:]


def _fuse_quantize_mx_last_axis(model: GraphModule):
    """
    Replace calculate_mx_qparam + quantize with quantize_mx when the
    quantization is performed along the last axis.
    """
    graph = model.graph
    for node in list(graph.nodes):
        if node.target != torch.ops.quantized_ops.calculate_mx_qparam.default:
            continue

        axes = get_arg_value(node, 1, "axes")
        rank = _rank(node)
        if axes != (rank - 1,) and axes != (-1,):
            continue

        args = node.args[1:] + (None,) * (5 - len(node.args[1:]))

        quantize_node = next(iter(
            n for n in node.users
            if n.target == torch.ops.quantized_ops.quantize.default
        ))

        assert quantize_node.args[0] == node.args[0], "Unexpected quantize input"

        qmap = quantize_node.args[5]
        output_code = get_arg_value(quantize_node, 6, "output_code")
        new_code = None

        with graph.inserting_before(node):
            new_qmap = graph.node_copy(qmap)
            if output_code is not None:
                new_code = graph.node_copy(output_code)
            quantize_mx_node = graph.call_function(
                torch.ops.quantized_ops.quantize_mx.default,
                (node.args[0], new_qmap) + args + (new_code,)
            )
            scale_node = graph.call_function(
                operator.getitem, (quantize_mx_node, 0)
            )
            output_node = graph.call_function(
                operator.getitem, (quantize_mx_node, 1)
            )

        propagate_shape(new_qmap, model)
        if new_code is not None:
            propagate_shape(new_code, model)
        propagate_shape(quantize_mx_node, model)
        propagate_shape(scale_node, model)
        propagate_shape(output_node, model)

        scale_node.meta["dtype"] = node.meta.get("dtype")
        output_node.meta["dtype"] = quantize_node.meta.get("dtype")
        quantize_mx_node.meta["dtype"] = (
            scale_node.meta.get("dtype"), output_node.meta.get("dtype")
        )

        node.replace_all_uses_with(scale_node)
        quantize_node.replace_all_uses_with(output_node)

        logger.info(f"Replaced {node} and {quantize_node} with {quantize_mx_node}")

    graph.lint()
    model.recompile()
    return model


def eliminate_canceling_transposes(
    model: GraphModule, chain: List[Node], transposed_nodes: Dict[Node, Node] = None
) -> bool:
    """
    Optimizes a chain like [select_3, select_2, quantize_default_1, transpose_3]
    when there's a matching matmul-side transpose (user of chain[0]).

    Steps:
      1. Check if the two transposes cancel (considering selects).
      2. If yes, detach intermediate nodes and remove the redundant transpose.

    Returns:
        bool: True if optimization was applied, else False.
    """
    graph = model.graph

    if transposed_nodes is None:
        transposed_nodes = {}

    chain = [n for n in chain if n.op == "call_function"]
    if not chain or len(chain) < 2:
        return False

    up_t = chain[0]
    down_t = chain[-1]

    if up_t.target != torch.ops.aten.transpose.int:
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if _rank(up_t) < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the upstream transpose node
    chain = duplicate_shared_nodes(graph, chain[1:])

    # Rewrite graph to remove cancelling transposes
    for n in chain:
        for arg in n.all_input_nodes:
            if arg == up_t:
                n.replace_input_with(up_t, up_t.args[0])
                continue
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _fix_axes_after_transpose(n)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not up_t.users:
        graph.erase_node(up_t)

    logger.info(f"Eliminated redundant transposes: {up_t} and {down_t}")

    return True


def move_transpose_before_dq(
    model: GraphModule, chain: List[Node], transposed_nodes: Dict[Node, Node] = None
) -> bool:
    """
    Optimizes a chain like [dequantize_default, select_3, select_2, transpose_3].

    Steps:
      1. Check if there's a dequantize operation in the chain.
      2. If yes, move the transpose before the dequantize.

    Returns:
        bool: True if optimization was applied, else False.
    """
    graph = model.graph

    if transposed_nodes is None:
        transposed_nodes = {}

    chain = [n for n in chain if n.op == "call_function"]
    for i, n in enumerate(chain):
        if n.target == torch.ops.quantized_ops.dequantize.default:
            break

    chain = chain[i:]  # Keep only from dequantize to end

    if not chain or len(chain) < 2:
        return False

    down_t = chain[-1]

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if any(n.args[1] != 0 for n in selects):
        return False

    chain = duplicate_shared_nodes(graph, chain)
    dequantize_node = chain[0]

    # Insert transpose after dequantize input
    dq_input = dequantize_node.args[0]
    up_t = next((
        n for n in dq_input.users if n.target == torch.ops.aten.transpose.int
    ), None)
    if up_t is not None and up_t.meta.get("dtype") == dq_input.meta.get("dtype"):
        dequantize_node.replace_input_with(dq_input, up_t)
    else:
        with graph.inserting_after(dq_input):
            up_t = graph.call_function(
                torch.ops.aten.transpose.int, (dq_input, -2, -1)
            )
        up_t.meta["dtype"] = dq_input.meta.get("dtype")
        dequantize_node.replace_input_with(dq_input, up_t)
        propagate_shape(up_t)

    for n in chain:
        for arg in n.all_input_nodes:
            if arg in chain or arg.value.ndim < 2 or arg == up_t:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _fix_axes_after_transpose(n)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    logger.info(f"Hoisted {up_t} before {dequantize_node} and removed {down_t}")

    return True


def fold_transpose_into_constant(
    model: GraphModule, chain: List[Node], transposed_nodes: Dict[Node, Node] = None
) -> bool:
    graph = model.graph
    if not chain or len(chain) < 2:
        return False

    if transposed_nodes is None:
        transposed_nodes = {}

    attr_node = chain[0]
    down_t = chain[-1]

    if attr_node.op != "get_attr":
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if _rank(attr_node) < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the transpose node
    chain = duplicate_shared_nodes(model.graph, chain[1:])

    for n in chain:
        for arg in n.all_input_nodes:
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not attr_node.users:
        graph.erase_node(attr_node)

    logger.info(f"Folded {down_t} into constant: {attr_node}")

    return True


def _update_tiled_shapes(node: Node) -> None:
    """Updates the tiled_shapes metadata for a transposed node."""
    if (tiled_shapes := node.meta.get("tiled_shapes")) is None:
        return

    for key in ["weight", "other", "weight_scale"]:
        if key in tiled_shapes:
            d0, d1 = tiled_shapes[key]
            tiled_shapes[key] = (d1, d0)


def _insert_transpose_op(
    model: GraphModule,
    node: Node,
    user: Node,
    transposed_nodes: dict
) -> Optional[List[Node]]:
    """Inserts a transpose operation before the user node."""
    with model.graph.inserting_before(user):
        new_node = model.graph.call_function(
            torch.ops.aten.transpose.int, (node, -2, -1)
        )

    new_node.meta["dtype"] = node.meta.get("dtype")
    propagate_shape(new_node, model)
    user.replace_input_with(node, new_node)

    path = find_upstream_transpose_or_param(new_node)
    if not path:
        return None

    node_order = {n: i for i, n in enumerate(model.graph.nodes)}
    sorted_path = sorted(path, key=lambda n: node_order[n])

    success = eliminate_canceling_transposes(
        model, sorted_path, transposed_nodes
    )
    return None if success else sorted_path


def _process_linear_node(
    model: GraphModule,
    node: Node,
    transpose_weight: bool,
    skip_fc: bool
) -> None:
    """Handles weight mutation for Linear nodes."""
    is_fc = is_fully_connected(node)
    if (is_fc and skip_fc) or (not is_fc and not transpose_weight):
        return

    logger.info(f"Transposing weight for linear node {node.name}")

    weight_node = node.args[1]
    weight = fetch_attr(model, weight_node.target)
    weight.data = weight.data.T

    if (scale_node := node.kwargs.get("weight_scale")) is not None:
        scale = fetch_attr(model, scale_node.target)
        scale.data = scale.data.T

    # Mark spmm_csr users as having a transposed weight
    for user in list(weight_node.users):
        if user.target in [
            torch.ops.quantized_ops.linear_mx.default,
            torch.ops.quantized_ops.spmm_csr.default,
        ]:
            user.kwargs = {**user.kwargs, "weight_transposed": True}

    if node.target == torch.ops.aten.linear.default:
        node.target = torch.ops.quantized_ops.linear.default

    _update_tiled_shapes(node)
    node.meta["transposed"] = True


def _process_matmul_node(
    model: GraphModule,
    node: Node,
    transpose_weight: bool,
    transpose_fc: bool,
    transposed_nodes: dict
) -> None:
    """Handles graph transformation for MatMul nodes."""
    is_fc = is_fully_connected(node)
    # Note: Logic preserved from original (returns if FC and we WANT transpose_fc)
    if (is_fc and transpose_fc) or (not is_fc and transpose_weight):
        return

    logger.info(f"Transposing weight for matmul node {node.name}")

    weight_node = node.args[1]
    path = _insert_transpose_op(model, weight_node, node, transposed_nodes)
    if path is not None:
        move_transpose_before_dq(model, path, transposed_nodes)

    if (scale_node := node.kwargs.get("weight_scale")) is not None:
        path = _insert_transpose_op(model, scale_node, node, transposed_nodes)
        if path is not None:
            fold_transpose_into_constant(model, path, transposed_nodes)

    if node.target == torch.ops.aten.matmul.default:
        node.target = torch.ops.quantized_ops.matmul.default

    _update_tiled_shapes(node)
    node.meta["transposed"] = True


def transpose_linear_weights(
    model: GraphModule, transpose_weight: bool, transpose_fc: bool = False
) -> GraphModule:
    """
    Transpose the weights of linear layers in the given FX graph module.
    """
    skip_fc = not transpose_fc

    torch.nn.functional.linear = make_linear_wrapper(transpose_weight, skip_fc)
    torch.matmul = make_matmul_wrapper(transpose_weight, skip_fc)

    transposed_nodes = {}

    for node in list(model.graph.nodes):
        if is_linear(node):
            _process_linear_node(model, node, transpose_weight, skip_fc)
        elif is_matmul(node):
            _process_matmul_node(
                model, node, transpose_weight, transpose_fc, transposed_nodes
            )

    deduplicate_nodes(model)
    _fuse_quantize_mx_last_axis(model)

    model.graph.lint()
    model.recompile()
    return model
