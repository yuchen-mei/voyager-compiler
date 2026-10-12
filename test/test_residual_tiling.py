"""Numerical coverage of residual handling during contraction-dimension tiling.

Run with ``python -m unittest discover -s test -p test_residual_tiling.py``
from the voyager-compiler checkout. These small CPU graphs exercise export,
shape propagation, tiling, and the real operator-fusion pass without codegen.
"""

import unittest

import torch
from torch import nn

from voyager_compiler import OpMatcher, export_model, fuse
from voyager_compiler.codegen.passes.tiling import (
    _build_gemm_shape_map,
    split_gemm_node,
)
from voyager_compiler.codegen.shape_prop import ShapeProp
from voyager_compiler.fake_quantize import get_quantization_map


class ResidualProjection(nn.Module):
    """Small projection graphs covering valid and misleading residual paths."""

    def __init__(self, scenario, dtype=torch.float32):
        """Build deterministic projection weights for the named test scenario.

        The unittest cases pass ``scenario`` to select the consumer graph and
        ``dtype`` to exercise FP32 or BF16 contraction arithmetic. Quantization
        buffers are registered only for the real custom-operator case.
        """
        super().__init__()
        self.scenario = scenario
        width = 12 if scenario == "split_bias" else 8
        self.projection = nn.Linear(8, width, bias=False, dtype=dtype)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.arange(width * 8).reshape(width, 8).remainder(17) / 16 - 0.5
            )
        self.bias = nn.Parameter(torch.tensor([0.5, -0.25], dtype=dtype))
        if scenario == "quantize":
            self.register_buffer("scale", torch.tensor(0.25))
            self.register_buffer("qmap", get_quantization_map("int8"))

    def forward(self, value, residual):
        """Return tensors exposing the selected projection-consumer semantics.

        ``torch.export`` traces this framework entry point with independent
        projection and residual inputs. Every case returns a tuple so tests
        compare all branches, including the two untouched Nemotron split outputs.
        """
        projected = self.projection(value)
        if self.scenario == "split_bias":
            gate, states, time_step = projected.split([4, 6, 2], dim=-1)
            return gate, states, time_step + self.bias
        if self.scenario == "nonlinear":
            return (torch.relu(projected) + residual,)
        if self.scenario == "shared":
            return projected, projected + residual
        if self.scenario == "late_residual":
            return (projected + torch.sin(residual),)
        if self.scenario == "alpha":
            return (torch.add(projected, residual, alpha=2),)
        if self.scenario == "reversed":
            return (residual + projected,)
        if self.scenario == "quantize":
            return (
                torch.ops.quantized_ops.quantize.default(
                    projected + residual, self.scale, qmap=self.qmap
                ),
            )
        return (projected + residual,)


class ResidualTilingTest(unittest.TestCase):
    def test_direct_residual_retains_per_tile_fusion(self):
        """Preserve direct residual values and fuse every C tile with its add.

        Both operand orders are exported and split into two C tiles, then run
        through Voyager's actual matcher and fusion pass. Numerical comparisons
        cover the tiled and fused graph; BF16 tolerates reduction reassociation.
        """
        for dtype in (torch.float32, torch.bfloat16):
            for scenario in ("direct", "reversed"):
                with self.subTest(dtype=dtype, scenario=scenario), torch.no_grad():
                    value = (torch.arange(24).reshape(1, 3, 8) / 16 - 0.5).to(dtype)
                    residual = value.flip(-1) / 2
                    inputs = (value, residual)
                    graph = export_model(ResidualProjection(scenario, dtype), inputs)
                    expected = graph(*inputs)
                    ShapeProp(graph).propagate(*inputs)
                    projection = next(
                        node
                        for node in graph.graph.nodes
                        if node.target == torch.ops.aten.linear.default
                    )
                    tile_sizes = (3, 4, 8)
                    split_gemm_node(
                        graph,
                        projection,
                        tile_sizes,
                        _build_gemm_shape_map(projection, tile_sizes),
                    )
                    graph.graph.eliminate_dead_code()
                    graph.graph.lint()
                    graph.recompile()
                    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-6
                    torch.testing.assert_close(
                        graph(*inputs), expected, rtol=tolerance, atol=tolerance
                    )
                    fuse(graph, [[OpMatcher("linear"), OpMatcher("add")]], inputs)
                    graph.graph.lint()
                    torch.testing.assert_close(
                        graph(*inputs), expected, rtol=tolerance, atol=tolerance
                    )
                    fused_calls = [
                        node for node in graph.graph.nodes if node.op == "call_module"
                    ]
                    self.assertEqual(len(fused_calls), 2)
                    self.assertFalse(
                        any(
                            node.target == torch.ops.aten.add.Tensor
                            for node in graph.graph.nodes
                        )
                    )
                    for call in fused_calls:
                        targets = {
                            node.target
                            for node in graph.get_submodule(call.target).graph.nodes
                        }
                        self.assertIn(torch.ops.aten.linear.default, targets)
                        self.assertIn(torch.ops.aten.add.Tensor, targets)

    def test_other_consumers_preserve_outputs_and_tail_operations(self):
        """Tile realistic non-residual paths without relocating their tail add.

        This includes Nemotron's split/bias path, nonlinear and shared outputs,
        late inputs that cannot dominate the inserted seed, scaled additions,
        physical or declared dtype changes, shape-expanding additions, and a
        real quantize consumer. Each
        exported graph must remain topologically valid and numerically match
        before and after ordinary GEMM/add fusion.
        """
        for scenario in (
            "split_bias",
            "nonlinear",
            "shared",
            "late_residual",
            "alpha",
            "promoted_dtype",
            "declared_dtype",
            "expanded_output",
            "quantize",
        ):
            with self.subTest(scenario=scenario), torch.no_grad():
                value = torch.arange(24).reshape(1, 3, 8) / 16 - 0.5
                residual = value.flip(-1) / 2
                if scenario == "promoted_dtype":
                    residual = residual.double()
                elif scenario == "expanded_output":
                    residual = torch.cat((residual, residual + 0.25), dim=0)
                inputs = (value, residual)
                graph = export_model(ResidualProjection(scenario), inputs)
                expected = graph(*inputs)
                ShapeProp(graph).propagate(*inputs)
                projection = next(
                    node
                    for node in graph.graph.nodes
                    if node.target == torch.ops.aten.linear.default
                )
                if scenario == "declared_dtype":
                    # Quantized operators may carry floating tensors while
                    # their public graph boundary declares an integer dtype.
                    projection.meta["dtype"] = "bfloat16"
                    residual_node = next(
                        node
                        for node in graph.graph.nodes
                        if node.op == "placeholder" and node.target == "residual"
                    )
                    residual_node.meta["dtype"] = "int8"
                tail_add = next(
                    node
                    for node in reversed(list(graph.graph.nodes))
                    if node.target == torch.ops.aten.add.Tensor
                )
                tile_sizes = (3, 4, projection.shape[-1])
                split_gemm_node(
                    graph,
                    projection,
                    tile_sizes,
                    _build_gemm_shape_map(projection, tile_sizes),
                )
                graph.graph.eliminate_dead_code()
                graph.graph.lint()
                graph.recompile()
                self.assertIn(tail_add, graph.graph.nodes)
                torch.testing.assert_close(
                    graph(*inputs), expected, rtol=1e-5, atol=1e-6
                )
                fuse(graph, [[OpMatcher("linear"), OpMatcher("add")]], inputs)
                graph.graph.lint()
                torch.testing.assert_close(
                    graph(*inputs), expected, rtol=1e-5, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
