import logging
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from torch.fx.graph import map_arg
from torch.fx.node import Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._pytree import tree_leaves, tree_map

logger = logging.getLogger(__name__)


class ShapeProp:
    """Execute a GraphModule to refresh compiler values and tensor metadata.

    Export, transformation, tiling, and code generation call ``propagate`` with
    real or fake inputs. Each node receives its current ``shape``, ``value``, and
    existing execution ``tensor_meta``. Explicit ``meta['dtype']`` remains the
    logical quantized representation, which can differ from the tensor's storage
    dtype; export's fake ``meta['val']`` and other compiler annotations remain.
    Unannotated nodes stay unannotated so fused placeholders still inherit their
    source tensor's logical dtype instead of shadowing it with a storage dtype.
    """

    def __init__(self, mod, mode: Optional[FakeTensorMode] = None, tensor_files=None):
        """Configure graph propagation or bounded real tensor-file execution.

        Compiler passes provide ``mode`` for shape-only execution. Code emission
        instead supplies ``tensor_files`` mapping resolved protobuf names to file
        paths: execute real inputs, save those values synchronously before later
        mutations, and retain only fake node values after each operation.
        """
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self._mode = mode
        self._tensor_files = tensor_files

    def propagate(self, *args):
        """Refresh every node from positional inputs and return the graph output.

        Compiler passes supply flattened example inputs. An optional fake mode
        converts activation inputs to storage-free tensors; already-fake inputs
        select their own mode. Parameter-only operations still execute normally,
        because quantization and layout passes may inspect their real contents.
        Tensor-free generators run in the selected fake mode. With real inputs
        and no requested fake mode, every operation runs normally.
        """
        fake_mode = next(
            (x.fake_mode for x in tree_leaves(args) if isinstance(x, FakeTensor)),
            self._mode,
        )
        if fake_mode is not None:
            args = tree_map(
                lambda x: fake_mode.from_tensor(x)
                if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor)
                else x,
                args,
            )
        return self._propagate(*args, fake_mode=fake_mode)

    def _propagate(self, *args, fake_mode=None):
        """Execute nodes in order and refresh execution metadata in place.

        Called only by ``propagate`` with its selected generator mode. Existing
        single-tensor metadata is refreshed without retaining tensor data. Remove
        it for tuple/list and non-tensor results; their values and explicit logical
        dtype annotations remain the compiler's source of element information.
        Return the graph's output after releasing expired interpreter values.
        """
        args_iter = iter(args)
        env : Dict[str, Node] = {}
        if self._tensor_files is not None:
            from ..pt2e_utils import FAKE_MODE
            from .mapping_utils import save_tensor
            pending_files = set(self._tensor_files)

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use : Dict[Node, Node] = {}
        user_to_last_uses : Dict[Node, List[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(self.graph.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        def delete_unused_values(user : Node):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            nodes_to_delete = user_to_last_uses.get(user, [])
            for n in nodes_to_delete:
                env.pop(n.name)

        for node in self.graph.nodes:
            try:
                if node.op == 'placeholder':
                    result = next(args_iter)
                elif node.op == 'get_attr':
                    result = fetch_attr(node.target)
                elif node.op in ('call_function', 'call_method', 'call_module'):
                    node_args, node_kwargs = load_arg(node.args), load_arg(node.kwargs)
                    factory_mode = (
                        fake_mode if not any(
                            isinstance(x, torch.Tensor)
                            for x in tree_leaves((node_args, node_kwargs))
                        ) else None
                    )
                    if node.op == 'call_function':
                        target = node.target
                    elif node.op == 'call_method':
                        self_obj, *node_args = node_args
                        target = getattr(self_obj, node.target)
                    else:
                        target = self.modules[node.target]
                    with factory_mode if factory_mode is not None else nullcontext():
                        result = target(*node_args, **node_kwargs)
                elif node.op == 'output':
                    result = load_arg(node.args[0])
            except:
                self.graph.print_tabular()
                print(f"Error in node {node}")
                raise

            if self._tensor_files is not None:
                tensors = (
                    [(f"{node.name}_{i}", value) for i, value in enumerate(result)]
                    if isinstance(result, (tuple, list))
                    else [(node.name, result)]
                )
                for name, value in tensors:
                    if name in pending_files:
                        save_tensor(value, self._tensor_files[name], synchronous=True)
                        pending_files.remove(name)

            if isinstance(result, torch.Tensor):
                if node.meta.get("tensor_meta") is not None:
                    node.meta["tensor_meta"] = _extract_tensor_metadata(result)
            else:
                node.meta.pop("tensor_meta", None)

            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                if self._tensor_files is not None:
                    node.value = FAKE_MODE.from_tensor(result)
                else:
                    node.value = (
                        result if isinstance(result, FakeTensor)
                        else result.cpu().clone()
                    )
            elif isinstance(result, (tuple, list)):
                node.shape = tuple(
                    x.shape if isinstance(x, torch.Tensor) else None for x in result
                )
                node.value = tuple(
                    (
                        FAKE_MODE.from_tensor(x) if self._tensor_files is not None
                        else x if isinstance(x, FakeTensor) else x.cpu().clone()
                    )
                    if isinstance(x, torch.Tensor) else x
                    for x in result
                )
            else:
                node.value = result
                logger.info(f"Node {node} produced non-tensor output {result}")

            env[node.name] = result

            delete_unused_values(node)

        if self._tensor_files is not None and pending_files:
            raise ValueError(
                f"[ERROR] Tensor dump nodes were not executed: {sorted(pending_files)}"
            )
        return load_arg(list(self.graph.nodes)[-1])
