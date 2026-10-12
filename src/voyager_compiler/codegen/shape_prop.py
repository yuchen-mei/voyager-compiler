import logging
from typing import Dict, List, Optional

import torch
from torch.fx.graph import map_arg
from torch.fx.node import Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._subclasses.fake_tensor import FakeTensorMode

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

    def __init__(self, mod, mode: Optional[FakeTensorMode] = None):
        """Retain the graph/modules and optional fake mode for compiler passes."""
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self._mode = mode

    def propagate(self, *args):
        """Refresh every node from positional inputs and return the graph output.

        Compiler passes supply flattened example inputs; an optional fake mode
        avoids materializing graph intermediates while running the same operators.
        """
        if self._mode is not None:
            with self._mode:
                return self._propagate(*args)
        else:
            return self._propagate(*args)

    def _propagate(self, *args):
        """Execute nodes in order and refresh execution metadata in place.

        Called only by ``propagate`` inside its selected execution mode. Existing
        single-tensor metadata is refreshed without retaining tensor data. Remove
        it for tuple/list and non-tensor results; their values and explicit logical
        dtype annotations remain the compiler's source of element information.
        Return the graph's output after releasing expired interpreter values.
        """
        args_iter = iter(args)
        env : Dict[str, Node] = {}

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
                elif node.op == 'call_function':
                    result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                elif node.op == 'call_method':
                    self_obj, *args = load_arg(node.args)
                    kwargs = load_arg(node.kwargs)
                    result = getattr(self_obj, node.target)(*args, **kwargs)
                elif node.op == 'call_module':
                    result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
                elif node.op == 'output':
                    result = load_arg(node.args[0])
            except:
                self.graph.print_tabular()
                print(f"Error in node {node}")
                raise

            if isinstance(result, torch.Tensor):
                if node.meta.get("tensor_meta") is not None:
                    node.meta["tensor_meta"] = _extract_tensor_metadata(result)
            else:
                node.meta.pop("tensor_meta", None)

            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.value = result.cpu().clone()
            elif isinstance(result, (tuple, list)):
                node.shape = tuple(x.shape if x is not None else None for x in result)
                node.value = tuple(x.cpu().clone() if x is not None else None for x in result)
            else:
                node.value = result
                logger.info(f"Node {node} produced non-tensor output {result}")

            env[node.name] = result

            delete_unused_values(node)

        return load_arg(list(self.graph.nodes)[-1])
