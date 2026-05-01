"""jax_experiments package.

Side effect: install a compatibility shim for older flax (0.10.x) that lacks
nnx.List. We provide a Module-based wrapper that exposes a `.layers` tuple
and is properly tracked by nnx state introspection (unlike a plain Python
list, which nnx skips).
"""
from __future__ import annotations


def _install_nnx_list_shim() -> None:
    try:
        import flax.nnx as nnx
    except Exception:
        return  # flax/nnx not available; nothing to do
    if hasattr(nnx, "List"):
        return  # newer flax already has it

    class _List(nnx.Module):
        """Tracked list of nnx Modules.

        Stores children as enumerated attributes so flax.nnx state
        introspection traverses them. Iteration and indexing work like
        a plain list.
        """

        def __init__(self, items):
            super().__init__()
            self._items = list(items)
            for i, item in enumerate(self._items):
                setattr(self, f"_item_{i}", item)

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    nnx.List = _List


_install_nnx_list_shim()
