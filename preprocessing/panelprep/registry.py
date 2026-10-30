"""Tiny generic registry used by the score and embedder catalogues.

MODIFY HERE: nothing in this file is domain-specific. To add a metric or an
embedder you decorate a function in ``panelprep/scores/`` or ``panelprep/embed/``;
you never edit this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator


@dataclass
class Entry:
    """One registered callable plus the metadata the reports need."""

    name: str
    fn: Callable
    meta: Dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Entry] = {}

    def register(self, name: str, **meta) -> Callable:
        """Decorator: ``@SCORES.register("avg_kl", orientation="lower_is_better")``."""

        def wrap(fn: Callable) -> Callable:
            if name in self._entries:
                raise ValueError(f"{self.kind} {name!r} already registered")
            self._entries[name] = Entry(name=name, fn=fn, meta=dict(meta))
            fn.registry_name = name
            fn.registry_meta = dict(meta)
            return fn

        return wrap

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Entry:
        if name not in self._entries:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {sorted(self._entries)}")
        return self._entries[name]

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def meta_table(self):
        import pandas as pd

        return pd.DataFrame(
            [{"name": e.name, **e.meta} for e in sorted(self._entries.values(), key=lambda e: e.name)]
        ).set_index("name") if self._entries else pd.DataFrame()
