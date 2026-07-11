'''
Result persistence (replaces the platform-dependent shelve files of the
original code with a single portable pickle file).
'''

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_results(path: str | Path, data: dict[str, Any]) -> Path:
    'Save dict of results to a pickle file (creates parent directories).'
    path = Path(path).with_suffix('.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path


def load_results(path: str | Path) -> dict[str, Any]:
    'Load dict of results from a pickle file.'
    path = Path(path).with_suffix('.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def update_results(path: str | Path, **entries: Any) -> Path:
    'Merge entries into an existing (or new) results file.'
    path = Path(path).with_suffix('.pkl')
    data = load_results(path) if path.exists() else {}
    data.update(entries)
    return save_results(path, data)
