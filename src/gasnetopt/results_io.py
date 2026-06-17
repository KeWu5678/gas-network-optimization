'''
Result persistence (replaces the platform-dependent shelve files of the
original code with a single portable pickle file).
'''

import pickle
from pathlib import Path


def save_results(path, data):
    'Save dict of results to a pickle file (creates parent directories).'
    path = Path(path).with_suffix('.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path


def load_results(path):
    'Load dict of results from a pickle file.'
    path = Path(path).with_suffix('.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def update_results(path, **entries):
    'Merge entries into an existing (or new) results file.'
    path = Path(path).with_suffix('.pkl')
    data = load_results(path) if path.exists() else {}
    data.update(entries)
    return save_results(path, data)
