from __future__ import annotations

import shutil
import tomllib
import types
import typing
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from . import REPO_ROOT
from .core.milp import BACKENDS

# sections shared by every experiment; each application defines its
# experiment schema from these and its own [data] and [model] sections


@dataclass(frozen=True)
class Switching:
    tau_min: float                  # minimal dwell time


@dataclass(frozen=True)
class NlpSettings:
    tol: float = 1e-8               # POC relaxation and rounding reopt.


@dataclass(frozen=True)
class MilpSettings:
    backend: str = 'highs'

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError('backend "{}" is not one of {}'.format(
                self.backend, ', '.join(BACKENDS)))


@dataclass(frozen=True)
class Output:
    dir: Path


class ConfigError(ValueError):
    pass


Experiment = TypeVar('Experiment')


def load(path: str | Path, schema: type[Experiment]) -> Experiment:
    with open(path, 'rb') as f:
        return _build(schema, tomllib.load(f), section='')


def copy_to_output(path: str | Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copyfile(path, output_dir / 'config.toml'))


def _build(schema: Any, table: Any, section: str) -> Any:
    if not isinstance(table, dict):
        raise ConfigError('{} must be a table'.format(_where(section)))
    unknown = sorted(set(table) - {f.name for f in fields(schema)})
    if unknown:
        raise ConfigError('unknown key(s) {} in {}'.format(
            ', '.join(unknown), _where(section)))
    hints = typing.get_type_hints(schema)
    values = {}
    for f in fields(schema):
        key = '{}.{}'.format(section, f.name) if section else f.name
        if f.name in table:
            values[f.name] = _convert(hints[f.name], table[f.name], key)
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError('missing key {}'.format(key))
    try:
        return schema(**values)
    except ValueError as err:
        raise ConfigError('{}: {}'.format(_where(section), err)) from err


def _convert(hint: Any, value: Any, key: str) -> Any:
    if is_dataclass(hint):
        return _build(hint, value, key)
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        (inner,) = [a for a in typing.get_args(hint) if a is not type(None)]
        return _convert(inner, value, key)
    if typing.get_origin(hint) is tuple:
        if not isinstance(value, list):
            raise ConfigError('{}: expected a list, got {!r}'.format(
                key, value))
        item_type = typing.get_args(hint)[0]
        return tuple(_convert(item_type, item, key) for item in value)
    if hint is Path and isinstance(value, str):
        return REPO_ROOT / value
    if isinstance(value, bool) and hint is not bool:
        raise ConfigError('{}: expected {}, got {!r}'.format(
            key, hint.__name__, value))
    if hint is float and isinstance(value, int):
        return float(value)
    if isinstance(value, hint):
        return value
    raise ConfigError('{}: expected {}, got {!r}'.format(
        key, hint.__name__, value))


def _where(section: str) -> str:
    return '[{}]'.format(section) if section else 'the top level'
