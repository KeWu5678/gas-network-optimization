import pytest

from gasnetopt import DATA_DIR, REPO_ROOT, config
from gasnetopt.gas.experiment import GasExperiment
from gasnetopt.translines.experiment import TranslinesExperiment

CONFIGS = REPO_ROOT / 'configs'

TRANSLINES = '''
[data]
network = "extended-tree"
demand = "data/translines/demand_extended_tree_coarse.dat"
[model]
grid = "coarse"
[switching]
tau_min = 1
[nlp]
[adm]
{adm}
[milp]
[output]
dir = "results/translines"
'''


def write(tmp_path, text):
    path = tmp_path / 'experiment.toml'
    path.write_text(text)
    return path


def test_shipped_configs_load_with_benchmark_values():
    gas = config.load(CONFIGS / 'gaslib11_1h.toml', GasExperiment)
    assert (gas.model.horizon, gas.model.nt, gas.model.nx) == (3600., 61, 2)
    assert gas.switching.tau_min == 900.
    assert gas.nlp.tol == 1e-6 and gas.adm.nlp_tol == 1e-8
    assert gas.data.network.parent == DATA_DIR / 'gaslib' / 'GasLib-11'
    assert gas.milp.backend == 'highs'

    lines = config.load(CONFIGS / 'translines_coarse.toml',
                        TranslinesExperiment)
    assert lines.data.network == 'extended-tree'
    assert lines.data.demand.exists()
    assert lines.model.grid == 'coarse'
    assert lines.switching.tau_min == 1.0
    assert lines.output.dir == REPO_ROOT / 'results' / 'translines'


def test_defaults_and_conversions(tmp_path):
    cfg = config.load(write(tmp_path, TRANSLINES.format(
        adm='rho_values = [1, 10]')), TranslinesExperiment)
    assert cfg.switching.tau_min == 1.0 and isinstance(
        cfg.switching.tau_min, float)
    assert cfg.adm.rho_values == (1.0, 10.0)
    assert cfg.adm.max_iter == 100 and cfg.adm.time_budget is None
    assert cfg.nlp.tol == 1e-8 and cfg.milp.backend == 'highs'


@pytest.mark.parametrize('adm, message', [
    ('max_itr = 5', 'unknown key.*max_itr.*\\[adm\\]'),
    ('max_iter = 2.5', 'adm.max_iter: expected int'),
    ('max_iter = true', 'adm.max_iter: expected int'),
    ('rho_values = 1.0', 'adm.rho_values: expected a list'),
])
def test_invalid_values_are_rejected(tmp_path, adm, message):
    with pytest.raises(config.ConfigError, match=message):
        config.load(write(tmp_path, TRANSLINES.format(adm=adm)),
                    TranslinesExperiment)


@pytest.mark.parametrize('change, message', [
    (('dir = ', '# dir = '), 'missing key output.dir'),
    (('[milp]', ''), 'missing key milp'),
    (('extended-tree', 'grid9'), '\\[data\\].*network "grid9"'),
    (('"coarse"', '"medium"'), '\\[model\\].*grid "medium"'),
    (('[milp]', '[milp]\nbackend = "cplex"'), '\\[milp\\].*backend "cplex"'),
])
def test_missing_keys_and_invalid_choices_are_rejected(tmp_path, change,
                                                       message):
    text = TRANSLINES.format(adm='').replace(*change)
    with pytest.raises(config.ConfigError, match=message):
        config.load(write(tmp_path, text), TranslinesExperiment)


def test_copy_to_output(tmp_path):
    source = write(tmp_path, TRANSLINES.format(adm=''))
    copied = config.copy_to_output(source, tmp_path / 'run')
    assert copied.read_text() == source.read_text()
