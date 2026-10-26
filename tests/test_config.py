import pytest

from gasnetopt import DATA_DIR, REPO_ROOT, config, experiment
from gasnetopt.gas.experiment import GasExperiment
from gasnetopt.translines.experiment import TranslinesExperiment

CONFIGS = REPO_ROOT / 'configs'

TRANSLINES = '''
application = "translines"
methods = ["sur"]
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
'''


def write(tmp_path, text):
    path = tmp_path / 'experiment.toml'
    path.write_text(text)
    return path


def load(tmp_path, text):
    return experiment.load(write(tmp_path, text))[1]


def test_shipped_configs_load_with_benchmark_values():
    gas = experiment.load(CONFIGS / 'gaslib11_1h.toml')[1]
    assert isinstance(gas, GasExperiment)
    assert gas.methods == ('sur', 'adm_ciap')
    assert (gas.model.horizon, gas.model.nt, gas.model.nx) == (3600., 61, 2)
    assert gas.switching.tau_min == 900.
    assert gas.nlp.tol == 1e-6 and gas.adm.nlp_tol == 1e-8
    assert gas.data.network.parent == DATA_DIR / 'gaslib' / 'GasLib-11'
    assert gas.milp.backend == 'highs'

    lines = experiment.load(CONFIGS / 'translines_coarse.toml')[1]
    assert isinstance(lines, TranslinesExperiment)
    assert lines.methods == ('sur', 'comb', 'adm', 'adm_ciap')
    assert lines.data.network == 'extended-tree'
    assert lines.data.demand.exists()
    assert lines.model.grid == 'coarse'
    assert lines.switching.tau_min == 1.0


def test_output_directory_is_named_after_the_config():
    assert experiment.output_dir(CONFIGS / 'gaslib11_1h.toml') == \
        REPO_ROOT / 'results' / 'gaslib11_1h'


def test_defaults_and_conversions(tmp_path):
    cfg = load(tmp_path, TRANSLINES.format(adm='rho_values = [1, 10]'))
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
        load(tmp_path, TRANSLINES.format(adm=adm))


@pytest.mark.parametrize('change, message', [
    (('[milp]', ''), 'missing key milp'),
    (('methods = ["sur"]', ''), 'missing key methods'),
    (('"sur"', '"sur", "rounding"'), 'unknown method.*rounding'),
    (('"translines"', '"power"'), 'application must be one of'),
    (('extended-tree', 'grid9'), '\\[data\\].*network "grid9"'),
    (('"coarse"', '"medium"'), '\\[model\\].*grid "medium"'),
    (('[milp]', '[milp]\nbackend = "cplex"'), '\\[milp\\].*backend "cplex"'),
    (('[nlp]', '[nlp'), 'invalid TOML'),
])
def test_missing_keys_and_invalid_choices_are_rejected(tmp_path, change,
                                                       message):
    text = TRANSLINES.format(adm='').replace(*change)
    with pytest.raises(config.ConfigError, match=message):
        load(tmp_path, text)


def test_errors_name_the_config_file(tmp_path):
    with pytest.raises(config.ConfigError, match='experiment.toml: '):
        load(tmp_path, TRANSLINES.format(adm='max_itr = 5'))


def test_copy_to_output(tmp_path):
    source = write(tmp_path, TRANSLINES.format(adm=''))
    copied = config.copy_to_output(source, tmp_path / 'run')
    assert copied.read_text() == source.read_text()
