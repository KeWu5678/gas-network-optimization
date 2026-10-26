import pytest

from gasnetopt import REPO_ROOT, cli, experiment

TINY = '''
application = "gas"
methods = ["sur"]
[data]
network = "tests/fixtures/tiny.net"
boundary = "tests/fixtures/tiny.bcd"
[model]
horizon = 600.0
nt = 42
nx = 2
[switching]
tau_min = 120.0
[nlp]
tol = 1e-6
[adm]
[milp]
'''


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'RESULTS_DIR', tmp_path / 'results')
    path = tmp_path / 'tiny.toml'
    path.write_text(TINY)
    return path


def test_run_writes_config_log_results_and_figures(tiny_config, capsys):
    assert cli.main(['run', str(tiny_config)]) == 0
    out_dir = experiment.output_dir(tiny_config)
    assert out_dir.name == 'tiny'
    assert {path.name for path in out_dir.iterdir()} == {
        'config.toml', 'run.log', 'results.pkl', 'poc.png', 'sur.png'}
    assert (out_dir / 'run.log').read_text() == capsys.readouterr().out


def test_table_and_present_read_a_finished_run(tiny_config, capsys):
    cli.main(['run', str(tiny_config)])
    out_dir = experiment.output_dir(tiny_config)
    capsys.readouterr()

    assert cli.main(['table', str(out_dir)]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].startswith('| method | objective')
    assert [row.split(' | ')[0] for row in table[2:]] == [
        '| POC relaxation', '| Sum-Up Rounding + reoptimization']

    assert cli.main(['present', str(out_dir), '--format', 'png']) == 0
    assert 'poc.png, sur.png' in capsys.readouterr().out


def test_topology_draws_gas_networks_only(tiny_config, tmp_path, capsys):
    assert cli.main(['topology', str(tiny_config)]) == 0
    assert (experiment.output_dir(tiny_config) / 'topology.png').exists()

    translines = tmp_path / 'lines.toml'
    translines.write_text(
        (REPO_ROOT / 'configs' / 'translines_coarse.toml')
        .read_text())
    assert cli.main(['topology', str(translines)]) == 2
    assert 'no network drawing' in capsys.readouterr().err


def test_failed_rerun_keeps_the_previous_run(tiny_config):
    cli.main(['run', str(tiny_config)])
    out_dir = experiment.output_dir(tiny_config)
    before = {path.name: path.read_bytes() for path in out_dir.iterdir()}
    tiny_config.write_text(TINY.replace('tiny.bcd', 'missing.bcd'))
    assert cli.main(['run', str(tiny_config)]) == 2
    assert {path.name: path.read_bytes()
            for path in out_dir.iterdir()} == before


def test_saved_config_reruns_into_its_own_directory(tiny_config):
    cli.main(['run', str(tiny_config)])
    saved = experiment.output_dir(tiny_config) / 'config.toml'
    assert experiment.output_dir(saved) == saved.parent
    assert cli.main(['run', str(saved)]) == 0
    assert cli.main(['run', str(saved)]) == 0
    assert saved.read_text() == TINY
    assert [path.name for path in experiment.RESULTS_DIR.iterdir()] == [
        'tiny']


def test_invalid_config_exits_with_one_line_error(tiny_config, capsys):
    tiny_config.write_text(TINY.replace('[adm]', '[adm]\nmax_itr = 5'))
    assert cli.main(['run', str(tiny_config)]) == 2
    err = capsys.readouterr().err
    assert err.startswith('error: ') and 'max_itr' in err
    assert not experiment.output_dir(tiny_config).exists()
