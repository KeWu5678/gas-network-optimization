import numpy as np

from gasnetopt.core.adm import Solution
from gasnetopt.results_io import load_results, update_results


def test_update_and_load_roundtrip(tmp_path):
    path = tmp_path / 'sub' / 'results'
    solution = Solution(alpha=np.eye(2), u=np.ones((2, 1)), objective=1.5,
                        w=np.arange(3.), timings={'poc_nlp': 0.1})
    update_results(path, T=26., poc=solution)
    update_results(path, tau_min=1.0)
    db = load_results(path)
    assert db['T'] == 26. and db['tau_min'] == 1.0
    assert db['poc'].objective == 1.5
    assert np.array_equal(db['poc'].alpha, solution.alpha)
