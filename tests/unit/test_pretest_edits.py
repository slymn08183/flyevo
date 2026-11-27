"""The hand edits that turn the melanogaster brain into a 'sechellia-fied' one (scripts/)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pretest_shiu import (
    add_partners,
    paired_effect_size,
    population_shift,
    remove_outputs,
)

# neuron 0 → {10, 11}; neuron 1 → {10}; neuron 2 → {0}
EDGES = (
    np.array([0, 0, 1, 2]),
    np.array([10, 11, 10, 0]),
    np.array([5.0, 7.0, 3.0, 9.0], dtype=np.float32),
)
POOL = np.arange(10, 20)


def test_remove_outputs_deletes_only_outgoing_edges() -> None:
    pre, post, w = remove_outputs(EDGES, np.array([0]))
    assert pre.tolist() == [1, 2]
    assert post.tolist() == [10, 0]  # the edge *into* neuron 0 survives
    assert w.tolist() == [3.0, 9.0]


def test_add_partners_multiplies_partner_count_with_resampled_weights() -> None:
    pre, post, w = add_partners(EDGES, 0, POOL, factor=3.0, rng=np.random.default_rng(0))
    mine = pre == 0
    assert mine.sum() == 6  # 2 partners x 3
    assert len(set(post[mine].tolist())) == 6  # all distinct, none duplicated
    assert set(post[mine].tolist()) <= set(POOL.tolist())
    assert set(w[mine].tolist()) <= {5.0, 7.0}
    assert (pre == 1).sum() == 1  # other neurons untouched


def test_add_partners_is_capped_by_the_pool_and_skips_neurons_without_partners() -> None:
    pre, _, _ = add_partners(EDGES, 0, POOL, factor=100.0, rng=np.random.default_rng(0))
    assert (pre == 0).sum() == len(POOL)
    assert add_partners(EDGES, 2, POOL, factor=5.0, rng=np.random.default_rng(0)) is EDGES


def test_effect_sizes() -> None:
    assert paired_effect_size(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0
    assert paired_effect_size(np.array([1, 2, 3]), np.array([3, 4, 5])) == np.inf
    assert paired_effect_size(np.array([10, 10, 10, 10]), np.array([12, 11, 13, 12])) > 1.0
    same = np.random.default_rng(0).normal(0, 1, (8, 5))
    assert population_shift(same, same) == 0.0
    assert population_shift(same, same + 10.0) > 3.0
