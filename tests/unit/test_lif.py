"""LIF simulator on tiny hand-built networks: every rule of the reference model, one test each."""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from flyevo.models.lif import (
    LIFParams,
    init_state,
    lif_step,
    make_network,
    make_stimulus,
    simulate,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "model" / "lif.yaml"
STRONG = 300.0  # synapse count large enough for one input neuron to drive its target


@pytest.fixture(scope="module")
def shipped() -> LIFParams:
    return LIFParams.from_config(OmegaConf.to_container(OmegaConf.load(CONFIG)))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def params(shipped: LIFParams) -> LIFParams:
    """Shipped model constants; budgets shrunk to fit the tiny test networks (speed only)."""
    return dataclasses.replace(shipped, max_spikes_per_step=64, edge_budget=2048)


def _chain(weight: float, n: int = 2):  # type: ignore[no-untyped-def]
    """0 → 1 with the given weight."""
    return make_network(np.array([0]), np.array([1]), np.array([weight]), n_neurons=n)


def test_config_matches_reference_defaults(shipped: LIFParams) -> None:
    assert (shipped.v_rest, shipped.v_thresh, shipped.tau_m, shipped.tau_syn) == (-52, -45, 20, 5)
    assert (shipped.w_syn, shipped.poisson_weight_factor) == (0.275, 250)
    assert (shipped.delay_steps, shipped.refrac_steps) == (18, 22)


def test_silent_without_input(params: LIFParams) -> None:
    rng = np.random.default_rng(0)
    net = make_network(
        rng.integers(0, 50, 400), rng.integers(0, 50, 400), rng.normal(0, 50, 400), n_neurons=50
    )
    out = simulate(jax.random.key(0), net, make_stimulus(50), params, t_run_ms=100, bin_ms=10)
    assert int(out.spike_counts.sum()) == 0
    assert out.binned.shape == (10, 50)


def test_single_psp_matches_analytic_solution(params: LIFParams) -> None:
    net, stim = _chain(1.0), make_stimulus(2)
    state = init_state(2, params)
    # a spike of neuron 0 emitted delay_steps ago arrives at neuron 1 in the first step
    state = state._replace(spike_buf=state.spike_buf.at[0, 0].set(True))
    step = jax.jit(lif_step, static_argnames="params")
    trace = []
    for _ in range(600):
        state, _ = step(state, jax.random.key(0), net, stim, params)
        trace.append(float(state.v[1]))
    tm, ts = params.tau_m, params.tau_syn
    t_peak = math.log(tm / ts) * tm * ts / (tm - ts)
    peak = params.w_syn * ts / (tm - ts) * (math.exp(-t_peak / tm) - math.exp(-t_peak / ts))
    assert max(trace) - params.v_rest == pytest.approx(peak, rel=1e-3)


def test_poisson_drive_fires_at_requested_rate(params: LIFParams) -> None:
    stim = make_stimulus(2, stimulated={0: 150.0})
    out = simulate(jax.random.key(1), _chain(0.0), stim, params, t_run_ms=1000, bin_ms=10)
    assert 110 < int(out.spike_counts[0]) < 190
    assert int(out.spike_counts[1]) == 0


def test_excitation_drives_inhibition_does_not(params: LIFParams) -> None:
    stim = make_stimulus(2, stimulated={0: 150.0})
    kw = {"t_run_ms": 500, "bin_ms": 10}
    assert int(simulate(jax.random.key(2), _chain(+STRONG), stim, params, **kw).spike_counts[1]) > 0
    assert (
        int(simulate(jax.random.key(2), _chain(-STRONG), stim, params, **kw).spike_counts[1]) == 0
    )


def test_synaptic_delay_is_respected(params: LIFParams) -> None:
    stim = make_stimulus(2, stimulated={0: 150.0})
    out = simulate(jax.random.key(3), _chain(+STRONG), stim, params, t_run_ms=200, bin_ms=params.dt)
    raster = np.asarray(out.binned)  # one bin per step → [T, N] raster
    first_pre, first_post = (int(np.argmax(raster[:, i] > 0)) for i in (0, 1))
    assert raster[:, 1].any()
    assert first_post - first_pre > params.delay_steps


def test_refractory_period_caps_the_rate(params: LIFParams) -> None:
    n_in = 5
    net = make_network(
        np.arange(n_in), np.full(n_in, n_in), np.full(n_in, 3000.0), n_neurons=n_in + 1
    )
    stim = make_stimulus(n_in + 1, stimulated=dict.fromkeys(range(n_in), 150.0))
    out = simulate(jax.random.key(4), net, stim, params, t_run_ms=500, bin_ms=10)
    ceiling = 500 / (params.t_refrac + params.dt)
    assert 100 < int(out.spike_counts[n_in]) <= ceiling


def test_silenced_neuron_spikes_but_does_not_transmit(params: LIFParams) -> None:
    stim = make_stimulus(2, stimulated={0: 150.0}, silenced=[0])
    out = simulate(jax.random.key(5), _chain(+STRONG), stim, params, t_run_ms=300, bin_ms=10)
    assert int(out.spike_counts[0]) > 0
    assert int(out.spike_counts[1]) == 0


def test_deterministic_for_a_key_and_vmappable_over_trials(params: LIFParams) -> None:
    net, stim = _chain(+STRONG), make_stimulus(2, stimulated={0: 150.0})

    def run(key: jax.Array) -> jax.Array:
        return simulate(key, net, stim, params, t_run_ms=300, bin_ms=10).binned

    a, b, c = run(jax.random.key(7)), run(jax.random.key(7)), run(jax.random.key(8))
    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(a, c)
    trials = jax.vmap(run)(jax.random.split(jax.random.key(9), 3))
    assert trials.shape == (3, 30, 2)


def test_make_network_validates_and_sorts() -> None:
    with pytest.raises(ValueError, match="out of range"):
        make_network(np.array([0]), np.array([5]), np.array([1.0]), n_neurons=3)
    net = make_network(np.array([2, 0, 2]), np.array([1, 2, 0]), np.ones(3), n_neurons=3)
    assert np.asarray(net.pre).tolist() == [0, 2, 2]
    assert np.asarray(net.post).tolist() == [2, 1, 0]
    assert np.asarray(net.row_ptr).tolist() == [0, 1, 1, 3]


def test_event_driven_and_dense_paths_agree(params: LIFParams) -> None:
    """Budgets are a speed knob only: forcing the dense fallback must not change one spike."""
    n, e = 200, 6000
    rng = np.random.default_rng(1)
    weight = rng.integers(-40, 120, e).astype(np.float32)  # integers → sums exact in any order
    net = make_network(rng.integers(0, n, e), rng.integers(0, n, e), weight, n_neurons=n)
    stim = make_stimulus(n, stimulated=dict.fromkeys(range(30), 150.0))
    forced_dense = dataclasses.replace(params, max_spikes_per_step=1, edge_budget=1)
    kw = {"t_run_ms": 200, "bin_ms": 10}
    fast = simulate(jax.random.key(11), net, stim, params, **kw).binned
    slow = simulate(jax.random.key(11), net, stim, forced_dense, **kw).binned
    assert int(fast[:, 30:].sum()) > 0  # activity really propagates beyond the stimulated set
    assert jnp.array_equal(fast, slow)
