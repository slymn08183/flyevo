"""L2: batched leaky integrate-and-fire point neurons on a fixed connectome (pure JAX).

A data-agnostic re-implementation of the whole-brain model of Shiu et al. 2024
(doi:10.1038/s41586-024-07763-9, reference code github.com/philshiu/Drosophila_brain_model,
Brian2, ``method='linear'``). Same equations, same per-step ordering, so that published results
can be reproduced before the model is trusted for anything new:

    dv/dt = (v_rest - v + g) / tau_m        (frozen while refractory)
    dg/dt = -g / tau_syn                    (frozen while refractory)
    spike:  v > v_thresh and not refractory
    on presynaptic spike, after t_delay:    g_post += weight * w_syn
    reset:  v = v_reset, g = 0

Stimulation follows the reference: a stimulated neuron receives Poisson events at ``rate_hz``,
each adding ``w_syn * poisson_weight_factor`` to v (enough to fire it), and has no refractory
period. Silenced neurons still integrate but their spikes are never transmitted.

Per-step order (Brian2 schedule): integrate → threshold → synapses + Poisson → reset.

Spike propagation is event-driven: per step only the outgoing edges of the neurons that spiked
are gathered (CSR by presynaptic neuron, fixed-size budgets for static shapes). If a step exceeds
the budgets it falls back to a dense pass over all edges, so the result is exact either way; the
budgets only affect speed. Under ``vmap`` both paths are evaluated (``cond`` becomes ``select``),
so batch trials with ``jax.lax.map`` when speed matters.

Units: ms, mV, Hz. All parameters come from ``configs/model/lif.yaml``; nothing is hard-coded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LIFParams:
    """Model constants. Hashable on purpose: passed to ``jit`` as a static argument."""

    dt: float  # ms
    v_rest: float  # mV
    v_reset: float  # mV
    v_thresh: float  # mV
    tau_m: float  # ms, membrane
    tau_syn: float  # ms, synaptic
    t_refrac: float  # ms
    t_delay: float  # ms, synaptic delay
    w_syn: float  # mV per synapse
    poisson_weight_factor: float  # Poisson event adds w_syn * factor to v
    max_spikes_per_step: int  # speed only: event-driven path handles up to this many spikes
    edge_budget: int  # speed only: ... and up to this many outgoing edges per step

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if math.isclose(self.tau_m, self.tau_syn):
            raise ValueError("tau_m == tau_syn makes the exact update singular")
        if self.delay_steps < 1:
            raise ValueError("t_delay must be at least one dt")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> LIFParams:
        """Build from the ``model=lif`` Hydra group (extra keys such as ``name`` are ignored)."""
        ints = {"max_spikes_per_step", "edge_budget"}
        kwargs: dict[str, Any] = {
            f: int(cfg[f]) if f in ints else float(cfg[f]) for f in cls.__dataclass_fields__
        }
        return cls(**kwargs)

    @property
    def delay_steps(self) -> int:
        return round(self.t_delay / self.dt)

    @property
    def refrac_steps(self) -> int:
        return round(self.t_refrac / self.dt)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["pre", "post", "weight", "row_ptr"],
    meta_fields=["n_neurons"],
)
@dataclass(frozen=True)
class Network:
    """Fixed connectome as an edge list in CSR order (sorted by ``pre``).

    pre, post: i32[E] neuron indices.  weight: f32[E] signed synapse count
    (positive = excitatory, negative = inhibitory); scaled by ``w_syn`` at run time.
    row_ptr: i32[N+1], edges of presynaptic neuron i are ``row_ptr[i]:row_ptr[i+1]``.
    """

    pre: jax.Array
    post: jax.Array
    weight: jax.Array
    row_ptr: jax.Array
    n_neurons: int


class Stimulus(NamedTuple):
    rate_hz: jax.Array  # f32[N] Poisson rate per neuron; 0 = not stimulated
    silenced: jax.Array  # bool[N] spikes of these neurons are not transmitted


class LIFState(NamedTuple):
    v: jax.Array  # f32[N] mV
    g: jax.Array  # f32[N] mV
    refrac: jax.Array  # i32[N] refractory steps left
    spike_buf: jax.Array  # bool[D, N] ring buffer of transmitted spikes, D = delay_steps
    t: jax.Array  # i32[] step counter


class SimResult(NamedTuple):
    spike_counts: jax.Array  # i32[N] spikes over the whole run
    binned: jax.Array  # i16[n_bins, N] spikes per time bin (for rates and visualization)


def make_network(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n_neurons: int) -> Network:
    """Validate an edge list and move it to the device in CSR order (sorted by ``pre``)."""
    pre, post, weight = np.asarray(pre), np.asarray(post), np.asarray(weight)
    if not (pre.shape == post.shape == weight.shape and pre.ndim == 1):
        raise ValueError("pre, post, weight must be 1-D arrays of equal length")
    if pre.size and (min(pre.min(), post.min()) < 0 or max(pre.max(), post.max()) >= n_neurons):
        raise ValueError("edge index out of range")
    order = np.argsort(pre, kind="stable")
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n_neurons))])
    return Network(
        pre=jnp.asarray(pre[order], dtype=jnp.int32),
        post=jnp.asarray(post[order], dtype=jnp.int32),
        weight=jnp.asarray(weight[order], dtype=jnp.float32),
        row_ptr=jnp.asarray(row_ptr, dtype=jnp.int32),
        n_neurons=int(n_neurons),
    )


def make_stimulus(
    n_neurons: int, stimulated: Mapping[int, float] | None = None, silenced: Any = ()
) -> Stimulus:
    """``stimulated`` maps neuron index → Poisson rate in Hz; ``silenced`` is a list of indices."""
    rate = np.zeros(n_neurons, dtype=np.float32)
    for idx, hz in (stimulated or {}).items():
        rate[idx] = hz
    mask = np.zeros(n_neurons, dtype=bool)
    mask[list(silenced)] = True
    return Stimulus(rate_hz=jnp.asarray(rate), silenced=jnp.asarray(mask))


def init_state(n_neurons: int, params: LIFParams) -> LIFState:
    return LIFState(
        v=jnp.full(n_neurons, params.v_rest, dtype=jnp.float32),
        g=jnp.zeros(n_neurons, dtype=jnp.float32),
        refrac=jnp.zeros(n_neurons, dtype=jnp.int32),
        spike_buf=jnp.zeros((params.delay_steps, n_neurons), dtype=bool),
        t=jnp.zeros((), dtype=jnp.int32),
    )


def _synaptic_input(net: Network, spikes: jax.Array, params: LIFParams) -> jax.Array:
    """Summed presynaptic weight per neuron, f32[N], for the given spikes (bool[N])."""
    n, k_max, budget = net.n_neurons, params.max_spikes_per_step, params.edge_budget

    def dense(_: None) -> jax.Array:
        return jax.ops.segment_sum(net.weight * spikes[net.pre], net.post, num_segments=n)

    def event_driven(_: None) -> jax.Array:
        idx = jnp.nonzero(spikes, size=k_max, fill_value=0)[0]
        starts = net.row_ptr[idx]
        lens = jnp.where(jnp.arange(k_max) < n_spikes, net.row_ptr[idx + 1] - starts, 0)
        ends = jnp.cumsum(lens)
        # slot j of the budget belongs to spike k = first k with ends[k] > j
        j = jnp.arange(budget)
        k = jnp.minimum(jnp.searchsorted(ends, j, side="right"), k_max - 1)
        edge = starts[k] + j - (ends[k] - lens[k])
        used = j < ends[-1]
        edge = jnp.where(used, edge, 0)
        return jax.ops.segment_sum(
            jnp.where(used, net.weight[edge], 0.0), net.post[edge], num_segments=n
        )

    n_spikes = spikes.sum()
    n_edges = jnp.where(spikes, net.row_ptr[1:] - net.row_ptr[:-1], 0).sum()
    fits = (n_spikes <= k_max) & (n_edges <= budget)
    return jax.lax.cond(fits, event_driven, dense, None)


def lif_step(
    state: LIFState, key: jax.Array, net: Network, stim: Stimulus, params: LIFParams
) -> tuple[LIFState, jax.Array]:
    """Advance one ``dt``. Returns the new state and ``spiked`` (bool[N])."""
    p = params
    n = net.n_neurons
    is_refrac = state.refrac > 0

    # 1. integrate: exact solution of the linear system over one dt (Brian2 method='linear')
    a_m = math.exp(-p.dt / p.tau_m)
    a_s = math.exp(-p.dt / p.tau_syn)
    c = p.tau_syn / (p.tau_m - p.tau_syn) * (a_m - a_s)
    u = state.v - p.v_rest
    v = jnp.where(is_refrac, state.v, p.v_rest + u * a_m + state.g * c)
    g = jnp.where(is_refrac, state.g, state.g * a_s)

    # 2. threshold
    spiked = (v > p.v_thresh) & ~is_refrac

    # 3. synapses (spikes emitted delay_steps ago) and Poisson drive
    slot = state.t % p.delay_steps
    g = g + _synaptic_input(net, state.spike_buf[slot], p) * p.w_syn
    events = jax.random.uniform(key, (n,)) < stim.rate_hz * (p.dt * 1e-3)
    v = v + events * (p.w_syn * p.poisson_weight_factor)

    # 4. reset; stimulated neurons have no refractory period (as in the reference)
    v = jnp.where(spiked, p.v_reset, v)
    g = jnp.where(spiked, 0.0, g)
    refrac_after_spike = jnp.where(stim.rate_hz > 0, 0, p.refrac_steps)
    refrac = jnp.where(spiked, refrac_after_spike, jnp.maximum(state.refrac - 1, 0))

    spike_buf = state.spike_buf.at[slot].set(spiked & ~stim.silenced)
    return LIFState(v, g, refrac, spike_buf, state.t + 1), spiked


@partial(jax.jit, static_argnames=("params", "n_bins", "steps_per_bin"))
def _simulate(
    key: jax.Array,
    net: Network,
    stim: Stimulus,
    params: LIFParams,
    n_bins: int,
    steps_per_bin: int,
) -> SimResult:
    def step(state: LIFState, _: None) -> tuple[LIFState, jax.Array]:
        return lif_step(state, jax.random.fold_in(key, state.t), net, stim, params)

    def one_bin(state: LIFState, _: None) -> tuple[LIFState, jax.Array]:
        def acc(carry: tuple[LIFState, jax.Array], _: None) -> tuple[Any, None]:
            s, counts = carry
            s, spiked = step(s, None)
            return (s, counts + spiked.astype(jnp.int16)), None

        zero = jnp.zeros(net.n_neurons, dtype=jnp.int16)
        (state, counts), _ = jax.lax.scan(acc, (state, zero), None, length=steps_per_bin)
        return state, counts

    _, binned = jax.lax.scan(one_bin, init_state(net.n_neurons, params), None, length=n_bins)
    return SimResult(spike_counts=binned.astype(jnp.int32).sum(axis=0), binned=binned)


def simulate(
    key: jax.Array,
    net: Network,
    stim: Stimulus,
    params: LIFParams,
    *,
    t_run_ms: float,
    bin_ms: float,
) -> SimResult:
    """Run one trial of ``t_run_ms`` from rest. Deterministic for a given ``key``.

    ``vmap`` over ``key`` for independent trials. Spikes are returned as counts per ``bin_ms``
    bin (i16[n_bins, N]), never as a full raster: 140k neurons x 10k steps does not fit in memory.
    """
    if stim.rate_hz.shape != (net.n_neurons,) or stim.silenced.shape != (net.n_neurons,):
        raise ValueError("stimulus shape does not match the network")
    steps_per_bin = round(bin_ms / params.dt)
    if steps_per_bin < 1:
        raise ValueError("bin_ms must be at least one dt")
    n_bins = round(t_run_ms / bin_ms)
    return _simulate(key, net, stim, params, n_bins, steps_per_bin)
