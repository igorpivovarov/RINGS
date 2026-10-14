"""
RINGS / TEVSER Ω2 prototype
MiniGrid + Brian2 closed-loop SNN controller with ring-state visualization.

Install:
    pip install minigrid gymnasium brian2 matplotlib numpy

Run:
    python rings_minigrid_brian2_prototype.py

Goal:
    Demonstrate a closed-loop agent whose behavior is controlled by persistent
    spiking ring states: Explore, EnergyLow, FoodSeeking, Danger, Escape.

Notes:
    This is a research prototype, not a final optimized controller.
    The important part is the causal loop:
        environment -> sensor pattern -> SNN rings -> action -> environment
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Ball
from minigrid.minigrid_env import MiniGridEnv

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
    Network,
    PoissonGroup,
    ms,
    Hz,
    mV,
    prefs,
)

# Make Brian2 startup faster and less noisy for small experiments.
prefs.codegen.target = "numpy"


# -----------------------------------------------------------------------------
# 1. MiniGrid environment: homeostatic bug world
# -----------------------------------------------------------------------------

class Food(Ball):
    """Food object. Using Ball rendering but semantically it restores energy."""

    def __init__(self):
        super().__init__(color="green")
        self.type = "food"


class Danger(Lava):
    """Danger object. Using Lava rendering but semantically it triggers escape."""

    def __init__(self):
        super().__init__()
        self.type = "danger"


class RingsHomeostasisEnv(MiniGridEnv):
    """
    A tiny homeostatic world.

    The agent loses energy each step.
    Food restores energy.
    Danger is not always terminal; it is used as a salient threat signal.

    Observation returned to the controller is symbolic and low-dimensional:
        energy, food vector, danger vector, wall ahead.
    This keeps the experiment interpretable.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        size: int = 9,
        max_steps: int = 250,
        n_food: int = 3,
        n_danger: int = 2,
        render_mode: str | None = "rgb_array",
    ):
        self.size = size
        self.n_food = n_food
        self.n_danger = n_danger
        self.energy = 1.0
        self.energy_decay = 0.008
        self.energy_gain = 0.55
        self.danger_penalty = 0.25
        self.last_event = "reset"

        mission_space = MissionSpace(mission_func=lambda: "maintain energy and avoid danger")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            render_mode=render_mode,
        )

        # We expose a compact symbolic observation in addition to the usual MiniGrid image.
        self.observation_space = spaces.Dict(
            {
                "image": self.observation_space["image"],
                "direction": spaces.Discrete(4),
                "mission": self.observation_space["mission"],
                "rstate": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            }
        )

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place agent.
        self.place_agent()

        # Place food and danger.
        for _ in range(self.n_food):
            self.place_obj(Food())
        for _ in range(self.n_danger):
            self.place_obj(Danger())

        self.mission = "maintain energy and avoid danger"
        self.energy = 1.0
        self.last_event = "reset"

    def _nearest_obj_vector(self, obj_type: str) -> Tuple[float, float, float]:
        """Return dx, dy, distance_to_nearest normalized to [-1,1] / [0,1]."""
        ax, ay = self.agent_pos
        best = None
        best_d = 1e9
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == obj_type:
                    d = abs(x - ax) + abs(y - ay)
                    if d < best_d:
                        best_d = d
                        best = (x, y)
        if best is None:
            return 0.0, 0.0, 1.0
        dx = (best[0] - ax) / max(1, self.size - 2)
        dy = (best[1] - ay) / max(1, self.size - 2)
        dist = min(1.0, best_d / max(1, 2 * (self.size - 2)))
        return float(dx), float(dy), float(dist)

    def _wall_ahead(self) -> float:
        fwd_pos = self.front_pos
        obj = self.grid.get(*fwd_pos)
        return 1.0 if obj is not None and obj.type == "wall" else 0.0

    def _get_obs(self):
        obs = super().gen_obs()
        fdx, fdy, fdist = self._nearest_obj_vector("food")
        ddx, ddy, ddist = self._nearest_obj_vector("danger")
        obs["rstate"] = np.array(
            [
                self.energy,
                fdx,
                fdy,
                1.0 - fdist,
                ddx,
                ddy,
                1.0 - ddist,
            ],
            dtype=np.float32,
        )
        return obs

       
    def reset2(self, seed):
        obs, info = super().reset(seed=seed)
        
        obs = self._get_obs()
        info.update({"energy": self.energy, "last_event": self.last_event})
        return obs, info
    
    
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        self.energy -= self.energy_decay
        self.last_event = "none"

        # Check current cell after movement.
        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == "food":
            self.energy = min(1.0, self.energy + self.energy_gain)
            self.grid.set(*self.agent_pos, None)
            self.place_obj(Food())
            self.last_event = "food"
            reward += 0.3

        if cell is not None and cell.type == "danger":
            self.energy -= self.danger_penalty
            self.last_event = "danger_contact"
            reward -= 0.5

        if self.energy <= 0.0:
            terminated = True
            reward -= 1.0
            self.last_event = "energy_depleted"

        # Reward is not the main object of the experiment, but helps diagnostics.
        reward += 0.02 * self.energy

        obs = self._get_obs()
        info.update({"energy": self.energy, "last_event": self.last_event})
        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# 2. Brian2 SNN controller with persistent rings
# -----------------------------------------------------------------------------

@dataclass
class RingSpec:
    name: str
    n: int
    start: int
    end: int


class RingsSNNController:
    """
    Ring-state SNN controller.

    Rings:
        Explore      default roaming state
        EnergyLow    internal homeostatic deficit
        FoodSeeking  approach food when hungry / food sensed
        Danger       threat detection
        Escape       active avoidance mode

    The network is intentionally interpretable. The scientific object is not
    maximum RL score but causal, readable, persistent internal state dynamics.
    """

    def __init__(self, ring_size: int = 12, snn_step_ms: int = 30, seed: int = 1):
        np.random.seed(seed)
        random.seed(seed)
        self.ring_size = ring_size
        self.snn_step_ms = snn_step_ms

        names = ["Explore", "EnergyLow", "FoodSeeking", "Danger", "Escape"]
        self.rings: Dict[str, RingSpec] = {}
        start = 0
        for name in names:
            self.rings[name] = RingSpec(name=name, n=ring_size, start=start, end=start + ring_size)
            start += ring_size
        self.n_state = start

        # Inputs: low_energy, food_left, food_right, food_ahead, danger_left, danger_right, danger_ahead, wall_ahead
        self.input_names = [
            "low_energy",
            "food_left",
            "food_right",
            "food_ahead",
            "danger_left",
            "danger_right",
            "danger_ahead",
            "wall_ahead",
        ]
        self.n_inputs = len(self.input_names)
        self.inputs = PoissonGroup(self.n_inputs, rates=np.zeros(self.n_inputs) * Hz)

        eqs = """
        dv/dt = (-v + bias)/tau : volt
        bias : volt
        tau : second
        """
        self.G = NeuronGroup(
            self.n_state,
            model=eqs,
            threshold="v > -50*mV",
            reset="v = -65*mV",
            refractory=3 * ms,
            method="exact",
        )
        self.G.v = -65 * mV
        self.G.tau = 15 * ms
        self.G.bias = -67 * mV

        # Persistent ring connections: each neuron excites the next one.
        self.S_ring = Synapses(self.G, self.G, on_pre="v_post += 7.0*mV")
        for ring in self.rings.values():
            i = np.arange(ring.start, ring.end)
            j = ring.start + (np.arange(ring.n) + 1) % ring.n
            self.S_ring.connect(i=i, j=j)

        # Within-ring recurrent support: sparse second-neighbor excitation.
        self.S_support = Synapses(self.G, self.G, on_pre="v_post += 2.2*mV")
        for ring in self.rings.values():
            i = np.arange(ring.start, ring.end)
            j = ring.start + (np.arange(ring.n) + 2) % ring.n
            self.S_support.connect(i=i, j=j)

        # Cross-ring logic.
        # Danger -> Escape; Escape inhibits FoodSeeking/Explore.
        # EnergyLow -> FoodSeeking. FoodSeeking inhibits Explore.
        self.S_cross_exc = Synapses(self.G, self.G, on_pre="v_post += 4.5*mV")
        self._connect_ring_to_ring(self.S_cross_exc, "Danger", "Escape", probability=0.25)
        self._connect_ring_to_ring(self.S_cross_exc, "EnergyLow", "FoodSeeking", probability=0.25)

        self.S_cross_inh = Synapses(self.G, self.G, on_pre="v_post -= 5.5*mV")
        self._connect_ring_to_ring(self.S_cross_inh, "Escape", "FoodSeeking", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "Escape", "Explore", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "FoodSeeking", "Explore", probability=0.25)
        self._connect_ring_to_ring(self.S_cross_inh, "Danger", "Explore", probability=0.35)

        # Sensory input projections.
        self.S_in = Synapses(self.inputs, self.G, on_pre="v_post += 8.0*mV")
        self._connect_input_to_ring("low_energy", "EnergyLow")
        self._connect_input_to_ring("food_left", "FoodSeeking")
        self._connect_input_to_ring("food_right", "FoodSeeking")
        self._connect_input_to_ring("food_ahead", "FoodSeeking")
        self._connect_input_to_ring("danger_left", "Danger")
        self._connect_input_to_ring("danger_right", "Danger")
        self._connect_input_to_ring("danger_ahead", "Danger")
        self._connect_input_to_ring("wall_ahead", "Explore")

        self.spikes = SpikeMonitor(self.G)
        self.state_mon = StateMonitor(self.G, "v", record=True)
        self.net = Network(
            self.inputs,
            self.G,
            self.S_ring,
            self.S_support,
            self.S_cross_exc,
            self.S_cross_inh,
            self.S_in,
            self.spikes,
            self.state_mon,
        )

        self.recent_spikes = np.zeros(self.n_state, dtype=np.float32)
        self.last_spike_count = 0

    def _ring_idx(self, name: str) -> np.ndarray:
        r = self.rings[name]
        return np.arange(r.start, r.end)

    def _connect_ring_to_ring(self, syn: Synapses, src: str, dst: str, probability: float):
        src_idx = self._ring_idx(src)
        dst_idx = self._ring_idx(dst)
        pairs_i = []
        pairs_j = []
        for i in src_idx:
            for j in dst_idx:
                if random.random() < probability:
                    pairs_i.append(i)
                    pairs_j.append(j)
        syn.connect(i=pairs_i, j=pairs_j)

    def _connect_input_to_ring(self, input_name: str, ring_name: str):
        input_idx = self.input_names.index(input_name)
        ring_idx = self._ring_idx(ring_name)
        self.S_in.connect(i=[input_idx] * len(ring_idx), j=ring_idx)

    def encode_observation(self, obs: dict, env: RingsHomeostasisEnv) -> np.ndarray:
        """
        Convert symbolic observation into input rates.

        MiniGrid orientation:
            0 = right, 1 = down, 2 = left, 3 = up.
        We turn food/danger vectors into approximate left/right/ahead signals.
        """
        print(f"Observation keys: {obs.keys()}")
        energy, fdx, fdy, fprox, ddx, ddy, dprox = obs["rstate"]
        direction = int(obs["direction"])

        def relative_signals(dx: float, dy: float, prox: float) -> Tuple[float, float, float]:
            # Convert world vector into egocentric ahead/left/right signals.
            # Forward vector depends on agent direction.
            dirs = {
                0: np.array([1.0, 0.0]),
                1: np.array([0.0, 1.0]),
                2: np.array([-1.0, 0.0]),
                3: np.array([0.0, -1.0]),
            }
            fwd = dirs[direction]
            left = np.array([-fwd[1], fwd[0]])
            v = np.array([dx, dy], dtype=float)
            if np.linalg.norm(v) < 1e-6:
                return 0.0, 0.0, 0.0
            ahead_score = max(0.0, float(np.dot(v, fwd))) * prox
            left_score = max(0.0, float(np.dot(v, left))) * prox
            right_score = max(0.0, float(np.dot(v, -left))) * prox
            return left_score, right_score, ahead_score

        food_left, food_right, food_ahead = relative_signals(float(fdx), float(fdy), float(fprox))
        danger_left, danger_right, danger_ahead = relative_signals(float(ddx), float(ddy), float(dprox))
        low_energy = max(0.0, (0.55 - float(energy)) / 0.55)
        wall_ahead = env._wall_ahead()

        rates = np.array(
            [
                low_energy,
                food_left,
                food_right,
                food_ahead,
                danger_left,
                danger_right,
                danger_ahead,
                wall_ahead,
            ],
            dtype=float,
        )

        # Convert normalized features into Poisson rates.
        return np.clip(rates, 0.0, 1.0) * 220.0

    def step(self, obs: dict, env: RingsHomeostasisEnv) -> Tuple[int, Dict[str, float]]:
        """Run SNN for one control window and decode action."""
        rates = self.encode_observation(obs, env)
        self.inputs.rates = rates * Hz

        before = len(self.spikes.i)
        self.net.run(self.snn_step_ms * ms)
        after = len(self.spikes.i)

        self.recent_spikes[:] = 0
        if after > before:
            ids = np.asarray(self.spikes.i[before:after])
            for idx in ids:
                self.recent_spikes[int(idx)] += 1
        self.last_spike_count = after

        acts = self.ring_activations()
        action = self.decode_action(obs, env, acts)
        return action, acts

    def ring_activations(self) -> Dict[str, float]:
        acts = {}
        for name, r in self.rings.items():
            count = float(np.sum(self.recent_spikes[r.start:r.end]))
            # Normalize by ring size and control window.
            acts[name] = count / max(1.0, r.n)
        return acts

    def decode_action(self, obs: dict, env: RingsHomeostasisEnv, acts: Dict[str, float]) -> int:
        """
        Decode ring states into MiniGrid actions.

        Actions in MiniGrid:
            0 left, 1 right, 2 forward, 3 pickup, 4 drop, 5 toggle, 6 done.
        """
        rates = self.encode_observation(obs, env) / 220.0
        _, food_left, food_right, food_ahead, danger_left, danger_right, danger_ahead, wall_ahead = rates

        # Priority 1: escape from danger.
        if acts["Danger"] > 0.05 or acts["Escape"] > 0.05:
            if danger_ahead > 0.12:
                return 0 if random.random() < 0.5 else 1
            if danger_left > danger_right:
                return 1  # turn away from left danger
            if danger_right > danger_left:
                return 0  # turn away from right danger
            return 2

        # Priority 2: if hungry and food visible, approach food.
        if acts["EnergyLow"] > 0.02 or acts["FoodSeeking"] > 0.02:
            if food_ahead > 0.05 and not wall_ahead:
                return 2
            if food_left > food_right:
                return 0
            if food_right > food_left:
                return 1

        # Default exploration.
        if wall_ahead > 0.5:
            return 0 if random.random() < 0.5 else 1
        if random.random() < 0.15:
            return 0 if random.random() < 0.5 else 1
        return 2


# -----------------------------------------------------------------------------
# 3. Visualization
# -----------------------------------------------------------------------------

class RingsVisualizer:
    def __init__(self, controller: RingsSNNController):
        self.controller = controller
        self.fig = plt.figure(figsize=(11, 5))
        self.ax_env = self.fig.add_subplot(1, 3, 1)
        self.ax_rings = self.fig.add_subplot(1, 3, 2)
        self.ax_time = self.fig.add_subplot(1, 3, 3)
        self.history: List[Dict[str, float]] = []
        plt.ion()
        plt.show(block=False)

    def update(self, env: RingsHomeostasisEnv, acts: Dict[str, float], step_idx: int, info: dict):
        self.history.append(dict(acts))
        self.history = self.history[-80:]

        # Environment image.
        self.ax_env.clear()
        frame = env.render()
        self.ax_env.imshow(frame)
        self.ax_env.axis("off")
        self.ax_env.set_title(f"MiniGrid agent\nstep={step_idx}, energy={info.get('energy', 0):.2f}")

        # Ring diagram.
        self.ax_rings.clear()
        self.ax_rings.set_aspect("equal")
        self.ax_rings.axis("off")
        self.ax_rings.set_title("RINGS state activity")

        ring_names = list(self.controller.rings.keys())
        centers = {
            "Explore": (-1.2, 0.9),
            "EnergyLow": (1.2, 0.9),
            "FoodSeeking": (1.2, -0.8),
            "Danger": (-1.2, -0.8),
            "Escape": (0.0, -1.55),
        }
        for name in ring_names:
            x, y = centers[name]
            a = min(1.0, acts[name] / 2.0)
            circle = plt.Circle((x, y), 0.45, fill=True, alpha=0.15 + 0.75 * a)
            self.ax_rings.add_patch(circle)
            self.ax_rings.text(x, y, f"{name}\n{acts[name]:.2f}", ha="center", va="center", fontsize=9)

        self.ax_rings.set_xlim(-2.1, 2.1)
        self.ax_rings.set_ylim(-2.2, 1.7)

        # Activation time series.
        self.ax_time.clear()
        self.ax_time.set_title("Ring activation history")
        self.ax_time.set_xlabel("recent steps")
        self.ax_time.set_ylabel("spikes / neuron")
        if self.history:
            xs = np.arange(len(self.history))
            for name in ring_names:
                ys = [h[name] for h in self.history]
                self.ax_time.plot(xs, ys, label=name)
            self.ax_time.legend(fontsize=7, loc="upper left")
            self.ax_time.set_ylim(0, max(0.5, max(max(h.values()) for h in self.history) + 0.2))

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# -----------------------------------------------------------------------------
# 4. Experiment runner
# -----------------------------------------------------------------------------

def run_episode(seed: int, visualize: bool = True):
    env = RingsHomeostasisEnv(size=9, max_steps=250, render_mode="rgb_array")
    obs, info = env.reset2(seed)

    controller = RingsSNNController(ring_size=12, snn_step_ms=30, seed=seed)
    vis = RingsVisualizer(controller) if visualize else None

    total_reward = 0.0
    terminated = truncated = False
    step_idx = 0

    while not (terminated or truncated):
        action, acts = controller.step(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if vis is not None:
            vis.update(env, acts, step_idx, info)

        step_idx += 1

    print(f"Episode finished: steps={step_idx}, total_reward={total_reward:.3f}, final_energy={info.get('energy', 0):.3f}")
    env.close()
    if visualize:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    run_episode(seed=7, visualize=True)
