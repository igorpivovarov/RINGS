"""
RINGS / TEVSER Ω2 prototype
MiniGrid + Brian2 closed-loop SNN controller with ring-state visualization.

v 0.03
- Replaces global nearest-object sensors with local egocentric sensors.
- Food and danger are detected only in left/ahead/right rays within sensor_range.
- Sensor intensity decays with distance.
- Adds tunable sensor_range.

Scientific note:
    The first optimization criterion is survival time. Later we should add
    secondary metrics: food found, danger contacts, ring stability, attractor
    persistence, and ablation sensitivity.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Lava, Ball
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
    start_scope,
)

prefs.codegen.target = "numpy"


# -----------------------------------------------------------------------------
# 1. Tunable parameter set
# -----------------------------------------------------------------------------

@dataclass
class RingsParams:
    ring_size: int = 12
    snn_step_ms: int = 30

    w_ring: float = 7.0
    w_support: float = 2.2
    w_cross_exc: float = 4.5
    w_cross_inh: float = 5.5
    w_input: float = 8.0

    energy_low_threshold: float = 0.55
    danger_threshold: float = 0.12
    food_threshold: float = 0.05

    escape_activation_threshold: float = 0.05
    seeking_activation_threshold: float = 0.02

    input_rate_hz: float = 220.0
    sensor_range: int = 5


def sample_params(rng: random.Random) -> RingsParams:
    """Random-search sampling space. Keep ranges conservative at first."""
    return RingsParams(
        ring_size=rng.choice([8, 10, 12, 16, 20]),
        snn_step_ms=rng.choice([15, 20, 30, 40, 60]),
        w_ring=rng.uniform(5.0, 10.5),
        w_support=rng.uniform(0.5, 4.0),
        w_cross_exc=rng.uniform(2.0, 8.0),
        w_cross_inh=rng.uniform(2.0, 9.0),
        w_input=rng.uniform(4.0, 12.0),
        energy_low_threshold=rng.uniform(0.35, 0.75),
        danger_threshold=rng.uniform(0.05, 0.25),
        food_threshold=rng.uniform(0.02, 0.18),
        escape_activation_threshold=rng.uniform(0.01, 0.15),
        seeking_activation_threshold=rng.uniform(0.01, 0.12),
        input_rate_hz=rng.uniform(120.0, 320.0),
        sensor_range=rng.choice([3, 4, 5, 6, 7]),
    )


# -----------------------------------------------------------------------------
# 2. MiniGrid environment: homeostatic bug world
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

    Sensors are local and egocentric:
        food_left, food_ahead, food_right
        danger_left, danger_ahead, danger_right
    Each sensor is a ray cast from the agent in relative directions.
    Intensity decays with distance.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        size: int = 9,
        max_steps: int = 250,
        n_food: int = 3,
        n_danger: int = 2,
        sensor_range: int = 5,
        render_mode: Optional[str] = "rgb_array",
    ):
        self.size = size
        self.n_food = n_food
        self.n_danger = n_danger
        self.sensor_range = sensor_range
        self.energy = 1.0
        self.energy_decay = 0.008
        self.energy_gain = 0.55
        self.danger_penalty = 0.25
        self.last_event = "reset"
        self.food_collected = 0
        self.danger_contacts = 0

        mission_space = MissionSpace(mission_func=lambda: "maintain energy and avoid danger")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            render_mode=render_mode,
        )

        self.observation_space = spaces.Dict(
            {
                "image": self.observation_space["image"],
                "direction": spaces.Discrete(4),
                "mission": self.observation_space["mission"],
                # energy + 3 food sensors + 3 danger sensors
                "rstate": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._get_obs()
        info.update(
            {
                "energy": self.energy,
                "last_event": self.last_event,
                "food_collected": self.food_collected,
                "danger_contacts": self.danger_contacts,
            }
        )
        return obs, info

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.place_agent()

        for _ in range(self.n_food):
            self.place_obj(Food())
        for _ in range(self.n_danger):
            self.place_obj(Danger())

        self.mission = "maintain energy and avoid danger"
        self.energy = 1.0
        self.last_event = "reset"
        self.food_collected = 0
        self.danger_contacts = 0

    def _dir_vec(self, direction: int) -> np.ndarray:
        dirs = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }
        return dirs[int(direction)]

    def _relative_dirs(self) -> Dict[str, np.ndarray]:
        fwd = self._dir_vec(self.agent_dir)
        left = np.array([-fwd[1], fwd[0]])
        right = -left
        return {"left": left, "ahead": fwd, "right": right}

    def _ray_sensor(self, rel_name: str, obj_type: str) -> float:
        """
        Cast a local egocentric ray.

        Returns 0 if object is not seen.
        If object is found at distance d, intensity is:
            (sensor_range - d + 1) / sensor_range
        so closer objects create stronger input.
        """
        direction = self._relative_dirs()[rel_name]
        x, y = map(int, self.agent_pos)
        for d in range(1, self.sensor_range + 1):
            px = x + int(direction[0]) * d
            py = y + int(direction[1]) * d
            obj = self.grid.get(px, py)
            if obj is None:
                continue
            if obj.type == "wall":
                return 0.0
            if obj.type == obj_type:
                return float((self.sensor_range - d + 1) / self.sensor_range)
        return 0.0

    def _wall_ahead(self) -> float:
        obj = self.grid.get(*self.front_pos)
        return 1.0 if obj is not None and obj.type == "wall" else 0.0

    def _get_obs(self):
        obs = super().gen_obs()
        obs["rstate"] = np.array(
            [
                self.energy,
                self._ray_sensor("left", "food"),
                self._ray_sensor("ahead", "food"),
                self._ray_sensor("right", "food"),
                self._ray_sensor("left", "danger"),
                self._ray_sensor("ahead", "danger"),
                self._ray_sensor("right", "danger"),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        self.energy -= self.energy_decay
        self.last_event = "none"

        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == "food":
            self.energy = min(1.0, self.energy + self.energy_gain)
            self.grid.set(*self.agent_pos, None)
            self.place_obj(Food())
            self.last_event = "food"
            self.food_collected += 1
            reward += 0.3

        if cell is not None and cell.type == "danger":
            self.energy -= self.danger_penalty
            self.last_event = "danger_contact"
            self.danger_contacts += 1
            reward -= 0.5

        if self.energy <= 0.0:
            terminated = True
            reward -= 1.0
            self.last_event = "energy_depleted"

        reward += 0.02 * self.energy

        obs = self._get_obs()
        info.update(
            {
                "energy": self.energy,
                "last_event": self.last_event,
                "food_collected": self.food_collected,
                "danger_contacts": self.danger_contacts,
            }
        )
        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# 3. Brian2 SNN controller with persistent rings
# -----------------------------------------------------------------------------

@dataclass
class RingSpec:
    name: str
    n: int
    start: int
    end: int


class RingsSNNController:
    """Ring-state SNN controller for RINGS."""

    def __init__(self, params: RingsParams, seed: int = 1):
        start_scope()
        self.params = params
        np.random.seed(seed)
        random.seed(seed)
        self.ring_size = params.ring_size
        self.snn_step_ms = params.snn_step_ms

        names = ["Explore", "EnergyLow", "FoodSeeking", "Danger", "Escape"]
        self.rings: Dict[str, RingSpec] = {}
        start = 0
        for name in names:
            self.rings[name] = RingSpec(name=name, n=params.ring_size, start=start, end=start + params.ring_size)
            start += params.ring_size
        self.n_state = start

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

        self.S_ring = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_ring}*mV")
        for ring in self.rings.values():
            i = np.arange(ring.start, ring.end)
            j = ring.start + (np.arange(ring.n) + 1) % ring.n
            self.S_ring.connect(i=i, j=j)

        self.S_support = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_support}*mV")
        for ring in self.rings.values():
            i = np.arange(ring.start, ring.end)
            j = ring.start + (np.arange(ring.n) + 2) % ring.n
            self.S_support.connect(i=i, j=j)

        self.S_cross_exc = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_cross_exc}*mV")
        self._connect_ring_to_ring(self.S_cross_exc, "Danger", "Escape", probability=0.25)
        self._connect_ring_to_ring(self.S_cross_exc, "EnergyLow", "FoodSeeking", probability=0.25)

        self.S_cross_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_cross_inh}*mV")
        self._connect_ring_to_ring(self.S_cross_inh, "Escape", "FoodSeeking", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "Escape", "Explore", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "FoodSeeking", "Explore", probability=0.25)
        self._connect_ring_to_ring(self.S_cross_inh, "Danger", "Explore", probability=0.35)

        self.S_in = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_input}*mV")
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

    def _ring_idx(self, name: str) -> np.ndarray:
        r = self.rings[name]
        return np.arange(r.start, r.end)

    def _connect_ring_to_ring(self, syn: Synapses, src: str, dst: str, probability: float):
        src_idx = self._ring_idx(src)
        dst_idx = self._ring_idx(dst)
        pairs_i, pairs_j = [], []
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
        Convert local egocentric sensors into SNN input rates.

        rstate now contains:
            energy,
            food_left, food_ahead, food_right,
            danger_left, danger_ahead, danger_right
        """
        energy, food_left, food_ahead, food_right, danger_left, danger_ahead, danger_right = obs["rstate"]
        low_energy = max(
            0.0,
            (self.params.energy_low_threshold - float(energy)) / max(1e-6, self.params.energy_low_threshold),
        )
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
        return np.clip(rates, 0.0, 1.0) * self.params.input_rate_hz

    def step(self, obs: dict, env: RingsHomeostasisEnv) -> Tuple[int, Dict[str, float]]:
        input_rates_hz = self.encode_observation(obs, env)
        self.inputs.rates = input_rates_hz * Hz

        before = len(self.spikes.i)
        self.net.run(self.snn_step_ms * ms)
        after = len(self.spikes.i)

        self.recent_spikes[:] = 0
        if after > before:
            ids = np.asarray(self.spikes.i[before:after])
            for idx in ids:
                self.recent_spikes[int(idx)] += 1

        acts = self.ring_activations()
        action = self.decode_action(obs, env, acts)
        return action, acts

    def ring_activations(self) -> Dict[str, float]:
        acts = {}
        for name, r in self.rings.items():
            count = float(np.sum(self.recent_spikes[r.start:r.end]))
            acts[name] = count / max(1.0, r.n)
        return acts

    def decode_action(self, obs: dict, env: RingsHomeostasisEnv, acts: Dict[str, float]) -> int:
        input_features = self.encode_observation(obs, env) / self.params.input_rate_hz
        _, food_left, food_right, food_ahead, danger_left, danger_right, danger_ahead, wall_ahead = input_features

        if acts["Danger"] > self.params.escape_activation_threshold or acts["Escape"] > self.params.escape_activation_threshold:
            if danger_ahead > self.params.danger_threshold:
                return 0 if random.random() < 0.5 else 1
            if danger_left > danger_right:
                return 1
            if danger_right > danger_left:
                return 0
            return 2

        if acts["EnergyLow"] > self.params.seeking_activation_threshold or acts["FoodSeeking"] > self.params.seeking_activation_threshold:
            if food_ahead > self.params.food_threshold and not wall_ahead:
                return 2
            if food_left > food_right:
                return 0
            if food_right > food_left:
                return 1

        if wall_ahead > 0.5:
            return 0 if random.random() < 0.5 else 1
        if random.random() < 0.15:
            return 0 if random.random() < 0.5 else 1
        return 2


# -----------------------------------------------------------------------------
# 4. Visualization
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

        self.ax_env.clear()
        frame = env.render()
        self.ax_env.imshow(frame)
        self.ax_env.axis("off")
        self.ax_env.set_title(
            f"MiniGrid agent\nstep={step_idx}, energy={info.get('energy', 0):.2f}, food={info.get('food_collected', 0)}"
        )

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
# 5. Episode evaluation and random-search tuning
# -----------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    final_energy: float
    food_collected: int
    danger_contacts: int


def run_episode(
    seed: int,
    params: Optional[RingsParams] = None,
    visualize: bool = True,
    max_steps: int = 250,
) -> EpisodeResult:
    params = params or RingsParams()
    env = RingsHomeostasisEnv(size=9, max_steps=max_steps, sensor_range=params.sensor_range, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    controller = RingsSNNController(params=params, seed=seed)
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

    result = EpisodeResult(
        steps=step_idx,
        total_reward=float(total_reward),
        final_energy=float(info.get("energy", 0.0)),
        food_collected=int(info.get("food_collected", 0)),
        danger_contacts=int(info.get("danger_contacts", 0)),
    )
    env.close()

    if visualize:
        plt.ioff()
        plt.show()

    return result


def evaluate_params(
    params: RingsParams,
    seeds: List[int],
    max_steps: int = 250,
) -> Dict[str, float]:
    results = [run_episode(seed=s, params=params, visualize=False, max_steps=max_steps) for s in seeds]
    steps = np.array([r.steps for r in results], dtype=float)
    rewards = np.array([r.total_reward for r in results], dtype=float)
    foods = np.array([r.food_collected for r in results], dtype=float)
    dangers = np.array([r.danger_contacts for r in results], dtype=float)

    # Main criterion: homeostatic success, not raw lifespan.
    # Food collection is the primary positive signal.
    # Danger contacts are the primary negative signal.
    # Lifespan remains only as a weak secondary term, so random wandering does not dominate.
    food_weight = 60.0
    danger_weight = 45.0
    step_weight = 0.20
    reward_weight = 0.10

    score = float(
        food_weight * np.mean(foods)
        - danger_weight * np.mean(dangers)
        + step_weight * np.mean(steps)
        + reward_weight * np.mean(rewards)
    )
    return {
        "score": score,
        "food_weight": food_weight,
        "danger_weight": danger_weight,
        "step_weight": step_weight,
        "reward_weight": reward_weight,
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_reward": float(np.mean(rewards)),
        "mean_food": float(np.mean(foods)),
        "mean_danger": float(np.mean(dangers)),
    }


def tune_random_search(
    trials: int = 50,
    seeds: Optional[List[int]] = None,
    max_steps: int = 250,
    search_seed: int = 123,
    output_path: str = "rings_best_params.json",
) -> Tuple[RingsParams, Dict[str, float]]:
    seeds = seeds or [1, 2, 3]
    rng = random.Random(search_seed)

    best_params: Optional[RingsParams] = None
    best_metrics: Optional[Dict[str, float]] = None
    all_results = []

    for t in range(1, trials + 1):
        params = sample_params(rng)
        metrics = evaluate_params(params, seeds=seeds, max_steps=max_steps)
        row = {"trial": t, "params": asdict(params), "metrics": metrics}
        all_results.append(row)

        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_params = params
            best_metrics = metrics
            print(f"[trial {t:03d}] NEW BEST score={metrics['score']:.2f}, food={metrics['mean_food']:.2f}, danger={metrics['mean_danger']:.2f}, mean_steps={metrics['mean_steps']:.1f}")
            print(json.dumps(asdict(params), indent=2))
        else:
            print(f"[trial {t:03d}] score={metrics['score']:.2f}, food={metrics['mean_food']:.2f}, danger={metrics['mean_danger']:.2f}, mean_steps={metrics['mean_steps']:.1f}")

    assert best_params is not None and best_metrics is not None
    payload = {
        "best_params": asdict(best_params),
        "best_metrics": best_metrics,
        "all_results": all_results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved tuning results to {output_path}")
    print("Best metrics:", json.dumps(best_metrics, indent=2))
    return best_params, best_metrics


def load_params(path: str) -> RingsParams:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload.get("best_params", payload)
    return RingsParams(**data)


# -----------------------------------------------------------------------------
# 6. CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "tune"], default="run")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--params", type=str, default=None, help="Path to rings_best_params.json")
    parser.add_argument("--output", type=str, default="rings_best_params.json")
    parser.add_argument("--visual", choices=["on", "off"], default="off")
    args = parser.parse_args()

    if args.mode == "tune":
        tune_random_search(
            trials=args.trials,
            seeds=[1, 2, 3],
            max_steps=args.max_steps,
            search_seed=args.seed,
            output_path=args.output,
        )
    else:
        params = load_params(args.params) if args.params else RingsParams()
        visualize = args.visual == "on"
        result = run_episode(seed=args.seed, params=params, visualize=visualize, max_steps=args.max_steps)
        print(result)


if __name__ == "__main__":
    main()
