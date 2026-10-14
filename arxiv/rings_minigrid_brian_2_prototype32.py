"""
RINGS / TEVSER Ω2 prototype
MiniGrid + Brian2 closed-loop SNN controller with ring-state visualization.

v 0.32
- Adds lateral danger turn-inhibition pathways.
- danger_left inhibits MotorLeft, preventing turns toward left-side danger.
- danger_right inhibits MotorRight, preventing turns toward right-side danger.
- This complements danger_ahead -> MotorForward inhibition from v0.31.
- Intended fix: when facing a wall, the agent should not escape by turning into a visible side danger.

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
import csv
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

    # Modulatory transverse rings.
    enable_modulatory_rings: bool = True
    fatigue_tau_ms: float = 300.0
    curiosity_tau_ms: float = 220.0
    curiosity_bias_mv: float = -65.2

    w_ring: float = 7.0
    w_support: float = 2.2
    w_cross_exc: float = 4.5
    w_cross_inh: float = 5.5
    w_input: float = 8.0
    w_food_sensor_to_motor: float = 16.0

    w_fatigue_in: float = 0.8
    w_fatigue_out: float = 1.2
    w_motor_to_fatigue: float = 0.25
    w_rest_to_fatigue_inh: float = 2.0
    w_explore_to_motor_forward: float = 3.0
    w_curiosity_to_motor_forward: float = 1.2
    w_lateral_danger_to_forward: float = 7.0
    w_escape_to_motor_forward: float = 7.0
    w_danger_to_motor_forward: float = 5.0
    w_danger_sensor_to_escape: float = 14.0
    danger_input_gain: float = 1.8
    w_danger_escape_rest_inh: float = 4.0
    w_danger_escape_fatigue_inh: float = 4.0
    w_energy_low_to_explore: float = 5.0
    w_foodseeking_to_motor_forward: float = 7.0
    w_hunger_rest_inh: float = 5.0
    w_danger_ahead_to_turn: float = 8.0
    w_danger_ahead_forward_inh: float = 10.0
    w_lateral_danger_turn_inh: float = 10.0
    w_curiosity_inh: float = 0.45
    w_curiosity_to_explore: float = 2.6
    w_idle_to_curiosity: float = 2.2
    w_fatigue_to_rest: float = 2.0
    w_fatigue_motor_inh: float = 1.2
    fatigue_rest_threshold: float = 0.18

    energy_low_threshold: float = 0.55
    danger_threshold: float = 0.12
    food_threshold: float = 0.05
    food_motor_threshold: float = 0.10
    idle_food_threshold: float = 0.10
    idle_danger_threshold: float = 0.10
    food_forward_bias: float = 0.0

    escape_activation_threshold: float = 0.05
    seeking_activation_threshold: float = 0.02

    input_rate_hz: float = 220.0
    sensor_range: int = 3
    idle_rise_per_step: float = 0.20
    idle_decay_per_step: float = 0.50
    idle_activation_threshold: float = 0.80
    idle_energy_threshold: float = 0.35


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
        w_food_sensor_to_motor=rng.uniform(10.0, 24.0),
        energy_low_threshold=rng.uniform(0.35, 0.75),
        danger_threshold=rng.uniform(0.05, 0.25),
        food_threshold=rng.uniform(0.02, 0.18),
        food_motor_threshold=rng.uniform(0.08, 0.18),
        idle_food_threshold=rng.uniform(0.08, 0.18),
        idle_danger_threshold=rng.uniform(0.08, 0.18),
        food_forward_bias=0.0,
        escape_activation_threshold=rng.uniform(0.01, 0.15),
        seeking_activation_threshold=rng.uniform(0.01, 0.12),
        input_rate_hz=rng.uniform(120.0, 320.0),
        sensor_range=rng.choice([1, 2, 3]),
        idle_rise_per_step=rng.uniform(0.12, 0.28),
        idle_decay_per_step=rng.uniform(0.35, 0.80),
        idle_activation_threshold=rng.uniform(0.65, 0.95),
        idle_energy_threshold=rng.uniform(0.25, 0.55),
        enable_modulatory_rings=True,
        fatigue_tau_ms=rng.uniform(200.0, 500.0),
        curiosity_tau_ms=rng.uniform(150.0, 350.0),
        curiosity_bias_mv=rng.uniform(-65.8, -64.5),
        w_fatigue_in=rng.uniform(0.2, 1.0),
        w_fatigue_out=rng.uniform(0.5, 2.2),
        w_motor_to_fatigue=rng.uniform(0.05, 0.45),
        w_rest_to_fatigue_inh=rng.uniform(1.0, 4.0),
        w_explore_to_motor_forward=rng.uniform(2.0, 8.0),
        w_curiosity_to_motor_forward=rng.uniform(0.2, 3.5),
        w_lateral_danger_to_forward=rng.uniform(5.0, 10.0),
        w_escape_to_motor_forward=rng.uniform(4.0, 10.0),
        w_danger_to_motor_forward=rng.uniform(3.0, 8.0),
        w_danger_sensor_to_escape=rng.uniform(8.0, 22.0),
        danger_input_gain=rng.uniform(1.2, 2.5),
        w_danger_escape_rest_inh=rng.uniform(2.0, 7.0),
        w_danger_escape_fatigue_inh=rng.uniform(2.0, 7.0),
        w_energy_low_to_explore=rng.uniform(2.0, 8.0),
        w_foodseeking_to_motor_forward=rng.uniform(4.0, 12.0),
        w_hunger_rest_inh=rng.uniform(2.0, 8.0),
        w_danger_ahead_to_turn=rng.uniform(5.0, 12.0),
        w_danger_ahead_forward_inh=rng.uniform(6.0, 16.0),
        w_lateral_danger_turn_inh=rng.uniform(6.0, 16.0),
        w_curiosity_inh=rng.uniform(0.15, 0.8),
        w_curiosity_to_explore=rng.uniform(1.5, 4.0),
        w_idle_to_curiosity=rng.uniform(0.8, 4.0),
        w_fatigue_to_rest=rng.uniform(1.0, 3.0),
        w_fatigue_motor_inh=rng.uniform(0.5, 2.0),
        fatigue_rest_threshold=rng.uniform(0.12, 0.35),
    )


# -----------------------------------------------------------------------------
# 2. MiniGrid environment: homeostatic bug world
# -----------------------------------------------------------------------------

class Food(Ball):
    """Food object. Using Ball rendering but semantically it restores energy."""

    def __init__(self):
        super().__init__(color="green")
        self.type = "food"

    def can_overlap(self) -> bool:
        # Important: Ball-like objects are normally not walkable in MiniGrid.
        # For RINGS, food must be consumable by moving onto its cell.
        return True

    def can_pickup(self) -> bool:
        return False


class Danger(Ball):
    """Danger object. Rendered as a red ball, but semantically it is a harmful cell."""

    def __init__(self):
        # Do not inherit from Lava here.
        # Lava may have environment-level special behavior in MiniGrid variants.
        # For RINGS, danger should be a normal walkable object with custom energy penalty.
        super().__init__(color="red")
        self.type = "danger"

    def can_overlap(self) -> bool:
        # The agent must be able to enter the cell so danger contacts can be measured.
        return True

    def can_pickup(self) -> bool:
        return False


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
        size: int = 15,
        max_steps: int = 250,
        n_food: int = 3,
        n_danger: int = 2,
        sensor_range: int = 5,
        danger_speed: float = 0.0,
        render_mode: Optional[str] = "rgb_array",
    ):
        self.size = size
        self.n_food = n_food
        self.n_danger = n_danger
        # Keep sensors local; larger ranges suppress Idle/Curiosity too much.
        self.sensor_range = min(int(sensor_range), 3)
        self.danger_speed = float(danger_speed)
        self.danger_speed_accumulator = 0.0
        self.danger_contact_signal = 0.0
        self.energy = 1.0
        self.energy_decay = 0.008
        self.energy_gain = 0.55
        self.danger_penalty = 0.25
        self.last_event = "reset"
        self.food_collected = 0
        self.danger_contacts = 0
        self.idle_level = 0.0
        self.danger_speed_accumulator = 0.0
        self.danger_contact_signal = 0.0
        self.idle_level = 0.0

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
                "last_action": action if 'action' in locals() else None,
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
        self.idle_level = 0.0
        self.danger_speed_accumulator = 0.0
        self.danger_contact_signal = 0.0

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
        # MiniGrid screen coordinates have y growing downward.
        # For egocentric left/right, this convention requires this orientation.
        left = np.array([fwd[1], -fwd[0]])
        right = -left
        return {"left": left, "ahead": fwd, "right": right}

    def _sector_sensors(self, obj_type: str) -> Tuple[float, float, float]:
        """
        Local sector/cone-like egocentric sensors.

        Returns:
            left, ahead, right

        Unlike strict ray sensors, this detects diagonal objects too.
        Example: an object front-right contributes to both ahead and right.
        Distance uses a quadratic decay so distant objects are weak.
        """
        dirs = self._relative_dirs()
        fwd = dirs["ahead"].astype(float)
        left = dirs["left"].astype(float)
        right = dirs["right"].astype(float)
        ax, ay = map(int, self.agent_pos)

        values = {"left": 0.0, "ahead": 0.0, "right": 0.0}

        for x in range(max(0, ax - self.sensor_range), min(self.width, ax + self.sensor_range + 1)):
            for y in range(max(0, ay - self.sensor_range), min(self.height, ay + self.sensor_range + 1)):
                if x == ax and y == ay:
                    continue
                obj = self.grid.get(x, y)
                if obj is None or obj.type != obj_type:
                    continue

                vec = np.array([x - ax, y - ay], dtype=float)
                dist = float(np.max(np.abs(vec)))  # Chebyshev distance for grid perception
                if dist < 1 or dist > self.sensor_range:
                    continue

                unit = vec / max(1e-6, np.linalg.norm(vec))
                distance_gain = ((self.sensor_range - dist + 1) / self.sensor_range) ** 2

                # Dot products distribute diagonal objects across adjacent sectors.
                ahead_score = max(0.0, float(np.dot(unit, fwd)))
                left_score = max(0.0, float(np.dot(unit, left)))
                right_score = max(0.0, float(np.dot(unit, right)))

                values["ahead"] = max(values["ahead"], ahead_score * distance_gain)
                values["left"] = max(values["left"], left_score * distance_gain)
                values["right"] = max(values["right"], right_score * distance_gain)

        return float(values["left"]), float(values["ahead"]), float(values["right"])

    def _wall_ahead(self) -> float:
        obj = self.grid.get(*self.front_pos)
        return 1.0 if obj is not None and obj.type == "wall" else 0.0

    def _danger_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "danger":
                    positions.append((x, y))
        return positions

    def _sign(self, value: int) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def _move_dangers_once(self):
        """Move each danger object once.

        Random walk by default. If the agent is adjacent, including diagonals,
        danger moves toward the agent. If it enters the agent cell, a contact
        penalty is applied.
        """
        positions = self._danger_positions()
        random.shuffle(positions)
        ax, ay = map(int, self.agent_pos)
        occupied_danger = set(positions)

        for x, y in positions:
            obj = self.grid.get(x, y)
            if obj is None or obj.type != "danger":
                continue

            occupied_danger.discard((x, y))
            dx = ax - x
            dy = ay - y
            cheb_dist = max(abs(dx), abs(dy))

            if cheb_dist <= 1:
                # Adjacent pursuit, but no diagonal jumps. If the agent is diagonal,
                # choose one orthogonal step that brings danger closer.
                if abs(dx) == 1 and abs(dy) == 1:
                    step_options = [(self._sign(dx), 0), (0, self._sign(dy))]
                    random.shuffle(step_options)
                elif abs(dx) == 1 and dy == 0:
                    step_options = [(self._sign(dx), 0)]
                elif dx == 0 and abs(dy) == 1:
                    step_options = [(0, self._sign(dy))]
                else:
                    step_options = []
            else:
                step_options = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                random.shuffle(step_options)

            moved = False
            for sx, sy in step_options:
                nx = x + sx
                ny = y + sy
                if nx <= 0 or ny <= 0 or nx >= self.width - 1 or ny >= self.height - 1:
                    continue
                if (nx, ny) in occupied_danger:
                    continue
                target = self.grid.get(nx, ny)
                if target is not None:
                    # Danger should not consume or overwrite food.
                    # It can only move into empty cells or the agent's current cell.
                    continue

                self.grid.set(x, y, None)
                if nx == ax and ny == ay:
                    self.energy -= self.danger_penalty
                    self.danger_contacts += 1
                    self.last_event = "danger_contact"
                    self.danger_contact_signal = 1.0
                self.grid.set(nx, ny, obj)
                occupied_danger.add((nx, ny))
                moved = True
                break

            if not moved:
                occupied_danger.add((x, y))

    def _move_dangers_by_speed(self):
        if self.danger_speed <= 0.0:
            return
        self.danger_speed_accumulator += self.danger_speed
        while self.danger_speed_accumulator >= 1.0:
            self._move_dangers_once()
            self.danger_speed_accumulator -= 1.0

    def _get_obs(self):
        obs = super().gen_obs()
        food_left, food_ahead, food_right = self._sector_sensors("food")
        danger_left, danger_ahead, danger_right = self._sector_sensors("danger")
        obs["rstate"] = np.array(
            [
                self.energy,
                food_left,
                food_ahead,
                food_right,
                danger_left,
                danger_ahead,
                danger_right,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        # Preserve homeostatic variables explicitly across MiniGrid's internal step.
        # This prevents any environment-side side effect from resetting energy.
        prev_energy = float(self.energy)
        prev_food_collected = int(self.food_collected)
        prev_danger_contacts = int(self.danger_contacts)

        # Action 6 is used here as REST/no-op, not as task termination.
        if action == 6:
            self.step_count += 1
            reward = 0.0
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = {}
            # Resting still consumes a small basal metabolic energy.
            # Otherwise the agent could survive forever by remaining inactive.
            # High-energy resting is cheap but not free.
            if prev_energy >= 0.5:
                energy_cost = self.energy_decay * 0.08
            else:
                energy_cost = self.energy_decay * 0.25
            self.energy = prev_energy - energy_cost
            self.last_event = "rest"
        else:
            obs, reward, terminated, truncated, info = super().step(action)
            self.energy = prev_energy - self.energy_decay
        self.food_collected = prev_food_collected
        self.danger_contacts = prev_danger_contacts
        self.last_event = "none"
        self.danger_contact_signal = 0.0

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
            self.danger_contact_signal = 1.0
            self.danger_contacts += 1
            reward -= 0.5

        # Move danger objects after the agent's action and immediate cell effects.
        self._move_dangers_by_speed()

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
                "last_action": action if 'action' in locals() else None,
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

    def __init__(self, params: RingsParams, seed: int = 1, record: bool = False):
        start_scope()
        self.params = params
        self.record = record
        np.random.seed(seed)
        random.seed(seed)
        self.ring_size = params.ring_size
        self.snn_step_ms = params.snn_step_ms

        names = [
            "Explore",
            "EnergyLow",
            "FoodSeeking",
            "Danger",
            "Escape",
            "MotorLeft",
            "MotorRight",
            "MotorForward",
            "MotorRest",
        ]
        if params.enable_modulatory_rings:
            names.extend(["Fatigue", "Curiosity"])
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
            "idle",
            "danger_contact",
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
        if params.enable_modulatory_rings:
            self.G.tau[self._ring_idx("Fatigue")] = params.fatigue_tau_ms * ms
            self.G.tau[self._ring_idx("Curiosity")] = params.curiosity_tau_ms * ms
            self.G.bias[self._ring_idx("Curiosity")] = params.curiosity_bias_mv * mV

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

        self.S_energy_explore = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_energy_low_to_explore}*mV")
        self._connect_ring_to_ring(self.S_energy_explore, "EnergyLow", "Explore", probability=0.35)

        # State-to-motor projections. These are still simple but make motor commands
        # part of the SNN dynamics rather than pure external if/else decoding.
        self.S_foodseeking_forward = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_foodseeking_to_motor_forward}*mV")
        self._connect_ring_to_ring(self.S_foodseeking_forward, "FoodSeeking", "MotorForward", probability=0.40)
        # Explore should reliably produce exploratory movement when activated by Curiosity.
        # This path uses its own tunable weight instead of the generic cross-excitation weight.
        self.S_explore_motor = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_explore_to_motor_forward}*mV")
        self._connect_ring_to_ring(self.S_explore_motor, "Explore", "MotorForward", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_exc, "Escape", "MotorLeft", probability=0.18)
        self._connect_ring_to_ring(self.S_cross_exc, "Escape", "MotorRight", probability=0.18)

        self.S_escape_forward = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_escape_to_motor_forward}*mV")
        self._connect_ring_to_ring(self.S_escape_forward, "Escape", "MotorForward", probability=0.45)

        self.S_danger_forward = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_danger_to_motor_forward}*mV")
        self._connect_ring_to_ring(self.S_danger_forward, "Danger", "MotorForward", probability=0.45)

        # Direct contact with danger is the strongest threat signal.
        self.S_contact_threat = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_input * 2.0}*mV")
        contact_idx = self.input_names.index("danger_contact")
        for ring_name in ["Danger", "Escape", "MotorForward"]:
            ring_idx = self._ring_idx(ring_name)
            self.S_contact_threat.connect(i=[contact_idx] * len(ring_idx), j=ring_idx)

        self.S_danger_escape_rest_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_danger_escape_rest_inh}*mV")
        self._connect_ring_to_ring_all(self.S_danger_escape_rest_inh, "Danger", "MotorRest")
        self._connect_ring_to_ring_all(self.S_danger_escape_rest_inh, "Escape", "MotorRest")

        self.S_danger_escape_fatigue_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_danger_escape_fatigue_inh}*mV")
        self._connect_ring_to_ring_all(self.S_danger_escape_fatigue_inh, "Danger", "Fatigue")
        self._connect_ring_to_ring_all(self.S_danger_escape_fatigue_inh, "Escape", "Fatigue")

        self.S_hunger_rest_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_hunger_rest_inh}*mV")
        self._connect_ring_to_ring_all(self.S_hunger_rest_inh, "EnergyLow", "MotorRest")
        self._connect_ring_to_ring_all(self.S_hunger_rest_inh, "FoodSeeking", "MotorRest")

        self.S_cross_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_cross_inh}*mV")
        self._connect_ring_to_ring(self.S_cross_inh, "Escape", "FoodSeeking", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "Escape", "Explore", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "FoodSeeking", "Explore", probability=0.25)
        self._connect_ring_to_ring(self.S_cross_inh, "Danger", "Explore", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "MotorLeft", "MotorRight", probability=0.35)
        self._connect_ring_to_ring(self.S_cross_inh, "MotorRight", "MotorLeft", probability=0.35)

        self.S_in = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_input}*mV")
        self._connect_input_to_ring("low_energy", "EnergyLow")
        self._connect_input_to_ring("food_left", "FoodSeeking")
        self._connect_input_to_ring("food_right", "FoodSeeking")
        self._connect_input_to_ring("food_ahead", "FoodSeeking")
        self._connect_input_to_ring("danger_left", "Danger")
        self._connect_input_to_ring("danger_right", "Danger")
        self._connect_input_to_ring("danger_ahead", "Danger")
        self._connect_input_to_ring("wall_ahead", "Explore")

        # Direct sensor-to-motor hints. The final action is still determined by motor-ring activity.
        # Separate stronger food-to-motor pathway.
        # This keeps weak food-guided movement inside SNN dynamics rather than decoder rules.
        self.S_food_motor = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_food_sensor_to_motor}*mV")
        for input_name, motor_ring in [
            ("food_left", "MotorLeft"),
            ("food_right", "MotorRight"),
            ("food_ahead", "MotorForward"),
        ]:
            input_idx = self.input_names.index(input_name)
            motor_idx = self._ring_idx(motor_ring)
            self.S_food_motor.connect(i=[input_idx] * len(motor_idx), j=motor_idx)
        self._connect_input_to_ring("danger_left", "MotorRight")
        self._connect_input_to_ring("danger_right", "MotorLeft")
        # danger_ahead should not force a single turning direction.
        # It activates both turn motors symmetrically; competition/noise decides direction.
        self.S_danger_ahead_turn = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_danger_ahead_to_turn}*mV")
        danger_ahead_idx = self.input_names.index("danger_ahead")
        for motor_ring in ["MotorLeft", "MotorRight"]:
            motor_idx = self._ring_idx(motor_ring)
            self.S_danger_ahead_turn.connect(i=[danger_ahead_idx] * len(motor_idx), j=motor_idx)

        # If danger is lateral, turning alone can create an oscillatory trap:
        # danger_right -> turn_left -> danger becomes left -> turn_right -> original state.
        # A weak forward component helps the agent eventually leave the local danger zone.
        self.S_lateral_danger_forward = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_lateral_danger_to_forward}*mV")
        forward_idx = self._ring_idx("MotorForward")
        for input_name in ["danger_left", "danger_right"]:
            input_idx = self.input_names.index(input_name)
            self.S_lateral_danger_forward.connect(i=[input_idx] * len(forward_idx), j=forward_idx)

        # danger_ahead is different from lateral danger: it should suppress forward movement.
        self.S_danger_ahead_forward_inh = Synapses(self.inputs, self.G, on_pre=f"v_post -= {params.w_danger_ahead_forward_inh}*mV")
        danger_ahead_idx_for_inh = self.input_names.index("danger_ahead")
        self.S_danger_ahead_forward_inh.connect(i=[danger_ahead_idx_for_inh] * len(forward_idx), j=forward_idx)

        # Lateral danger should inhibit turning into the danger side.
        self.S_lateral_danger_turn_inh = Synapses(self.inputs, self.G, on_pre=f"v_post -= {params.w_lateral_danger_turn_inh}*mV")
        danger_left_idx = self.input_names.index("danger_left")
        danger_right_idx = self.input_names.index("danger_right")
        motor_left_idx = self._ring_idx("MotorLeft")
        motor_right_idx = self._ring_idx("MotorRight")
        self.S_lateral_danger_turn_inh.connect(i=[danger_left_idx] * len(motor_left_idx), j=motor_left_idx)
        self.S_lateral_danger_turn_inh.connect(i=[danger_right_idx] * len(motor_right_idx), j=motor_right_idx)

        # Direct danger-sensor-to-Escape pathway.
        # This makes visible danger trigger Escape even when the Danger ring itself
        # does not spike strongly in a short control window.
        self.S_danger_sensor_escape = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_danger_sensor_to_escape}*mV")
        escape_idx = self._ring_idx("Escape")
        for input_name in ["danger_left", "danger_right", "danger_ahead", "danger_contact"]:
            input_idx = self.input_names.index(input_name)
            self.S_danger_sensor_escape.connect(i=[input_idx] * len(escape_idx), j=escape_idx)

        # Idle does not directly activate motor behavior.
        # It weakly activates Curiosity, which can then activate Explore.
        self.S_idle_curiosity = None
        self.S_curiosity_motor = None
        if params.enable_modulatory_rings:
            idle_idx = self.input_names.index("idle")
            curiosity_idx = self._ring_idx("Curiosity")
            self.S_idle_curiosity = Synapses(self.inputs, self.G, on_pre=f"v_post += {params.w_idle_to_curiosity}*mV")
            self.S_idle_curiosity.connect(i=[idle_idx] * len(curiosity_idx), j=curiosity_idx)

            # Weak direct pathway: Curiosity can start movement even if Explore is too weak.
            # This prevents pathological Idle -> Curiosity -> rest loops.
            self.S_curiosity_motor = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_curiosity_to_motor_forward}*mV")
            self._connect_ring_to_ring(self.S_curiosity_motor, "Curiosity", "MotorForward", probability=0.20)

        # Transverse modulatory rings proposed in v0.06.
        self.modulatory_synapses = []
        if params.enable_modulatory_rings:
            # Fatigue: real motor activity -> Fatigue -> rest and motor inhibition.
            # Fatigue should accumulate from actual movement, not from Explore/FoodSeeking intention.
            self.S_fatigue_in = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_motor_to_fatigue}*mV")
            self._connect_ring_to_ring_all(self.S_fatigue_in, "MotorLeft", "Fatigue")
            self._connect_ring_to_ring_all(self.S_fatigue_in, "MotorRight", "Fatigue")
            self._connect_ring_to_ring_all(self.S_fatigue_in, "MotorForward", "Fatigue")

            self.S_fatigue_out = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_fatigue_out}*mV")
            self._connect_ring_to_ring_all(self.S_fatigue_out, "Fatigue", "Explore")
            self._connect_ring_to_ring_all(self.S_fatigue_out, "Fatigue", "FoodSeeking")

            self.S_fatigue_motor = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_fatigue_motor_inh}*mV")
            self._connect_ring_to_ring_all(self.S_fatigue_motor, "Fatigue", "MotorLeft")
            self._connect_ring_to_ring_all(self.S_fatigue_motor, "Fatigue", "MotorRight")
            self._connect_ring_to_ring_all(self.S_fatigue_motor, "Fatigue", "MotorForward")

            self.S_fatigue_rest = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_fatigue_to_rest}*mV")
            self._connect_ring_to_ring_all(self.S_fatigue_rest, "Fatigue", "MotorRest")

            # Rest discharges fatigue by inhibiting the Fatigue ring.
            self.S_rest_fatigue_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_rest_to_fatigue_inh}*mV")
            self._connect_ring_to_ring_all(self.S_rest_fatigue_inh, "MotorRest", "Fatigue")

            # Curiosity: suppressed by meaningful states, excites Explore.
            self.S_curiosity_inh = Synapses(self.G, self.G, on_pre=f"v_post -= {params.w_curiosity_inh}*mV")
            self._connect_ring_to_ring_all(self.S_curiosity_inh, "Danger", "Curiosity")
            self._connect_ring_to_ring_all(self.S_curiosity_inh, "FoodSeeking", "Curiosity")
            self._connect_ring_to_ring_all(self.S_curiosity_inh, "EnergyLow", "Curiosity")
            self._connect_ring_to_ring_all(self.S_curiosity_inh, "Fatigue", "Curiosity")
            self._connect_ring_to_ring_all(self.S_curiosity_inh, "MotorRest", "Curiosity")

            self.S_curiosity_out = Synapses(self.G, self.G, on_pre=f"v_post += {params.w_curiosity_to_explore}*mV")
            self._connect_ring_to_ring_all(self.S_curiosity_out, "Curiosity", "Explore")

            self.modulatory_synapses.extend([
                self.S_fatigue_in,
                self.S_fatigue_out,
                self.S_curiosity_inh,
                self.S_curiosity_out,
                self.S_fatigue_motor,
                self.S_fatigue_rest,
                self.S_rest_fatigue_inh,
            ])

        self.spikes = SpikeMonitor(self.G)
        self.state_mon = StateMonitor(self.G, "v", record=True) if self.record else None
        extra_objects = [
            self.S_food_motor,
            self.S_lateral_danger_forward,
            self.S_danger_ahead_forward_inh,
            self.S_lateral_danger_turn_inh,
            self.S_danger_sensor_escape,
            self.S_danger_ahead_turn,
            self.S_escape_forward,
            self.S_danger_forward,
            self.S_contact_threat,
            self.S_danger_escape_rest_inh,
            self.S_danger_escape_fatigue_inh,
            self.S_hunger_rest_inh,
            self.S_energy_explore,
            self.S_foodseeking_forward,
            self.S_explore_motor,
        ]
        if self.S_curiosity_motor is not None:
            extra_objects.append(self.S_curiosity_motor)
        if self.S_idle_curiosity is not None:
            extra_objects.append(self.S_idle_curiosity)

        self.net = Network(
            self.inputs,
            self.G,
            self.S_ring,
            self.S_support,
            self.S_cross_exc,
            self.S_cross_inh,
            self.S_in,
            *extra_objects,
            *self.modulatory_synapses,
            self.spikes,
            *([self.state_mon] if self.state_mon is not None else []),
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

    def _connect_ring_to_ring_all(self, syn: Synapses, src: str, dst: str):
        """Deterministic all-to-all connection between two rings."""
        src_idx = self._ring_idx(src)
        dst_idx = self._ring_idx(dst)
        pairs_i, pairs_j = [], []
        for i in src_idx:
            for j in dst_idx:
                pairs_i.append(i)
                pairs_j.append(j)
        syn.connect(i=pairs_i, j=pairs_j)

    def _connect_input_to_ring(self, input_name: str, ring_name: str):
        input_idx = self.input_names.index(input_name)
        ring_idx = self._ring_idx(ring_name)
        self.S_in.connect(i=[input_idx] * len(ring_idx), j=ring_idx)

    def update_idle_level(self, obs: dict, env: RingsHomeostasisEnv) -> float:
        """
        Update slow idle_level exactly once per control step.

        idle_level grows when there is no food, no danger, no low-energy drive,
        and no wall directly ahead. It decays quickly when a meaningful stimulus
        appears. This value is then used as the idle input to Curiosity.
        """
        energy, food_left, food_ahead, food_right, danger_left, danger_ahead, danger_right = obs["rstate"]
        low_energy = max(
            0.0,
            (self.params.energy_low_threshold - float(energy)) / max(1e-6, self.params.energy_low_threshold),
        )
        food_signal = max(float(food_left), float(food_right), float(food_ahead))
        danger_contact = 1.0 if getattr(env, "danger_contact_signal", 0.0) > 0.0 else 0.0
        danger_signal = max(float(danger_left), float(danger_right), float(danger_ahead), danger_contact)
        wall_ahead = env._wall_ahead()
        idle_condition = (
            food_signal < self.params.idle_food_threshold
            and danger_signal < self.params.idle_danger_threshold
            and float(energy) > self.params.idle_energy_threshold
        )
        if idle_condition:
            env.idle_level = min(1.0, float(env.idle_level) + self.params.idle_rise_per_step)
        else:
            env.idle_level = max(0.0, float(env.idle_level) - self.params.idle_decay_per_step)
        return float(env.idle_level)

    def encode_observation(self, obs: dict, env: RingsHomeostasisEnv) -> np.ndarray:
        """
        Convert local egocentric sensors into SNN input rates.

        rstate now contains:
            energy,
            food_left, food_ahead, food_right,
            danger_left, danger_ahead, danger_right
        This function must be side-effect free.
        """
        energy, food_left, food_ahead, food_right, danger_left, danger_ahead, danger_right = obs["rstate"]
        low_energy = max(
            0.0,
            (self.params.energy_low_threshold - float(energy)) / max(1e-6, self.params.energy_low_threshold),
        )
        wall_ahead = env._wall_ahead()
        danger_contact = 1.0 if getattr(env, "danger_contact_signal", 0.0) > 0.0 else 0.0
        idle = float(env.idle_level) if hasattr(env, "idle_level") else 0.0

        rates = np.array(
            [
                low_energy,
                food_left,
                food_right,
                food_ahead,
                danger_left * self.params.danger_input_gain,
                danger_right * self.params.danger_input_gain,
                danger_ahead * self.params.danger_input_gain,
                wall_ahead,
                idle,
                danger_contact,
            ],
            dtype=float,
        )
        return np.clip(rates, 0.0, 1.0) * self.params.input_rate_hz

    def step(self, obs: dict, env: RingsHomeostasisEnv) -> Tuple[int, Dict[str, float]]:
        self.update_idle_level(obs, env)
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
        """
        Decode action primarily from motor-ring activity.

        MiniGrid actions:
            0 left, 1 right, 2 forward.
        """
        input_features = self.encode_observation(obs, env) / self.params.input_rate_hz
        _, food_left, food_right, food_ahead, danger_left, danger_right, danger_ahead, wall_ahead, _idle, danger_contact = input_features

        motor_scores = {
            0: acts.get("MotorLeft", 0.0),
            1: acts.get("MotorRight", 0.0),
            2: acts.get("MotorForward", 0.0),
            6: acts.get("MotorRest", 0.0),
        }

        # Wall avoidance is kept as an action-feasibility constraint, not as a learned skill.
        # The agent cannot benefit from trying to walk into a wall.
        if wall_ahead > 0.5:
            motor_scores[2] -= 10.0
            # Facing a wall should not suppress Idle/Curiosity. Instead, if the
            # agent is in exploratory/curiosity mode, bias turning to escape the wall.
            exploratory_drive = acts.get("Explore", 0.0) + acts.get("Curiosity", 0.0)
            if exploratory_drive > 0.001:
                if motor_scores[0] <= 0 and motor_scores[1] <= 0:
                    if random.random() < 0.5:
                        motor_scores[0] += exploratory_drive + 0.5
                    else:
                        motor_scores[1] += exploratory_drive + 0.5
                else:
                    motor_scores[0] += 0.25 * exploratory_drive
                    motor_scores[1] += 0.25 * exploratory_drive
            elif motor_scores[0] == 0 and motor_scores[1] == 0 and motor_scores[6] == 0:
                return 0 if random.random() < 0.5 else 1

        # No hard-coded danger avoidance here.
        # Danger must affect actions through the SNN:
        # danger sensors -> Danger/Escape rings -> MotorLeft/MotorRight/MotorForward.

        # Optional weak food-forward bias. For a stricter test set food_forward_bias=0.
        # No decoder-level food movement rule here.
        # Food-guided movement is handled only by SNN pathways:
        # food_left/right/ahead -> MotorLeft/MotorRight/MotorForward.

        # Fatigue can force a resting action via MotorRest. This is still an SNN-mediated
        # decision: Fatigue activates MotorRest and inhibits other motor rings.
        if acts.get("Fatigue", 0.0) > self.params.fatigue_rest_threshold:
            motor_scores[6] += acts.get("Fatigue", 0.0)

        # If no motor ring is active, do nothing/rest.
        # This makes motor behavior fully dependent on SNN motor rings.
        if max(motor_scores.values()) <= 0.001:
            return 6

        # Argmax action with deterministic tie-breaking preference:
        # rest wins when fatigue/motor-rest is clearly active; otherwise forward can win
        # if it is close to left/right because food collection requires entering the cell.
        best_action = max(motor_scores, key=motor_scores.get)
        if best_action == 6 and motor_scores[6] >= max(motor_scores[0], motor_scores[1], motor_scores[2]) - 0.02:
            return 6
        if motor_scores[2] >= max(motor_scores[0], motor_scores[1]) - 0.05 and wall_ahead <= 0.5:
            return 2
        return best_action


# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------

class RingsVisualizer:
    def __init__(self, controller: RingsSNNController):
        self.controller = controller
        self.fig = plt.figure(figsize=(18, 5))
        self.ax_env = self.fig.add_subplot(1, 5, 1)
        self.ax_sensors = self.fig.add_subplot(1, 5, 2)
        self.ax_rings = self.fig.add_subplot(1, 5, 3)
        self.ax_timeline = self.fig.add_subplot(1, 5, (4, 5))
        self.history: List[Dict[str, float]] = []
        self.timeline_history: List[Dict[str, float]] = []
        plt.ion()
        plt.show(block=False)

    def update(self, env: RingsHomeostasisEnv, acts: Dict[str, float], step_idx: int, info: dict):
        self.history.append(dict(acts))
        self.history = self.history[-20:]

        self.ax_env.clear()
        frame = env.render()
        self.ax_env.imshow(frame)
        self.ax_env.axis("off")
        action_names = {
            0: "turn_left",
            1: "turn_right",
            2: "forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done",
            None: "none",
        }
        dominant_ring = max(acts, key=acts.get) if acts else "none"
        last_action = action_names.get(info.get("last_action"), "unknown")
        title = "MiniGrid agent v0.32\n" + f"step={step_idx}, energy={info.get('energy', 0):.2f}\n" + f"food={info.get('food_collected', 0)}, danger={info.get('danger_contacts', 0)}\n" + f"action={last_action}, state={dominant_ring}"
        self.ax_env.set_title(title)

        # Sensor panel.
        self.ax_sensors.clear()
        self.ax_sensors.set_title("Local sensors")
        self.ax_sensors.set_ylim(0, 1.05)
        self.ax_sensors.set_ylabel("intensity")

        obs = env._get_obs()
        energy, food_left, food_ahead, food_right, danger_left, danger_ahead, danger_right = obs["rstate"]
        wall_ahead = env._wall_ahead()
        low_energy = max(
            0.0,
            (self.controller.params.energy_low_threshold - float(energy)) / max(1e-6, self.controller.params.energy_low_threshold),
        )
        food_signal = max(float(food_left), float(food_right), float(food_ahead))
        danger_signal = max(float(danger_left), float(danger_right), float(danger_ahead))
        idle_condition = (
            food_signal < self.controller.params.food_threshold
            and danger_signal < self.controller.params.danger_threshold
            and float(energy) > self.controller.params.idle_energy_threshold
            and wall_ahead < 0.5
        )
        idle = float(env.idle_level) if hasattr(env, "idle_level") else (1.0 if idle_condition else 0.0)
        self.timeline_history.append(
            {
                "food_left": float(food_left),
                "food_ahead": float(food_ahead),
                "food_right": float(food_right),
                "danger_left": float(danger_left),
                "danger_ahead": float(danger_ahead),
                "danger_right": float(danger_right),
                "low_energy": float(low_energy),
                "wall_ahead": float(wall_ahead),
                "idle": idle,
                "Curiosity": acts.get("Curiosity", 0.0),
                "Explore": acts.get("Explore", 0.0),
                "MotorLeft": acts.get("MotorLeft", 0.0),
                "MotorRight": acts.get("MotorRight", 0.0),
                "MotorForward": acts.get("MotorForward", 0.0),
                "MotorRest": acts.get("MotorRest", 0.0),
            }
        )
        self.timeline_history = self.timeline_history[-60:]

        labels = [
            "food L", "food A", "food R",
            "danger L", "danger A", "danger R", "danger contact",
            "energy", "wall A",
        ]
        danger_contact_value = float(getattr(env, "danger_contact_signal", 0.0))
        values = [
            food_left, food_ahead, food_right,
            danger_left, danger_ahead, danger_right, danger_contact_value,
            energy, wall_ahead,
        ]
        bar_colors = ["tab:blue", "tab:blue", "tab:blue", "tab:red", "tab:red", "tab:red", "tab:red", "tab:green", "tab:gray"]
        self.ax_sensors.bar(range(len(labels)), values, color=bar_colors)
        self.ax_sensors.set_xticks(range(len(labels)))
        self.ax_sensors.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        for i, v in enumerate(values):
            self.ax_sensors.text(i, min(1.02, float(v) + 0.03), f"{float(v):.2f}", ha="center", fontsize=7)

        self.ax_rings.clear()
        self.ax_rings.set_aspect("equal")
        self.ax_rings.axis("off")
        self.ax_rings.set_title("RINGS state activity")

        ring_names = list(self.controller.rings.keys())
        centers = {
            "Explore": (-1.6, 1.1),
            "EnergyLow": (0.0, 1.1),
            "FoodSeeking": (1.6, 1.1),
            "Danger": (-1.6, -0.2),
            "Escape": (0.0, -0.2),
            "MotorLeft": (-1.6, -1.5),
            "MotorForward": (0.0, -1.5),
            "MotorRight": (1.6, -1.5),
            "MotorRest": (0.0, -2.35),
            "Fatigue": (-0.8, -3.05),
            "Curiosity": (0.8, -3.05),
        }
        for name in ring_names:
            x, y = centers[name]
            a = min(1.0, acts[name] / 2.0)
            circle = plt.Circle((x, y), 0.45, fill=True, alpha=0.15 + 0.75 * a)
            self.ax_rings.add_patch(circle)
            self.ax_rings.text(x, y, f"{name}\n{acts[name]:.2f}", ha="center", va="center", fontsize=9)

        self.ax_rings.set_xlim(-2.4, 2.4)
        self.ax_rings.set_ylim(-3.6, 1.7)


        # Lane-based temporal diagram for Idle -> Curiosity -> Explore -> Motor activity.
        self.ax_timeline.clear()
        self.ax_timeline.set_title("Temporal diagram")
        if self.timeline_history:
            lanes = [
                ("Food sensors", lambda h: max(h.get("food_left", 0.0), h.get("food_ahead", 0.0), h.get("food_right", 0.0))),
                ("Danger sensors", lambda h: max(h.get("danger_left", 0.0), h.get("danger_ahead", 0.0), h.get("danger_right", 0.0))),
                ("Low energy", lambda h: h.get("low_energy", 0.0)),
                ("Wall ahead", lambda h: h.get("wall_ahead", 0.0)),
                ("IDLE", lambda h: h.get("idle", 0.0)),
                ("EnergyLow", lambda h: h.get("EnergyLow", 0.0)),
                ("FoodSeeking", lambda h: h.get("FoodSeeking", 0.0)),
                ("Danger", lambda h: h.get("Danger", 0.0)),
                ("Escape", lambda h: h.get("Escape", 0.0)),
                ("Explore", lambda h: h.get("Explore", 0.0)),
                ("Curiosity", lambda h: h.get("Curiosity", 0.0)),
                ("Fatigue", lambda h: h.get("Fatigue", 0.0)),
                ("MotorLeft", lambda h: h.get("MotorLeft", 0.0)),
                ("MotorRight", lambda h: h.get("MotorRight", 0.0)),
                ("MotorForward", lambda h: h.get("MotorForward", 0.0)),
                ("MotorRest", lambda h: h.get("MotorRest", 0.0)),
            ]
            xs2 = np.arange(len(self.timeline_history))
            n_lanes = len(lanes)
            for lane_idx, (label, getter) in enumerate(lanes):
                y_base = n_lanes - 1 - lane_idx
                values = np.array([float(getter(h)) for h in self.timeline_history], dtype=float)
                if values.size and np.max(values) > 1.0:
                    values = values / max(1e-6, np.max(values))
                self.ax_timeline.plot(xs2, y_base + 0.65 * values, linewidth=1.6)
                self.ax_timeline.axhline(y_base, linewidth=0.5, alpha=0.35)
            self.ax_timeline.set_yticks(range(n_lanes))
            self.ax_timeline.set_yticklabels([label for label, _ in reversed(lanes)], fontsize=7)
            self.ax_timeline.set_xlabel("recent steps")
            self.ax_timeline.set_ylim(-0.5, n_lanes + 0.3)
            self.ax_timeline.grid(axis="x", alpha=0.2)

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
    rest_steps: int


def run_episode(
    seed: int,
    params: Optional[RingsParams] = None,
    visualize: bool = True,
    max_steps: int = 250,
    status: bool = False,
    record: bool = False,
    debug_log: Optional[str] = None,
    grid_size: int = 15,
    danger_speed: float = 0.0,
) -> EpisodeResult:
    params = params or RingsParams()
    env = RingsHomeostasisEnv(size=grid_size, max_steps=max_steps, sensor_range=params.sensor_range, danger_speed=danger_speed, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    controller = RingsSNNController(params=params, seed=seed, record=record)
    vis = RingsVisualizer(controller) if visualize else None

    debug_file = None
    debug_writer = None
    if debug_log:
        debug_path = Path(debug_log)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path.open("w", newline="", encoding="utf-8")
        debug_fields = [
            "step",
            "energy",
            "food_left", "food_ahead", "food_right",
            "danger_left", "danger_ahead", "danger_right",
            "food_signal", "danger_signal", "low_energy", "wall_ahead",
            "is_no_food", "is_no_danger", "is_energy_ok", "idle_condition", "idle_energy_threshold", "idle_level",
            "Explore", "EnergyLow", "FoodSeeking", "Danger", "Escape",
            "Curiosity", "Fatigue",
            "MotorLeft", "MotorRight", "MotorForward", "MotorRest",
            "action", "last_event", "food_collected", "danger_contacts",
        ]
        debug_writer = csv.DictWriter(debug_file, fieldnames=debug_fields)
        debug_writer.writeheader()

    total_reward = 0.0
    terminated = truncated = False
    step_idx = 0
    rest_steps = 0

    while not (terminated or truncated):
        action, acts = controller.step(obs, env)
        if action == 6:
            rest_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if debug_writer is not None:
            dbg_obs = env._get_obs()
            energy, food_left, food_ahead, food_right, danger_left, danger_ahead, danger_right = dbg_obs["rstate"]
            low_energy = max(
                0.0,
                (params.energy_low_threshold - float(energy)) / max(1e-6, params.energy_low_threshold),
            )
            food_signal = max(float(food_left), float(food_ahead), float(food_right))
            danger_signal = max(float(danger_left), float(danger_ahead), float(danger_right))
            wall_ahead = env._wall_ahead()
            is_no_food = food_signal < params.food_threshold
            is_no_danger = danger_signal < params.danger_threshold
            is_energy_ok = float(energy) > params.idle_energy_threshold
            idle_condition = is_no_food and is_no_danger and is_energy_ok
            action_names = {
                0: "turn_left", 1: "turn_right", 2: "forward", 3: "pickup",
                4: "drop", 5: "toggle", 6: "rest", None: "none",
            }
            debug_writer.writerow({
                "step": step_idx,
                "energy": float(energy),
                "food_left": float(food_left),
                "food_ahead": float(food_ahead),
                "food_right": float(food_right),
                "danger_left": float(danger_left),
                "danger_ahead": float(danger_ahead),
                "danger_right": float(danger_right),
                "food_signal": float(food_signal),
                "danger_signal": float(danger_signal),
                "low_energy": float(low_energy),
                "wall_ahead": float(wall_ahead),
                "is_no_food": int(is_no_food),
                "is_no_danger": int(is_no_danger),
                "is_energy_ok": int(is_energy_ok),
                "idle_condition": int(idle_condition),
                "idle_energy_threshold": float(params.idle_energy_threshold),
                "idle_level": float(env.idle_level),
                "Explore": float(acts.get("Explore", 0.0)),
                "EnergyLow": float(acts.get("EnergyLow", 0.0)),
                "FoodSeeking": float(acts.get("FoodSeeking", 0.0)),
                "Danger": float(acts.get("Danger", 0.0)),
                "Escape": float(acts.get("Escape", 0.0)),
                "Curiosity": float(acts.get("Curiosity", 0.0)),
                "Fatigue": float(acts.get("Fatigue", 0.0)),
                "MotorLeft": float(acts.get("MotorLeft", 0.0)),
                "MotorRight": float(acts.get("MotorRight", 0.0)),
                "MotorForward": float(acts.get("MotorForward", 0.0)),
                "MotorRest": float(acts.get("MotorRest", 0.0)),
                "action": action_names.get(info.get("last_action"), "unknown"),
                "last_event": info.get("last_event", ""),
                "food_collected": int(info.get("food_collected", 0)),
                "danger_contacts": int(info.get("danger_contacts", 0)),
            })
            debug_file.flush()

        if vis is not None:
            vis.update(env, acts, step_idx, info)
        elif status:
            action_names = {
                0: "turn_left",
                1: "turn_right",
                2: "forward",
                3: "pickup",
                4: "drop",
                5: "toggle",
                6: "rest",
                None: "none",
            }
            print(
                f"\rstep {step_idx}: energy={info.get('energy', 0):.3f}, "
                f"food={info.get('food_collected', 0)}, "
                f"danger={info.get('danger_contacts', 0)}, "
                f"action={action_names.get(info.get('last_action'), 'unknown')}     ",
                end="",
                flush=True,
            )

        step_idx += 1

    result = EpisodeResult(
        steps=step_idx,
        total_reward=float(total_reward),
        final_energy=float(info.get("energy", 0.0)),
        food_collected=int(info.get("food_collected", 0)),
        danger_contacts=int(info.get("danger_contacts", 0)),
        rest_steps=int(rest_steps),
    )
    env.close()

    if debug_file is not None:
        debug_file.close()

    if status and not visualize:
        print()

    if visualize:
        plt.ioff()
        plt.show()

    return result


def evaluate_params(
    params: RingsParams,
    seeds: List[int],
    max_steps: int = 250,
    status: bool = False,
    trial: int | None = None,
    record: bool = False,
    grid_size: int = 15,
    danger_speed: float = 0.0,
) -> Dict[str, float]:
    results = []
    for idx, s in enumerate(seeds, start=1):
        if status:
            prefix = f"trial {trial} seed {idx}/{len(seeds)}" if trial is not None else f"seed {idx}/{len(seeds)}"
            print(f"\n{prefix}")
        results.append(run_episode(seed=s, params=params, visualize=False, max_steps=max_steps, status=status, record=record, grid_size=grid_size, danger_speed=danger_speed))
    steps = np.array([r.steps for r in results], dtype=float)
    rewards = np.array([r.total_reward for r in results], dtype=float)
    foods = np.array([r.food_collected for r in results], dtype=float)
    dangers = np.array([r.danger_contacts for r in results], dtype=float)
    rest_ratios = np.array([r.rest_steps / max(1, r.steps) for r in results], dtype=float)

    # Main criterion: homeostatic success, not raw lifespan.
    # Food collection is the primary positive signal.
    # Danger contacts are the primary negative signal.
    # Lifespan remains only as a weak secondary term, so random wandering does not dominate.
    food_weight = 60.0
    danger_weight = 45.0
    step_weight = 0.20
    reward_weight = 0.10
    rest_penalty_weight = 80.0

    score = float(
        food_weight * np.mean(foods)
        - danger_weight * np.mean(dangers)
        + step_weight * np.mean(steps)
        + reward_weight * np.mean(rewards)
        - rest_penalty_weight * np.mean(rest_ratios)
    )
    return {
        "score": score,
        "food_weight": food_weight,
        "danger_weight": danger_weight,
        "step_weight": step_weight,
        "reward_weight": reward_weight,
        "rest_penalty_weight": rest_penalty_weight,
        "mean_rest_ratio": float(np.mean(rest_ratios)),
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
    record: bool = False,
    grid_size: int = 15,
    danger_speed: float = 0.0,
) -> Tuple[RingsParams, Dict[str, float]]:
    seeds = seeds or [1, 2, 3]
    rng = random.Random(search_seed)

    best_params: Optional[RingsParams] = None
    best_metrics: Optional[Dict[str, float]] = None
    all_results = []

    for t in range(1, trials + 1):
        params = sample_params(rng)
        metrics = evaluate_params(params, seeds=seeds, max_steps=max_steps, status=True, trial=t, record=record, grid_size=grid_size, danger_speed=danger_speed)
        row = {"trial": t, "params": asdict(params), "metrics": metrics}
        all_results.append(row)

        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_params = params
            best_metrics = metrics
            print(f"[trial {t:03d}] NEW BEST score={metrics['score']:.2f}, food={metrics['mean_food']:.2f}, danger={metrics['mean_danger']:.2f}, rest={metrics['mean_rest_ratio']:.2f}, mean_steps={metrics['mean_steps']:.1f}")
            print(json.dumps(asdict(params), indent=2))

            # Save every new best immediately so long remote runs can be inspected
            # before the full tuning process finishes.
            interim_payload = {
                "best_params": asdict(best_params),
                "best_metrics": best_metrics,
                "best_trial": t,
            }
            Path(output_path).write_text(json.dumps(interim_payload, indent=2), encoding="utf-8")
            print(f"Saved current best to {output_path}")
        else:
            print(f"[trial {t:03d}] score={metrics['score']:.2f}, food={metrics['mean_food']:.2f}, danger={metrics['mean_danger']:.2f}, rest={metrics['mean_rest_ratio']:.2f}, mean_steps={metrics['mean_steps']:.1f}")

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
    parser.add_argument("--record", choices=["on", "off"], default="off")
    parser.add_argument("--debug-log", type=str, default=None, help="Write per-step diagnostic CSV log to this path")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--danger-speed", type=float, default=0.0)
    args = parser.parse_args()

    record = args.record == "on"

    if args.mode == "tune":
        tune_random_search(
            trials=args.trials,
            seeds=[1, 2, 3],
            max_steps=args.max_steps,
            search_seed=args.seed,
            output_path=args.output,
            record=record,
            grid_size=args.grid_size,
            danger_speed=args.danger_speed,
        )
    else:
        params = load_params(args.params) if args.params else RingsParams()
        visualize = args.visual == "on"
        result = run_episode(seed=args.seed, params=params, visualize=visualize, max_steps=args.max_steps, status=not visualize, record=record, debug_log=args.debug_log, grid_size=args.grid_size, danger_speed=args.danger_speed)
        print(result)


if __name__ == "__main__":
    main()
