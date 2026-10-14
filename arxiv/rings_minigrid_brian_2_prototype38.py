"""
RINGS / TEVSER hybrid prototype v0.38.0 (based on v0.37.2).
Fast recurrent rings: EscapeSwitch, MotorLeft/Right/Forward/Rest.
Slow chemical fields: ForagingMod, DefenseMod, FatigueMod, CuriosityMod.
Each field is secreted ONLY by spikes of its dedicated LIF neuron in Brian2:
    dc/dt = -c/tau_chem; on spike c += release*(1-c).
Concentrations are dimensionless fractions of a finite release capacity [0,1].
Receptor occupancy c/(Kd+c) continuously changes excitability and sensory gain.
Biological analogues in labels are functional inspirations, not biochemical claims.
EnergyLow remains a homeostatic scalar. Danger has four fast interneurons.
All chemical time constants are specified in environment control steps; actual
Brian time is tau_steps*snn_step_ms. No Python concentration update or action
rule based directly on chemical levels. Idle is a sensory absence accumulator.
Tune retains spawn workers, shared seeds, atomic checkpoints, resume and telemetry.
New architecture requires new tuning; v0.37 checkpoints/parameters are rejected.

Examples:
    python3 rings_minigrid_brian_2_prototype38.py --visual on --max-steps 250
    python3 rings_minigrid_brian_2_prototype38.py --mode tune --workers 32 --trials 100 --output rings_v038_best.json
    python3 rings_minigrid_brian_2_prototype38.py --params rings_v038_best.json --visual on

Default chemical decay constants (in control steps): Foraging 20, Defense 10,
Fatigue 40, Curiosity 25. These are initial modelling choices, not fitted data.
Fast ring recurrence is delayed beyond refractory and limited by adaptation.
An isolated default ring impulse produces a burst over two control windows;
continuous sensory or chemical drive can maintain activity for longer.
CSV rows align pre-action input, network output and resulting energy separately.
record=off keeps spike counts only; record=on additionally retains full spikes,
ring membrane potentials and chemical traces on the controller monitors.
"""

from __future__ import annotations

import argparse
import signal
import sys
import warnings

# Harmless distutils compiler-discovery warning; do not hide other warnings.
warnings.filterwarnings(
    "ignore", message=r"^_get_vc_env is private; find an alternative \(pypa/distutils#340\)$",
    category=UserWarning, module=r"^(setuptools\._distutils|distutils)\._msvccompiler$",
)
import time
import queue
import json
import random
from dataclasses import dataclass, asdict
import os
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
    linked_var,
    seed as brian_seed,
)

prefs.codegen.target = "numpy"

AGENT_VERSION = "0.38.0"


# -----------------------------------------------------------------------------
# 1. Tunable parameter set
# -----------------------------------------------------------------------------

@dataclass
class RingsParams:
    ring_size: int = 12
    snn_step_ms: int = 30
    w_ring: float = 19.0
    w_support: float = 4.0
    ring_delay_ms: float = 5.0
    adaptation_tau_steps: float = 2.0
    adaptation_jump_mv: float = 1.5
    w_cross_inh: float = 2.0
    w_input: float = 12.0
    w_food_sensor_to_motor: float = 16.0
    w_lateral_food_sensor_to_motor: float = 24.0
    w_lateral_danger_to_forward: float = 7.0
    w_danger_sensor_to_escape: float = 14.0
    w_danger_ahead_to_turn: float = 18.0
    w_danger_ahead_forward_inh: float = 22.0
    w_lateral_danger_turn_inh: float = 12.0
    w_escape_to_turn: float = 1.0
    w_escape_to_motor_forward: float = 0.3
    danger_input_gain: float = 1.8
    energy_low_threshold: float = 0.55
    idle_food_threshold: float = 0.10
    idle_danger_threshold: float = 0.10
    idle_energy_threshold: float = 0.35
    idle_rise_per_step: float = 0.20
    idle_decay_per_step: float = 0.50
    input_rate_hz: float = 220.0
    sensor_range: int = 3
    mod_neuron_tau_ms: float = 40.0
    mod_refractory_ms: float = 15.0
    foraging_tau_steps: float = 20.0
    defense_tau_steps: float = 10.0
    fatigue_tau_steps: float = 40.0
    curiosity_tau_steps: float = 25.0
    foraging_release: float = 0.10
    defense_release: float = 0.15
    fatigue_release: float = 0.06
    curiosity_release: float = 0.10
    receptor_kd: float = 0.25
    w_sensor_to_mod: float = 8.0
    w_idle_to_mod: float = 8.0
    w_motor_to_mod: float = 7.0  # normalized by ring size
    w_escape_to_mod: float = 8.0  # normalized by ring size
    w_rest_to_mod_inh: float = 6.0
    foraging_gain: float = 1.5
    defense_gain: float = 1.5
    foraging_drive_mv: float = 17.0
    curiosity_drive_mv: float = 19.0
    fatigue_inhibition_mv: float = 16.0
    fatigue_rest_drive_mv: float = 24.0
    defense_rest_inhibition_mv: float = 20.0
    noise_rate_hz: float = 70.0
    noise_weight_mv: float = 4.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.ring_delay_ms <= 3:
            raise ValueError("ring_delay_ms must exceed the 3 ms refractory period")
        if self.ring_size < 3 or int(self.ring_size) != self.ring_size:
            raise ValueError("ring_size must be an integer >= 3")
        if int(self.sensor_range) != self.sensor_range:
            raise ValueError("sensor_range must be an integer")
        for name in MOD_NAMES:
            prefix = name[:-3].lower()
            if getattr(self, prefix + "_release") > 1:
                raise ValueError("release fractions must be <= 1")
            if getattr(self, prefix + "_tau_steps") <= self.adaptation_tau_steps:
                raise ValueError("chemical decay must be slower than ring adaptation")


MOD_NAMES = ("ForagingMod", "DefenseMod", "FatigueMod", "CuriosityMod")
MOD_ANALOGUES = ("octopamine-like", "tyramine-like", "adenosine-like", "dopamine-like")


def sample_params(rng: random.Random) -> RingsParams:
    """Search only parameters used by this hybrid architecture."""
    return RingsParams(
        ring_size=rng.choice([8, 12, 16]), snn_step_ms=rng.choice([20, 30, 40]),
        w_ring=rng.uniform(17, 23), w_support=rng.uniform(3, 5),
        ring_delay_ms=rng.uniform(4, 7), adaptation_tau_steps=rng.uniform(1, 3),
        adaptation_jump_mv=rng.uniform(1, 2.5), w_cross_inh=rng.uniform(1, 4),
        w_input=rng.uniform(10, 18), w_food_sensor_to_motor=rng.uniform(12, 24),
        w_lateral_food_sensor_to_motor=rng.uniform(18, 32),
        w_lateral_danger_to_forward=rng.uniform(5, 12),
        w_danger_sensor_to_escape=rng.uniform(12, 20),
        w_danger_ahead_to_turn=rng.uniform(14, 26),
        w_danger_ahead_forward_inh=rng.uniform(18, 32),
        w_lateral_danger_turn_inh=rng.uniform(8, 18),
        danger_input_gain=rng.uniform(1.2, 2.5),
        energy_low_threshold=rng.uniform(.4, .75), input_rate_hz=rng.uniform(150, 300),
        sensor_range=rng.choice([2, 3]),
        foraging_tau_steps=rng.uniform(10, 40), defense_tau_steps=rng.uniform(5, 20),
        fatigue_tau_steps=rng.uniform(25, 80), curiosity_tau_steps=rng.uniform(15, 60),
        foraging_release=rng.uniform(.05, .18), defense_release=rng.uniform(.08, .25),
        fatigue_release=rng.uniform(.02, .10), curiosity_release=rng.uniform(.05, .18),
        w_sensor_to_mod=rng.uniform(5, 12), w_idle_to_mod=rng.uniform(5, 12),
        w_motor_to_mod=rng.uniform(4, 10), w_escape_to_mod=rng.uniform(5, 12),
        foraging_gain=rng.uniform(.5, 2), defense_gain=rng.uniform(.5, 2),
        foraging_drive_mv=rng.uniform(12, 22), curiosity_drive_mv=rng.uniform(14, 24),
        fatigue_inhibition_mv=rng.uniform(10, 22), fatigue_rest_drive_mv=rng.uniform(20, 30),
    )


# -----------------------------------------------------------------------------
# 2. MiniGrid environment: homeostatic bug world
# -----------------------------------------------------------------------------

class Food(Ball):
    """Food object. Using Ball rendering but semantically it restores energy."""

    def __init__(self):
        super().__init__(color="green")
        self.type = "food"

    def encode(self):
        # Keep semantic self.type for RINGS; standard MiniGrid image uses Ball.
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        return (OBJECT_TO_IDX["ball"], COLOR_TO_IDX[self.color], 0)

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

    def encode(self):
        # Keep semantic self.type for RINGS; standard MiniGrid image uses Ball.
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        return (OBJECT_TO_IDX["ball"], COLOR_TO_IDX[self.color], 0)

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
        agent_view_size = max(3, 2 * self.sensor_range - 1)
        if agent_view_size % 2 == 0:
            agent_view_size += 1

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            agent_view_size=agent_view_size,
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
    """Fast recurrent attractor candidates coupled to spike-secreted slow fields."""

    def __init__(self, params: RingsParams, seed: int = 1, record: bool = False):
        start_scope()
        brian_seed(seed)
        random.seed(seed)
        self.params, self.record = params, record
        self.snn_step_ms, self.ring_size = params.snn_step_ms, params.ring_size
        self.rings = {}
        for k, name in enumerate(("EscapeSwitch", "MotorLeft", "MotorRight", "MotorForward", "MotorRest")):
            self.rings[name] = RingSpec(name, params.ring_size, k*params.ring_size, (k+1)*params.ring_size)
        self.n_state = len(self.rings)*params.ring_size
        self.input_names = ["low_energy", "food_left", "food_right", "food_ahead",
                            "danger_left", "danger_right", "danger_ahead", "wall_ahead", "idle", "danger_contact"]
        self.inputs = PoissonGroup(len(self.input_names), rates=0*Hz)
        # One secreting LIF neuron per chemical. Only its reset releases chemical.
        self.modulators = NeuronGroup(4, """
            dv/dt = (-65*mV-v)/tau_v : volt (unless refractory)
            dc/dt = -c/tau_chem : 1
            tau_v : second (constant)
            tau_chem : second (constant)
            release : 1 (constant)
            refractory_period : second (constant)
        """, threshold="v > -50*mV", reset="v = -65*mV; c += release*(1-c)",
            refractory="refractory_period", method="exact")
        self.modulators.v = -65*mV
        self.modulators.tau_v = params.mod_neuron_tau_ms*ms
        self.modulators.refractory_period = params.mod_refractory_ms*ms
        for k, name in enumerate(MOD_NAMES):
            prefix = name[:-3].lower()
            self.modulators.tau_chem[k] = getattr(params, prefix+"_tau_steps")*params.snn_step_ms*ms
            self.modulators.release[k] = getattr(params, prefix+"_release")
        self.G = NeuronGroup(self.n_state, """
            dv/dt = (bias + drive - v)/tau_v : volt (unless refractory)
            da/dt = -a/tau_a : volt
            drive = move*(forage_drive*F*(1-D) + curious_drive*Q*(1-D)*(1-H)
                          - fatigue_inh*H*(1-D))
                    + rest*(rest_drive*H - defense_inh*D - forage_drive*F) : volt
            F = cf/(kd+cf) : 1 (constant over dt)
            D = cd/(kd+cd) : 1 (constant over dt)
            H = ch/(kd+ch) : 1 (constant over dt)
            Q = cq/(kd+cq) : 1 (constant over dt)
            cf : 1 (linked)
            cd : 1 (linked)
            ch : 1 (linked)
            cq : 1 (linked)
            move : 1 (constant)
            rest : 1 (constant)
            bias : volt (constant)
            tau_v : second (constant)
            tau_a : second (constant)
        """, threshold="v > -50*mV + a", reset="v = -65*mV; a += adapt_jump",
            refractory=3*ms, method="euler", namespace={
                "kd": params.receptor_kd, "forage_drive": params.foraging_drive_mv*mV,
                "curious_drive": params.curiosity_drive_mv*mV,
                "fatigue_inh": params.fatigue_inhibition_mv*mV,
                "rest_drive": params.fatigue_rest_drive_mv*mV,
                "defense_inh": params.defense_rest_inhibition_mv*mV,
                "adapt_jump": params.adaptation_jump_mv*mV})
        for k, var in enumerate(("cf", "cd", "ch", "cq")):
            setattr(self.G, var, linked_var(self.modulators, "c", index=np.full(self.n_state, k)))
        self.G.v = -65*mV
        self.G.bias = -65*mV
        self.G.tau_v = 15*ms
        self.G.tau_a = params.adaptation_tau_steps*params.snn_step_ms*ms
        for name in ("MotorLeft", "MotorRight", "MotorForward"):
            self.G.move[self._ring_idx(name)] = 1
        self.G.rest[self._ring_idx("MotorRest")] = 1
        # Four fast threat channels preserve direction; no global ThreatLevel state.
        self.danger = NeuronGroup(4, "dv/dt=(-65*mV-v)/(8*ms):volt (unless refractory)",
                                 threshold="v>-50*mV", reset="v=-65*mV", refractory=2*ms, method="exact")
        self.danger.v = -65*mV
        self.connections = []
        def connect(source, target, pairs, weight, effect="1", delay=0):
            syn = Synapses(source, target, "w:volt", on_pre=f"v_post += w*({effect})",
                           namespace={"fg": params.foraging_gain, "dg": params.defense_gain})
            ii, jj = zip(*pairs)
            syn.connect(i=list(ii), j=list(jj))
            syn.w = weight*mV
            syn.delay = delay*ms
            self.connections.append(syn)
            return syn
        def to_ring(source, idx, name, weight, effect="1"):
            return connect(source, self.G, [(idx,j) for j in self._ring_idx(name)], weight, effect)
        # Local delayed recurrence plus spike-frequency adaptation: no fixed episode latch.
        for offset, weight in ((1,params.w_ring),(2,params.w_support)):
            pairs = [(i, r.start+(i-r.start+offset)%r.n) for r in self.rings.values() for i in range(r.start,r.end)]
            connect(self.G,self.G,pairs,weight,delay=params.ring_delay_ms)
        motors = ("MotorLeft", "MotorRight", "MotorForward", "MotorRest")
        pairs = [(i,j) for src in motors for dst in motors if src!=dst
                 for i in self._ring_idx(src) for j in self._ring_idx(dst)]
        connect(self.G,self.G,pairs,-params.w_cross_inh/params.ring_size)
        for name, weight in (("MotorLeft",params.w_escape_to_turn),("MotorRight",params.w_escape_to_turn),
                             ("MotorForward",params.w_escape_to_motor_forward)):
            connect(self.G,self.G,[(i,j) for i in self._ring_idx("EscapeSwitch") for j in self._ring_idx(name)],weight/params.ring_size)
        for sensor, ring in (("food_left","MotorLeft"),("food_right","MotorRight"),("food_ahead","MotorForward")):
            w = params.w_food_sensor_to_motor if sensor=="food_ahead" else params.w_lateral_food_sensor_to_motor
            to_ring(self.inputs,self.input_names.index(sensor),ring,w,"(1+fg*F_post)*(1-0.8*D_post)")
        for k, sensor in enumerate(("danger_left","danger_right","danger_ahead","danger_contact")):
            connect(self.inputs,self.danger,[(self.input_names.index(sensor),k)],params.w_input*1.5)
            to_ring(self.danger,k,"EscapeSwitch",params.w_danger_sensor_to_escape,"1+dg*D_post")
            connect(self.danger,self.modulators,[(k,1)],params.w_sensor_to_mod)
        for k, away, toward in ((0,"MotorRight","MotorLeft"),(1,"MotorLeft","MotorRight")):
            to_ring(self.danger,k,away,params.w_input,"1+dg*D_post")
            to_ring(self.danger,k,toward,-params.w_lateral_danger_turn_inh,"1+dg*D_post")
            to_ring(self.danger,k,"MotorForward",params.w_lateral_danger_to_forward,"1+dg*D_post")
        for name in ("MotorLeft","MotorRight"):
            to_ring(self.danger,2,name,params.w_danger_ahead_to_turn,"1+dg*D_post")
            to_ring(self.inputs,self.input_names.index("wall_ahead"),name,params.w_input)
        to_ring(self.danger,2,"MotorForward",-params.w_danger_ahead_forward_inh,"1+dg*D_post")
        to_ring(self.danger,3,"MotorForward",params.w_input*2,"1+dg*D_post")
        to_ring(self.inputs,self.input_names.index("wall_ahead"),"MotorForward",-params.w_danger_ahead_forward_inh)
        connect(self.inputs,self.modulators,[(self.input_names.index(n),0) for n in ("low_energy","food_left","food_right","food_ahead")],params.w_sensor_to_mod)
        connect(self.inputs,self.modulators,[(self.input_names.index("idle"),3)],params.w_idle_to_mod)
        connect(self.G,self.modulators,[(i,2) for n in motors[:3] for i in self._ring_idx(n)],params.w_motor_to_mod/params.ring_size)
        connect(self.G,self.modulators,[(i,1) for i in self._ring_idx("EscapeSwitch")],params.w_escape_to_mod/params.ring_size)
        connect(self.G,self.modulators,[(i,2) for i in self._ring_idx("MotorRest")],-params.w_rest_to_mod_inh/params.ring_size)
        connect(self.G,self.modulators,[(i,3) for n in motors[:3] for i in self._ring_idx(n)],-params.w_motor_to_mod/params.ring_size)
        # Independent tonic spiking breaks left/right symmetry; chemistry gates its gain.
        self.noise = PoissonGroup(self.n_state,rates=params.noise_rate_hz*Hz)
        connect(self.noise,self.G,[(i,i) for n in motors[:3] for i in self._ring_idx(n)],params.noise_weight_mv,
                "(F_post+Q_post)*(1-D_post)*(1-H_post)")
        self.spikes = SpikeMonitor(self.G,record=record)
        self.mod_spikes = SpikeMonitor(self.modulators,record=record)
        self.danger_spikes = SpikeMonitor(self.danger,record=record)
        self.state_mon = StateMonitor(self.G,"v",record=True) if record else None
        self.chem_mon = StateMonitor(self.modulators,"c",record=True,dt=params.snn_step_ms*ms) if record else None
        self.net = Network(self.inputs,self.modulators,self.G,self.danger,self.noise,
                           *self.connections,self.spikes,self.mod_spikes,self.danger_spikes,
                           *([self.state_mon,self.chem_mon] if record else []))
        self.recent_spikes = np.zeros(self.n_state)
        self.last_inputs = dict.fromkeys(self.input_names,0.0)
        self.last_mod_spikes = np.zeros(4,dtype=int)

    def _ring_idx(self, name):
        r = self.rings[name]
        return np.arange(r.start,r.end)

    def update_idle_level(self, obs, env):
        energy,fl,fa,fr,dl,da,dr = obs["rstate"]
        idle = (max(fl,fa,fr)<self.params.idle_food_threshold
                and max(dl,da,dr,env.danger_contact_signal)<self.params.idle_danger_threshold
                and energy>self.params.idle_energy_threshold)
        env.idle_level = float(np.clip(env.idle_level + (self.params.idle_rise_per_step if idle else -self.params.idle_decay_per_step),0,1))

    def encode_observation(self, obs, env):
        energy,fl,fa,fr,dl,da,dr = obs["rstate"]
        low = max(0,(self.params.energy_low_threshold-float(energy))/self.params.energy_low_threshold)
        return np.clip([low,fl,fr,fa,dl*self.params.danger_input_gain,dr*self.params.danger_input_gain,
                        da*self.params.danger_input_gain,env._wall_ahead(),env.idle_level,env.danger_contact_signal],0,1)*self.params.input_rate_hz

    def step(self, obs, env):
        self.update_idle_level(obs,env)
        rates = self.encode_observation(obs,env)
        self.last_inputs = dict(zip(self.input_names,rates/self.params.input_rate_hz))
        self.inputs.rates = rates*Hz
        before = np.asarray(self.spikes.count[:]).copy()
        mod_before = np.asarray(self.mod_spikes.count[:]).copy()
        danger_before = np.asarray(self.danger_spikes.count[:]).copy()
        self.net.run(self.snn_step_ms*ms, namespace={})
        self.recent_spikes = np.asarray(self.spikes.count[:])-before
        self.last_mod_spikes = np.asarray(self.mod_spikes.count[:])-mod_before
        acts = self.ring_activations()
        acts.update(zip(MOD_NAMES,map(float,self.modulators.c[:])))
        acts["EnergyLow"] = self.last_inputs["low_energy"]
        self.last_danger_spikes = np.asarray(self.danger_spikes.count[:])-danger_before
        return self.decode_action(obs,env,acts), acts

    def ring_activations(self):
        return {name:float(self.recent_spikes[r.start:r.end].sum()/r.n) for name,r in self.rings.items()}

    def decode_action(self, obs, env, acts):
        scores = np.array([acts[n] for n in ("MotorLeft","MotorRight","MotorForward","MotorRest")])
        if env._wall_ahead()>0.5:
            scores[2] = -1  # physical feasibility only; no fabricated motor activity
        if scores.max()<=0:
            return 6
        winners = np.flatnonzero(np.isclose(scores,scores.max()))
        return (0,1,2,6)[random.choice(winners.tolist())]


# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------

class RingsVisualizer:
    def __init__(self, controller):
        self.controller = controller
        self.fig = plt.figure(figsize=(21,8),layout="constrained")
        gs = self.fig.add_gridspec(2,4,width_ratios=[1,1,1.35,2.2])
        self.ax_env = self.fig.add_subplot(gs[0,0:2])
        self.ax_sensors = self.fig.add_subplot(gs[1,0:2])
        self.ax_rings = self.fig.add_subplot(gs[0,2])
        self.ax_mods = self.fig.add_subplot(gs[1,2])
        self.ax_timeline = self.fig.add_subplot(gs[:,3])
        self.timeline_history = []
        plt.ion()
        plt.show(block=False)

    def update(self, env, acts, step_idx, info):
        c = self.controller
        # Sensors that actually drove these spikes, not next-step observations.
        h = dict(c.last_inputs, **acts, step=step_idx)
        self.timeline_history = (self.timeline_history+[h])[-60:]
        self.ax_env.clear()
        self.ax_env.imshow(env.render())
        self.ax_env.axis("off")
        self.ax_env.set_title(f"RINGS {AGENT_VERSION} | step {step_idx} | energy {info.get('energy',0):.2f}\n"
                              f"food {info.get('food_collected',0)} | contacts {info.get('danger_contacts',0)} | action {info.get('last_action')}")
        ax = self.ax_sensors
        ax.clear()
        labels = c.input_names
        ax.bar(np.arange(len(labels)),[c.last_inputs[n] for n in labels],color=["tab:red" if "danger" in n else "tab:blue" for n in labels])
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right",fontsize=8)
        ax.set_ylim(0,1.1)
        ax.set_title("Input intensities (before action)")
        ax = self.ax_rings
        ax.clear(); ax.axis("off")
        ax.set_title("Fast rings: spikes / neuron / step")
        for k,name in enumerate(c.rings):
            x,y = (k%2)*1.6, -(k//2)*1.3
            ax.add_patch(plt.Circle((x,y),.52,color="tab:blue",alpha=.12+.8*min(1,acts[name]/max(1,np.ceil(c.snn_step_ms/3)))))
            ax.text(x,y,f"{name}\n{acts[name]:.2f}",ha="center",va="center",fontsize=8)
        ax.set_xlim(-.8,2.4); ax.set_ylim(-3.3,.8); ax.set_aspect("equal")
        ax = self.ax_mods
        ax.clear(); ax.axis("off")
        ax.set_title("Slow chemicals: concentration [0,1]")
        for k,(name,analogue) in enumerate(zip(MOD_NAMES,MOD_ANALOGUES)):
            y = 4-k
            ax.add_patch(plt.Rectangle((0,y-.28),.55,.55,color="tab:orange",alpha=.12+.8*acts[name]))
            ax.text(.27,y,f"{acts[name]:.2f}",ha="center",va="center",fontsize=8)
            ax.text(.7,y,f"{name} ({analogue})\nsecreting neuron: {c.last_mod_spikes[k]} spikes/step",va="center",fontsize=8)
        ax.add_patch(plt.Rectangle((0,-.28),.55,.55,color="tab:green",alpha=.12+.8*acts["EnergyLow"]))
        ax.text(.27,0,f"{acts['EnergyLow']:.2f}",ha="center",va="center",fontsize=8)
        ax.text(.7,0,"EnergyLow (homeostatic scalar)",va="center",fontsize=8)
        ax.set_xlim(-.1,4.6); ax.set_ylim(-.7,4.7)
        ax = self.ax_timeline
        ax.clear()
        lanes = ["food_left","food_ahead","food_right","danger_left","danger_ahead","danger_right",
                 "danger_contact","EnergyLow","idle",*c.rings,*MOD_NAMES]
        xs = [h["step"] for h in self.timeline_history]
        for k,name in enumerate(lanes):
            base = len(lanes)-1-k
            vals = np.array([h[name] for h in self.timeline_history])
            # Fixed physical scale: maximum spikes/neuron allowed by refractory; chemicals/inputs 0..1.
            scale = max(1,np.ceil(c.snn_step_ms/3)) if name in c.rings else 1
            color = "tab:orange" if name in MOD_NAMES else ("tab:blue" if name in c.rings else "tab:gray")
            ax.plot(xs,base+.7*np.clip(vals/scale,0,1),color=color,lw=1.5)
            ax.axhline(base,color="gray",alpha=.2,lw=.5)
        ax.set_yticks(range(len(lanes)),list(reversed(lanes)),fontsize=8)
        ax.set_ylim(-.3,len(lanes)); ax.set_xlim(max(0,step_idx-59),max(59,step_idx))
        ax.set_xlabel("Environment step")
        ax.set_title(f"Shared timeline: inputs → rings → chemicals\nFixed scales: rings 0–{int(np.ceil(c.snn_step_ms/3))}; other lanes 0–1")
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        plt.pause(.001)


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
    progress_callback=None,
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
        debug_fields = ["step", "energy_before", "energy_after", *controller.input_names,
                        *controller.rings, *MOD_NAMES, "EnergyLow",
                        *[n+"_spikes" for n in MOD_NAMES],
                        "action", "food_collected", "danger_contacts"]
        debug_writer = csv.DictWriter(debug_file, fieldnames=debug_fields)
        debug_writer.writeheader()

    total_reward = 0.0
    terminated = truncated = False
    step_idx = 0
    rest_steps = 0

    while not (terminated or truncated):
        energy_before = float(obs["rstate"][0])
        action, acts = controller.step(obs, env)
        if action == 6:
            rest_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if debug_writer is not None:
            debug_writer.writerow(dict(step=step_idx, energy_before=energy_before,
                energy_after=float(info.get("energy",0)), **controller.last_inputs, **acts,
                **{n+"_spikes":int(controller.last_mod_spikes[k]) for k,n in enumerate(MOD_NAMES)},
                action=action, food_collected=info.get("food_collected",0),
                danger_contacts=info.get("danger_contacts",0)))
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
        if progress_callback is not None:
            progress_callback(step_idx)

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
    debug_log_dir: Optional[str] = None,
) -> Dict[str, float]:
    results = []
    for idx, s in enumerate(seeds, start=1):
        def telemetry(step, force=False, done=False):
            now = time.monotonic()
            if _TUNE_QUEUE is not None and (force or now - telemetry.last >= 1.0):
                _TUNE_QUEUE.put((trial, idx, s, step, done))
                telemetry.last = now
        telemetry.last = 0.0
        telemetry(0, force=True)
        if status:
            prefix = f"trial {trial} seed {idx}/{len(seeds)}" if trial is not None else f"seed {idx}/{len(seeds)}"
            print(f"\n{prefix}")
        episode_debug_log = None
        if debug_log_dir is not None:
            Path(debug_log_dir).mkdir(parents=True, exist_ok=True)
            trial_label = f"trial{trial:03d}" if trial is not None else "trial_unknown"
            episode_debug_log = str(Path(debug_log_dir) / f"{trial_label}_seed{s}.csv")
        results.append(run_episode(seed=s, params=params, visualize=False, max_steps=max_steps, status=status, record=record, grid_size=grid_size, danger_speed=danger_speed, debug_log=episode_debug_log, progress_callback=telemetry if _TUNE_QUEUE is not None else None))
        telemetry(results[-1].steps, force=True, done=True)
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


_TUNE_QUEUE = None


def _init_tune_worker(progress_queue):
    global _TUNE_QUEUE
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _TUNE_QUEUE = progress_queue


def _tune_trial(job):
    """Top-level picklable worker; one candidate, all shared evaluation seeds."""
    trial, params_dict, settings = job
    params = RingsParams(**params_dict)
    metrics = evaluate_params(params, status=False, trial=trial, **settings)
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError(f"Non-finite metrics in trial {trial}")
    return {"trial": trial, "params": params_dict, "metrics": metrics}


def _save_tune_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def tune_random_search(
    trials: int = 50,
    seeds: Optional[List[int]] = None,
    max_steps: int = 250,
    search_seed: int = 123,
    output_path: str = "rings_best_params.json",
    record: bool = False,
    grid_size: int = 15,
    danger_speed: float = 0.0,
    debug_log_dir: Optional[str] = None,
    workers: int = 32,
    resume: bool = False,
    progress_interval: float = 5.0,
) -> Tuple[RingsParams, Dict[str, float]]:
    """Parallelize hybrid candidates, retaining survival scoring and checkpoint mechanics.

    Each candidate sees the same seeds. No nested process pools. Only the
    parent writes the checkpoint. Equal scores choose the lowest trial index.
    """
    import multiprocessing as mp
    import hashlib
    import traceback

    if not np.isfinite(progress_interval) or progress_interval <= 0:
        raise ValueError("progress_interval must be finite and positive")
    seeds = [1, 2, 3] if seeds is None else list(seeds)
    if trials < 1 or not 1 <= workers <= 100 or max_steps < 1:
        raise ValueError("trials/max_steps must be positive; workers must be 1..100")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a nonempty list of distinct scenario seeds")
    if any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Scenario seeds must fit uint32")
    # Set before spawning: each Brian/NumPy worker uses one compute thread.
    thread_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    settings = dict(seeds=seeds, max_steps=max_steps, record=record,
                    grid_size=grid_size, danger_speed=danger_speed,
                    debug_log_dir=debug_log_dir)
    config = dict(agent_version=AGENT_VERSION, trials=trials, search_seed=search_seed,
                  settings=settings, model_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    completed = {}
    failed = {}
    path = Path(output_path)
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; choose another --output or use --resume")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("tune_config") != config:
            raise ValueError("Resume settings or model hash do not match checkpoint")
        completed = {row["trial"]: row for row in previous.get("all_results", [])}
    elif resume:
        raise FileNotFoundError(f"Cannot resume missing checkpoint: {path}")

    # Sampling only in parent, in original trial order, independently of scheduling.
    rng = random.Random(search_seed)
    jobs = [(t, asdict(sample_params(rng)), settings) for t in range(1, trials + 1)]
    jobs = [job for job in jobs if job[0] not in completed]

    def checkpoint():
        best = max(completed.values(), key=lambda row: (row["metrics"]["score"], -row["trial"])) if completed else None
        payload = dict(tune_config=config, completed_trials=len(completed),
                       failed_trials=sorted(failed.values(), key=lambda row: row["trial"]),
                       all_results=sorted(completed.values(), key=lambda row: row["trial"]))
        if best is not None:
            payload.update(best_params=best["params"], best_metrics=best["metrics"], best_trial=best["trial"])
        _save_tune_payload(output_path, payload)
        return best

    checkpoint()
    print(f"Tune: {trials} candidates x {len(seeds)} scenarios = {trials * len(seeds)} episodes; "
          f"{len(jobs)} pending candidates; up to {min(workers, len(jobs))} processes", flush=True)
    old_env = {key: os.environ.get(key) for key in (*thread_keys, "MPLBACKEND")}
    for key in thread_keys:
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    started = time.monotonic()
    last_report = 0.0
    initial_episodes = len(completed) * len(seeds)
    episode_counts = {t: len(seeds) for t in completed}
    active = {}

    def show_progress(force=False):
        nonlocal last_report
        while True:
            try:
                trial, index, seed, step, done = progress_queue.get_nowait()
            except queue.Empty:
                break
            if trial in completed or trial in failed:
                continue
            episode_counts[trial] = max(episode_counts.get(trial, 0), index if done else index - 1)
            active[trial] = (index, seed, step)
        now = time.monotonic()
        if not force and now - last_report < progress_interval:
            return
        last_report = now
        total = trials * len(seeds)
        finished = sum(episode_counts.values())
        elapsed = now - started
        new_finished = finished - initial_episodes
        eta = f"{elapsed * (total - finished) / new_finished:.0f}s" if new_finished else "estimating"
        fraction = finished / total
        bar = "#" * int(24 * fraction) + "-" * (24 - int(24 * fraction))
        details = "; ".join(f"trial {t}: seed {seed} ({idx}/{len(seeds)}), step {step}/{max_steps}"
                            for t, (idx, seed, step) in sorted(active.items())[:4])
        print(f"[progress {bar}] {fraction:6.1%} episodes={finished}/{total} "
              f"candidates={len(completed)}/{trials} active={len(active)} failed={len(failed)} "
              f"elapsed={elapsed:.0f}s ETA~{eta}" + (f" | {details}" if details else (" | complete" if len(completed) == trials else " | no active telemetry yet")), flush=True)

    pool = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)

    def interrupt_tune(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_tune)
    signal.signal(signal.SIGINT, interrupt_tune)
    try:
        show_progress(force=True)
        if jobs:
            # Pool exposes terminate()/join() on Python 3.8 too.
            pool = ctx.Pool(processes=min(workers, len(jobs)),
                            initializer=_init_tune_worker, initargs=(progress_queue,))
            pending = {job[0]: pool.apply_async(_tune_trial, (job,)) for job in jobs}
            pool.close()  # no more tasks; queued tasks continue normally
            while pending:
                show_progress()
                ready = [t for t, result in pending.items() if result.ready()]
                for trial in ready:
                    result = pending.pop(trial)
                    active.pop(trial, None)
                    try:
                        row = result.get()
                        completed[trial] = row
                        episode_counts[trial] = len(seeds)
                    except Exception:
                        failed[trial] = dict(trial=trial, error=traceback.format_exc())
                        checkpoint()
                        print(f"[trial {trial}] FAILED; details in {output_path}", flush=True)
                        continue
                    best = checkpoint()
                    metrics = row["metrics"]
                    print(f"[{len(completed)}/{trials}] trial={trial} score={metrics['score']:.2f}, "
                          f"steps={metrics['mean_steps']:.1f}, food={metrics['mean_food']:.2f}, "
                          f"danger={metrics['mean_danger']:.2f}; best_trial={best['trial']}", flush=True)
                if pending:
                    time.sleep(0.1)
            pool.join()
            pool = None
        show_progress(force=True)
    except BaseException as exc:
        # Ignore repeated interrupts until children are reaped and checkpoint saved.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if isinstance(exc, KeyboardInterrupt):
            print("\nStopping tuning workers...", flush=True)
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        checkpoint()
        if isinstance(exc, KeyboardInterrupt):
            print(f"All tuning workers stopped. Saved {len(completed)} candidates to {output_path}. "
                  "Use --resume to continue.", flush=True)
            raise SystemExit(130)
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        progress_queue.cancel_join_thread()
        progress_queue.close()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    best = checkpoint()
    if failed:
        raise RuntimeError(f"{len(failed)} failed candidates; results saved. Fix the cause and use --resume.")
    if best is None:
        raise RuntimeError("No successful candidates")
    print(f"Saved {len(completed)} candidates to {output_path}; best trial={best['trial']}", flush=True)
    return RingsParams(**best["params"]), best["metrics"]


def load_params(path: str) -> RingsParams:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload.get("best_params", payload)
    unknown = set(data)-set(RingsParams.__dataclass_fields__)
    if unknown:
        raise ValueError("Incompatible parameter schema (v0.38 requires new tuning): " + ", ".join(sorted(unknown)))
    return RingsParams(**data)


# -----------------------------------------------------------------------------
# 6. CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "tune"], default="run")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--workers", type=int, default=32, help="Tune worker processes: 1..100")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of tuning scenarios")
    parser.add_argument("--start-seed", type=int, default=1, help="First scenario seed")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit scenario seeds; overrides num/start-seed")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Tune progress update interval in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume tuning from --output")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--params", type=str, default=None, help="Path to rings_best_params.json")
    parser.add_argument("--output", type=str, default="rings_best_params.json")
    parser.add_argument("--visual", choices=["on", "off"], default="off")
    parser.add_argument("--record", choices=["on", "off"], default="off")
    parser.add_argument("--debug-log", type=str, default=None, help="Write per-step diagnostic CSV log to this path")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--danger-speed", type=float, default=0.0)
    args = parser.parse_args()
    print(f"RINGS {AGENT_VERSION} | Python {sys.version.split()[0]} | {sys.executable}", flush=True)

    record = args.record == "on"

    if args.mode == "tune":
        tune_random_search(
            trials=args.trials,
            seeds=args.seeds if args.seeds is not None else list(range(args.start_seed, args.start_seed + args.num_seeds)),
            max_steps=args.max_steps,
            search_seed=args.seed,
            output_path=args.output,
            record=record,
            grid_size=args.grid_size,
            danger_speed=args.danger_speed,
            debug_log_dir=args.debug_log,
            workers=args.workers,
            resume=args.resume,
            progress_interval=args.progress_interval,
        )
    else:
        params = load_params(args.params) if args.params else RingsParams()
        visualize = args.visual == "on"
        result = run_episode(seed=args.seed, params=params, visualize=visualize, max_steps=args.max_steps, status=not visualize, record=record, debug_log=args.debug_log, grid_size=args.grid_size, danger_speed=args.danger_speed)
        print(result)


if __name__ == "__main__":
    main()
