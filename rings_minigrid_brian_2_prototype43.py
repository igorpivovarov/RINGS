"""
RINGS v0.43.0 -- minimal agent with recurrent excitatory rings and slow chemistry.
RESEARCH PROTOTYPE, not a species-specific physiological reconstruction.

ARCHITECTURE / ASSUMPTIONS
1. Ideal labelled food/danger/proximity receptors: geometry -> saturating current
   in 12 non-adapting sensory LIF neurons. Geometry is not object recognition.
   Energy deficit/satiety and actual actuator effort/turn displacement are ideal
   intero-/proprioceptive signals. ONLY sensory neurons receive these currents.
2. Five excitatory recurrent populations: EscapeSwitch and four motor rings.
   Delayed +1/+2 connections and intrinsic spike adaptation produce transient
   recurrent activity. Ring topology is a hypothesis, not proof of an attractor
   or a self-representation. All learned/innate-looking weights are preset/tuned.
3. Dedicated inhibitory interneurons mediate competition, danger, satiety and
   turn feedback. All neural synapses add nonnegative excitatory OR inhibitory
   conductance, with reversal potentials 0/-80 mV. No remote voltage resets.
4. Four secreting LIF neurons: dc/dt=-c/tau; spike c+=release*(1-c).
   c is a normalized, well-mixed field; c/(Kd+c) is receptor occupancy.
   No concentration updates in Python, stimulus-dependent clearance, or external
   behavioral gates. Biological labels indicate inspiration, not fitted chemistry.
5. Curiosity secretion is tonic and inhibited by sensory events/satiety/activity;
   there is no Python idle accumulator. Fatigue secretion follows effort afference
   and weak motor efference copy; it clears with the same tau during activity/rest.
6. Motor readout counts spikes per fixed control window, selects maximum activity
   (seeded random ties), and uses no-op when all motor populations are silent.
   It NEVER consults sensors/energy/world or vetoes an action. Physical walls are
   enforced only by MiniGrid. Bad neural decisions remain observable.
7. Body: discrete 90-degree turns and cell moves; normalized energy, instantaneous
   ingestion, basal cost .002 per step plus .006 per attempted motor action;
   food +.55 (capacity 1), danger -.25. No neural metabolic cost or detailed muscles.
   This is a declared task/plant model, not a calibrated metabolism.
8. dt=.1 ms, control window=30 ms by default. All intrinsic taus are in ms.
   Tune holds control window, sensor range and population size fixed. World speed
   is cells/control step; video FPS is playback only. Environment, Brian and motor
   tie-breaking have separate RNG streams. seed=7 is a development scenario.
9. Tune selects weights/taus offline; there is no lifetime learning. Its explicit
   v43 objective is mean survival steps. Food, harm, active rest, silence, coverage
   and wall bumps are reported separately. Survival alone is not evidence of Self.

CONNECTION AUDIT
The controller.connection_manifest contains every synaptic pathway (source,
target, neuron indices, sign, weight, delay and chemical gain). NPZ stores this,
all population names, parameters, spikes, states, concentrations and scene grids.
--audit-path writes the same architecture/assumption metadata as readable JSON.
Recordings and videos are compatible with this file's --replay command; external
neural renderers should read population names from metadata (12 receptors added).
v43 adds Defense occupancy -> EscapeSwitch depolarization, removes EscapeSwitch
-> Defense secretion (avoiding a self-sustaining chemical loop), and rebalances
Foraging/Curiosity/Rest drive. Escape recurrence is 0.70 of motor recurrence
by default, so it can end as Defense clears. Diffuse exploration drive is equal
across the three movement populations; sensory wiring still provides direction.
This is a circuit hypothesis, not a fitted
biological pathway. No spatial memory or rear-facing danger input is introduced.
Use fresh v43 tuning: older parameters would restore the old drive balance.
Ablation switches are deliberately deferred until this architecture is evaluated.

Examples:
 python3 rings_minigrid_brian_2_prototype43.py --visual on --danger-speed .5 --record on --record-path run43.npz --debug-log log43.csv
 python3 rings_minigrid_brian_2_prototype43.py --self-test
 python3 rings_minigrid_brian_2_prototype43.py --mode tune --workers 32 --output rings_v043_best.json
 python3 rings_minigrid_brian_2_prototype43.py --replay run43.npz --video-path run43.mp4
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
import uuid
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
    defaultclock,
    ms,
    Hz,
    mV,
    prefs,
    start_scope,
    linked_var,
    seed as brian_seed,
)

prefs.codegen.target = "numpy"

AGENT_VERSION = "0.43.0"


# -----------------------------------------------------------------------------
# 1. Tunable parameter set
# -----------------------------------------------------------------------------

MOD_NAMES = ("ForagingMod", "DefenseMod", "FatigueMod", "CuriosityMod")
MOD_ANALOGUES = ("octopamine-like", "tyramine-like", "adenosine-like", "dopamine-like")

@dataclass
class RingsParams:
    # Structural/task settings are fixed by sample_params, not optimized.
    ring_size: int = 12
    snn_step_ms: float = 30.0
    dt_ms: float = 0.1
    sensor_range: int = 3
    membrane_tau_ms: float = 15.0
    excitatory_tau_ms: float = 4.0
    inhibitory_tau_ms: float = 6.0
    ring_delay_ms: float = 5.0
    adaptation_tau_ms: float = 60.0
    adaptation_jump_mv: float = 3.0
    receptor_drive_mv: float = 50.0
    receptor_half: float = 0.12
    satiety_midpoint: float = 0.95
    mod_tau_ms: float = 40.0
    mod_refractory_ms: float = 15.0
    foraging_tau_ms: float = 450.0
    defense_tau_ms: float = 240.0
    fatigue_tau_ms: float = 600.0
    curiosity_tau_ms: float = 600.0
    foraging_release: float = 0.12
    defense_release: float = 0.14
    fatigue_release: float = 0.05
    curiosity_release: float = 0.14
    receptor_kd: float = 0.25
    curiosity_tonic_mv: float = 18.0
    foraging_drive_mv: float = 30.0
    curiosity_drive_mv: float = 32.0
    fatigue_inhibition_mv: float = 12.0
    fatigue_rest_drive_mv: float = 20.0
    defense_rest_inhibition_mv: float = 18.0
    defense_escape_drive_mv: float = 70.0
    chemical_defense_inhibition_mv: float = 20.0
    foraging_gain: float = 0.8
    defense_gain: float = 0.8
    turn_exploration_ratio: float = 1.0
    # Conductance increments in units of leak conductance, NEVER voltage jumps.
    g_ring: float = 2.0
    g_support: float = 0.4
    escape_recurrence_ratio: float = 0.70
    g_food: float = 1.3
    g_danger: float = 1.8
    g_escape_motor: float = 1.6
    g_lateral_escape: float = 0.8
    g_afferent_mod: float = 1.0
    g_effort_mod: float = 0.55
    g_motor_mod: float = 0.30
    g_interneuron_in: float = 1.8
    g_competition_in: float = 1.5
    g_competition_out: float = 0.35
    g_safety_inhibition: float = 3.0
    g_satiety_inhibition: float = 1.8
    g_completion_inhibition: float = 1.0
    g_event_inhibition: float = 0.5
    noise_rate_hz: float = 40.0
    noise_g: float = 0.12

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive")
        if int(self.ring_size)!=self.ring_size or self.ring_size<3:
            raise ValueError("ring_size must be integer >=3")
        if int(self.sensor_range)!=self.sensor_range or not 1<=self.sensor_range<=3:
            raise ValueError("sensor_range must be integer 1..3")
        if not 0<self.satiety_midpoint<1 or self.turn_exploration_ratio>1 or self.escape_recurrence_ratio>1:
            raise ValueError("invalid normalized receptor/drive setting")
        if self.ring_delay_ms<=3 or self.dt_ms>0.1:
            raise ValueError("ring delay must exceed refractory; dt must be <=.1 ms")
        if not np.isclose(self.snn_step_ms/self.dt_ms,round(self.snn_step_ms/self.dt_ms)):
            raise ValueError("control duration must be an integer multiple of dt")
        for n in MOD_NAMES:
            if getattr(self,n[:-3].lower()+"_release")>1:
                raise ValueError("release fractions must be <=1")
            if getattr(self,n[:-3].lower()+"_tau_ms")<=self.adaptation_tau_ms:
                raise ValueError("chemical taus must exceed adaptation tau")


def sample_params(rng):
    """Offline optimization of circuit parameters only; fixed task/sensory budget."""
    return RingsParams(
        g_ring=rng.uniform(1.8,2.5),g_support=rng.uniform(.3,.6),
        ring_delay_ms=rng.uniform(4,7),adaptation_tau_ms=rng.uniform(40,80),
        adaptation_jump_mv=rng.uniform(2.5,4.5),g_food=rng.uniform(.8,2),
        g_danger=rng.uniform(1.3,2.5),g_escape_motor=rng.uniform(.8,2),
        g_competition_out=rng.uniform(.2,.7),g_safety_inhibition=rng.uniform(2,5),
        g_afferent_mod=rng.uniform(.7,1.5),g_effort_mod=rng.uniform(.25,.7),
        g_satiety_inhibition=rng.uniform(1,3),g_completion_inhibition=rng.uniform(.5,1.5),
        foraging_drive_mv=rng.uniform(26,36),curiosity_drive_mv=rng.uniform(28,38),
        defense_escape_drive_mv=rng.uniform(55,85),
        fatigue_inhibition_mv=rng.uniform(8,16),fatigue_rest_drive_mv=rng.uniform(18,25),
        curiosity_tonic_mv=rng.uniform(16,21),
        foraging_tau_ms=rng.uniform(300,700),defense_tau_ms=rng.uniform(150,400),
        fatigue_tau_ms=rng.uniform(400,900),curiosity_tau_ms=rng.uniform(350,800),
        fatigue_release=rng.uniform(.02,.07),curiosity_release=rng.uniform(.08,.18))


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
    Sensors use sectors within the highlighted view and the local distance limit.
    Intensity decays with distance.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        size: int = 15,
        max_steps: int = 250,
        n_food: int = 3,
        n_danger: int = 2,
        sensor_range: int = 3,
        danger_speed: float = 0.0,
        render_mode: Optional[str] = "rgb_array",
    ):
        self.size = size
        self.n_food = n_food
        self.n_danger = n_danger
        # Fixed local perception budget; not adjusted to rescue controller behavior.
        self.sensor_range = int(sensor_range)
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
        self.last_effort = 0.0
        self.last_turn = 0.0
        self.danger_speed_accumulator = 0.0
        self.danger_contact_signal = 0.0
        self.last_effort = 0.0
        self.last_turn = 0.0

        mission_space = MissionSpace(mission_func=lambda: "maintain energy and avoid danger")
        agent_view_size = max(3, 2 * self.sensor_range - 1)
        if agent_view_size % 2 == 0:
            agent_view_size += 1

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
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
        self.motion_rng = random.Random(None if seed is None else int(seed)+1009)
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
        self.last_effort = 0.0
        self.last_turn = 0.0
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
        _, visibility = self.gen_obs_grid()

        for x in range(max(0, ax - self.sensor_range), min(self.width, ax + self.sensor_range + 1)):
            for y in range(max(0, ay - self.sensor_range), min(self.height, ay + self.sensor_range + 1)):
                if x == ax and y == ay:
                    continue
                obj = self.grid.get(x, y)
                if obj is None or obj.type != obj_type:
                    continue

                # Match the highlighted MiniGrid view; no hidden rear sensors.
                vx, vy = self.get_view_coords(x, y)
                if not (0 <= vx < self.agent_view_size and 0 <= vy < self.agent_view_size):
                    continue
                if not visibility[vx,vy]:
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
        self.motion_rng.shuffle(positions)
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
                    self.motion_rng.shuffle(step_options)
                elif abs(dx) == 1 and dy == 0:
                    step_options = [(self._sign(dx), 0)]
                elif dx == 0 and abs(dy) == 1:
                    step_options = [(0, self._sign(dy))]
                else:
                    step_options = []
            else:
                step_options = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                self.motion_rng.shuffle(step_options)

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
        previous_direction = int(self.agent_dir)
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
            # Plant assumption: basal .002/step, plus .006 when actuators work.
            self.energy = prev_energy - 0.002
            self.last_event = "rest"
        else:
            obs, reward, terminated, truncated, info = super().step(action)
            self.energy = prev_energy - self.energy_decay
        self.last_effort = float(action in (0,1,2))
        self.last_turn = float(int(self.agent_dir) != previous_direction)
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
            self.energy = 0.0
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
    """No hidden action rules: receptors -> E/I circuit -> ring spike readout."""

    def __init__(self, params, seed=1, record=False):
        start_scope()
        defaultclock.dt = params.dt_ms*ms
        brian_seed((int(seed)+2003) % 2**32)
        self.action_rng = random.Random(int(seed)+3001)
        self.params,self.record = params,record
        self.snn_step_ms,self.ring_size = params.snn_step_ms,params.ring_size
        self.rings = {n:RingSpec(n,params.ring_size,k*params.ring_size,(k+1)*params.ring_size)
                      for k,n in enumerate(("EscapeSwitch","MotorLeft","MotorRight","MotorForward","MotorRest"))}
        self.n_state = 5*params.ring_size
        self.input_names = ["low_energy","food_left","food_right","food_ahead",
            "danger_left","danger_right","danger_ahead","wall_ahead","danger_contact",
            "satiety","effort","turn_feedback"]
        # Twelve receptor cells. Analog stimuli reach ONLY these membranes.
        common = """
            dv/dt = (-65*mV-v + drive + ge*(0*mV-v)+gi*(-80*mV-v))/tau_v : volt (unless refractory)
            dge/dt = -ge/tau_e : 1
            dgi/dt = -gi/tau_i : 1
            tau_v : second (constant)
        """
        ns = dict(tau_e=params.excitatory_tau_ms*ms,tau_i=params.inhibitory_tau_ms*ms)
        self.receptors = NeuronGroup(len(self.input_names),common+"drive : volt",
            threshold="v>-50*mV",reset="v=-65*mV",refractory=3*ms,method="euler",namespace=ns)
        self.receptors.v=-65*mV;self.receptors.tau_v=10*ms
        self.inputs=self.receptors  # compatibility name, now LIF receptors, not Poisson inputs
        # All interneurons have exclusively inhibitory outputs.
        self.inh_names = ["CompetitionLeft","CompetitionRight","CompetitionForward","CompetitionRest",
            "DangerLeft","DangerRight","DangerAhead","WallAhead","Satiety",
            "TurnFeedback","Event","ActiveMotor"]
        self.I = NeuronGroup(len(self.inh_names),common+"drive : volt",
            threshold="v>-50*mV",reset="v=-65*mV",refractory=2*ms,method="euler",namespace=ns)
        self.I.v=-65*mV;self.I.tau_v=10*ms
        self.modulators = NeuronGroup(4,common+"""
            drive = tonic - defense_sensitive*defense_inh*cd/(kd+cd) : volt
            dc/dt = -c/tau_chem : 1
            cd : 1 (linked)
            tau_chem : second (constant)
            release : 1 (constant)
            tonic : volt (constant)
            defense_sensitive : 1 (constant)
        """,threshold="v>-50*mV",reset="v=-65*mV; c+=release*(1-c)",
            refractory=params.mod_refractory_ms*ms,method="exponential_euler",
            namespace=dict(ns,kd=params.receptor_kd,defense_inh=params.chemical_defense_inhibition_mv*mV))
        self.modulators.v=-65*mV;self.modulators.tau_v=params.mod_tau_ms*ms
        self.modulators.cd=linked_var(self.modulators,"c",index=np.ones(4,dtype=int))
        self.modulators.defense_sensitive=[1,0,0,1]
        self.modulators.tonic=[0,0,0,params.curiosity_tonic_mv]*mV
        for k,n in enumerate(MOD_NAMES):
            self.modulators.tau_chem[k]=getattr(params,n[:-3].lower()+"_tau_ms")*ms
            self.modulators.release[k]=getattr(params,n[:-3].lower()+"_release")
        self.G = NeuronGroup(self.n_state,common+"""
            da/dt = -a/tau_a : volt
            drive = explore*(forage_drive*F+curious_drive*Q)/(1+2*D)
                    - move*fatigue_inh*H + rest*(rest_drive*H-defense_rest*D)
                    + escape*defense_escape_drive*D : volt
            F=cf/(kd+cf) : 1 (constant over dt)
            D=cd/(kd+cd) : 1 (constant over dt)
            H=ch/(kd+ch) : 1 (constant over dt)
            Q=cq/(kd+cq) : 1 (constant over dt)
            cf : 1 (linked)
            cd : 1 (linked)
            ch : 1 (linked)
            cq : 1 (linked)
            escape : 1 (constant)
            explore : 1 (constant)
            move : 1 (constant)
            rest : 1 (constant)
        """,threshold="v>-50*mV+a",reset="v=-65*mV; a+=adapt_jump",refractory=3*ms,
            method="euler",namespace=dict(ns,tau_a=params.adaptation_tau_ms*ms,
                kd=params.receptor_kd,forage_drive=params.foraging_drive_mv*mV,
                curious_drive=params.curiosity_drive_mv*mV,fatigue_inh=params.fatigue_inhibition_mv*mV,
                defense_escape_drive=params.defense_escape_drive_mv*mV,
                rest_drive=params.fatigue_rest_drive_mv*mV,defense_rest=params.defense_rest_inhibition_mv*mV,
                adapt_jump=params.adaptation_jump_mv*mV))
        self.G.v=-65*mV;self.G.tau_v=params.membrane_tau_ms*ms
        for k,v in enumerate(("cf","cd","ch","cq")):
            setattr(self.G,v,linked_var(self.modulators,"c",index=np.full(self.n_state,k,dtype=int)))
        motors=("MotorLeft","MotorRight","MotorForward","MotorRest")
        for n in motors[:3]:
            self.G.move[self._ring_idx(n)]=1
            self.G.explore[self._ring_idx(n)]=1 if n=="MotorForward" else params.turn_exploration_ratio
        self.G.rest[self._ring_idx("MotorRest")]=1
        self.G.escape[self._ring_idx("EscapeSwitch")]=1
        self.connections=[];self.connection_manifest=[]
        groups={id(self.receptors):"receptors",id(self.G):"rings",id(self.I):"inhibitory",id(self.modulators):"modulators"}
        def conn(label,src,dst,pairs,g,inhibitory=False,effect="1",delay_ms=1):
            if inhibitory != (src is self.I):
                raise ValueError("Dale convention violated")
            ii,jj=zip(*pairs)
            syn=Synapses(src,dst,"w : 1 (constant)",on_pre=("gi" if inhibitory else "ge")+"_post += w*("+effect+")",
                         namespace={"fg":params.foraging_gain,"dg":params.defense_gain,
                                    "escape_recurrence_ratio":params.escape_recurrence_ratio})
            syn.connect(i=list(ii),j=list(jj));syn.w=g;syn.delay=delay_ms*ms
            self.connections.append(syn)
            self.connection_manifest.append(dict(label=label,source=groups[id(src)],target=groups[id(dst)],
                sign="inhibitory" if inhibitory else "excitatory",weight=g,delay_ms=delay_ms,
                gain=effect,i=list(map(int,ii)),j=list(map(int,jj))))
            return syn
        def rec(n):return self.input_names.index(n)
        def inh(n):return self.inh_names.index(n)
        def to_ring(label,src,i,n,g,**kw):
            return conn(label,src,self.G,[(i,j) for j in self._ring_idx(n)],g,**kw)
        # Recurrent excitatory +1/+2 loops, retained as individually identifiable pathways.
        self.recurrent_synapses=[]
        # Escape recurrence is weaker than motor recurrence: Defense can recruit
        # it, but a travelling spike must not maintain escape after chemistry clears.
        for offset,g in ((1,params.g_ring),(2,params.g_support)):
            pairs=[(i,r.start+(i-r.start+offset)%r.n) for r in self.rings.values() for i in range(r.start,r.end)]
            self.recurrent_synapses.append(conn("ring_offset_%d"%offset,self.G,self.G,pairs,g,delay_ms=params.ring_delay_ms,
                effect="1-escape_post*(1-escape_recurrence_ratio)"))
        # Motor population -> inhibitory cell -> other motor populations.
        for k,n in enumerate(motors):
            conn(n+"_competition_in",self.G,self.I,[(i,k) for i in self._ring_idx(n)],params.g_competition_in/params.ring_size)
            for other in motors:
                if other!=n:to_ring(n+"_inhibits_"+other,self.I,k,other,params.g_competition_out,inhibitory=True)
        for sensor,n in (("food_left","MotorLeft"),("food_right","MotorRight"),("food_ahead","MotorForward")):
            to_ring(sensor+"_motor",self.receptors,rec(sensor),n,params.g_food,effect="(1+fg*F_post)/(1+dg*D_post)")
            conn(sensor+"_foraging",self.receptors,self.modulators,[(rec(sensor),0)],params.g_afferent_mod)
        conn("hunger_foraging",self.receptors,self.modulators,[(rec("low_energy"),0)],params.g_afferent_mod)
        for sensor in ("danger_left","danger_right","danger_ahead","danger_contact"):
            to_ring(sensor+"_escape",self.receptors,rec(sensor),"EscapeSwitch",params.g_danger,effect="1+dg*D_post")
            conn(sensor+"_defense",self.receptors,self.modulators,[(rec(sensor),1)],params.g_afferent_mod)
        # Defense secretion is sensory-driven: no Escape -> Defense feedback loop.
        # Otherwise chemistry-driven escape could perpetually secrete its own drive.
        for n,weight in (("MotorForward",params.g_escape_motor),("MotorLeft",.25*params.g_escape_motor),("MotorRight",.25*params.g_escape_motor)):
            conn("escape_"+n,self.G,self.G,[(i,j) for i in self._ring_idx("EscapeSwitch") for j in self._ring_idx(n)],weight/params.ring_size)
        for sensor,ii,toward,away in (("danger_left","DangerLeft","MotorLeft","MotorRight"),("danger_right","DangerRight","MotorRight","MotorLeft")):
            conn(sensor+"_inh",self.receptors,self.I,[(rec(sensor),inh(ii))],params.g_interneuron_in)
            to_ring(sensor+"_avoid",self.I,inh(ii),toward,params.g_safety_inhibition,inhibitory=True)
            to_ring(sensor+"_turn",self.receptors,rec(sensor),away,params.g_danger)
            to_ring(sensor+"_forward",self.receptors,rec(sensor),"MotorForward",params.g_lateral_escape)
        for sensor,ii in (("danger_ahead","DangerAhead"),("wall_ahead","WallAhead")):
            conn(sensor+"_inh",self.receptors,self.I,[(rec(sensor),inh(ii))],params.g_interneuron_in)
            to_ring(sensor+"_brake",self.I,inh(ii),"MotorForward",params.g_safety_inhibition,inhibitory=True)
            for n in motors[:2]:to_ring(sensor+"_"+n,self.receptors,rec(sensor),n,params.g_danger)
        to_ring("contact_forward",self.receptors,rec("danger_contact"),"MotorForward",params.g_danger)
        conn("satiety_inh",self.receptors,self.I,[(rec("satiety"),inh("Satiety"))],params.g_interneuron_in)
        for n in motors[:3]:to_ring("satiety_"+n,self.I,inh("Satiety"),n,params.g_satiety_inhibition,inhibitory=True)
        conn("satiety_mods",self.I,self.modulators,[(inh("Satiety"),0),(inh("Satiety"),3)],params.g_satiety_inhibition,inhibitory=True)
        to_ring("satiety_rest",self.receptors,rec("satiety"),"MotorRest",.45*params.g_food)
        conn("effort_fatigue",self.receptors,self.modulators,[(rec("effort"),2)],params.g_effort_mod)
        conn("motor_effcopy",self.G,self.modulators,[(i,2) for n in motors[:3] for i in self._ring_idx(n)],params.g_motor_mod/params.ring_size)
        conn("turn_feedback_inh",self.receptors,self.I,[(rec("turn_feedback"),inh("TurnFeedback"))],params.g_interneuron_in)
        for n in motors[:2]:to_ring("turn_feedback_"+n,self.I,inh("TurnFeedback"),n,params.g_completion_inhibition,inhibitory=True)
        event_sensors=("food_left","food_right","food_ahead","danger_left","danger_right","danger_ahead","danger_contact")
        conn("events_inh",self.receptors,self.I,[(rec(n),inh("Event")) for n in event_sensors],params.g_interneuron_in)
        conn("events_curiosity",self.I,self.modulators,[(inh("Event"),3)],params.g_event_inhibition,inhibitory=True)
        conn("movement_inh",self.G,self.I,[(i,inh("ActiveMotor")) for n in motors[:3] for i in self._ring_idx(n)],params.g_competition_in/params.ring_size)
        conn("movement_curiosity",self.I,self.modulators,[(inh("ActiveMotor"),3)],params.g_event_inhibition,inhibitory=True)
        # Independent tonic background input; no stimulus/state conditional noise switch.
        self.noise=PoissonGroup(self.n_state,rates=params.noise_rate_hz*Hz)
        groups[id(self.noise)]="background"
        conn("background",self.noise,self.G,[(i,i) for i in range(self.n_state)],params.noise_g)
        self.spikes=SpikeMonitor(self.G,record=record)
        self.mod_spikes=SpikeMonitor(self.modulators,record=record)
        self.receptor_spikes=SpikeMonitor(self.receptors,record=record)
        self.inh_spikes=SpikeMonitor(self.I,record=record)
        self.state_mon=StateMonitor(self.G,["v","a","ge","gi"],record=True,dt=1*ms) if record else None
        self.chem_mon=StateMonitor(self.modulators,["c","v"],record=True,dt=1*ms) if record else None
        self.receptor_mon=StateMonitor(self.receptors,"v",record=True,dt=1*ms) if record else None
        self.inh_mon=StateMonitor(self.I,"v",record=True,dt=1*ms) if record else None
        self.net=Network(self.receptors,self.G,self.I,self.modulators,self.noise,*self.connections,
            self.spikes,self.mod_spikes,self.receptor_spikes,self.inh_spikes,
            *([self.state_mon,self.chem_mon,self.receptor_mon,self.inh_mon] if record else []))
        self.recent_spikes=np.zeros(self.n_state)
        self.last_inputs=dict.fromkeys(self.input_names,0.0)
        self.last_mod_spikes=np.zeros(4,dtype=int)
        self.last_receptor_spikes=np.zeros(len(self.input_names),dtype=int)
        self.last_inh_spikes=np.zeros(len(self.inh_names),dtype=int)
        self.last_decision="silent"

    def _ring_idx(self,name):
        r=self.rings[name];return np.arange(r.start,r.end)

    def encode_observation(self,obs,env):
        """Memoryless ideal transduction; does not select or prohibit any action."""
        energy,fl,fa,fr,dl,da,dr=obs["rstate"]
        hunger=1-float(energy)
        satiety=max(0,(float(energy)-self.params.satiety_midpoint)/(1-self.params.satiety_midpoint))
        return np.clip([hunger,fl,fr,fa,dl,dr,da,env._wall_ahead(),env.danger_contact_signal,
                        satiety,env.last_effort,env.last_turn],0,1)

    def step(self,obs,env):
        signals=self.encode_observation(obs,env)
        self.last_inputs=dict(zip(self.input_names,map(float,signals)))
        self.receptors.drive=self.params.receptor_drive_mv*signals/(self.params.receptor_half+signals)*mV
        monitors=(self.spikes,self.mod_spikes,self.receptor_spikes,self.inh_spikes)
        before=[np.array(m.count[:]) for m in monitors]
        self.net.run(self.snn_step_ms*ms,namespace={})
        counts=[np.array(m.count[:])-n for m,n in zip(monitors,before)]
        self.recent_spikes,self.last_mod_spikes,self.last_receptor_spikes,self.last_inh_spikes=counts
        acts=self.ring_activations();acts.update(zip(MOD_NAMES,map(float,self.modulators.c[:])))
        acts["EnergyLow"]=self.last_inputs["low_energy"]
        return self.decode_action(acts),acts

    def ring_activations(self):
        return {n:float(self.recent_spikes[r.start:r.end].sum()/r.n) for n,r in self.rings.items()}

    def decode_action(self,acts):
        """Pure readout. No sensors, energy, world access or safety veto."""
        names=("MotorLeft","MotorRight","MotorForward","MotorRest")
        scores=np.array([acts[n] for n in names])
        if scores.max()<=0:
            self.last_decision="silent";return 6
        winners=np.flatnonzero(scores==scores.max()).tolist()
        chosen=self.action_rng.choice(winners)
        self.last_decision="active_rest" if chosen==3 else "motor"
        return (0,1,2,6)[chosen]

    def audit(self):
        import hashlib
        from importlib.metadata import version
        runtime = {n:version(n) for n in ("brian2","minigrid","gymnasium","numpy")}
        return dict(agent_version=AGENT_VERSION,record_schema=42,python=sys.version,runtime=runtime,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            params=asdict(self.params),assumptions=__doc__,
            populations=dict(receptors=self.input_names,inhibitory=self.inh_names,
                rings={n:[r.start,r.end] for n,r in self.rings.items()},modulators=list(MOD_NAMES)),
            connections=self.connection_manifest,
            chemical_coupling=dict(occupancy="c/(Kd+c)",
                ring_drive=str(self.G.equations["drive"].expr),
                modulator_drive=str(self.modulators.equations["drive"].expr),
                ring_coefficients={n:np.asarray(getattr(self.G,n)[:]).tolist()
                                   for n in ("escape","explore","move","rest")},
                defense_secretion="danger receptor spikes only; no EscapeSwitch feedback"),
            conductance_units="relative to leak conductance",reversal_mv=dict(excitatory=0,inhibitory=-80),
            rng_offsets=dict(environment=1009,brian=2003,action=3001))


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

    def update(self, env, acts, step_idx, info, frame=None):
        c = self.controller
        # Sensors that actually drove these spikes, not next-step observations.
        h = dict(c.last_inputs, **acts, step=step_idx)
        self.timeline_history = (self.timeline_history+[h])[-60:]
        self.ax_env.clear()
        self.ax_env.imshow(env.render() if frame is None else frame)
        self.ax_env.axis("off")
        self.ax_env.set_title(f"RINGS {AGENT_VERSION} | step {step_idx} | energy {info.get('energy',0):.2f}\n"
                              f"food {info.get('food_collected',0)} | contacts {info.get('danger_contacts',0)} | chosen action {info.get('last_action')}\nScene and sensors BEFORE chosen action")
        ax = self.ax_sensors
        ax.clear()
        labels = c.input_names
        ax.bar(np.arange(len(labels)),[c.last_inputs[n] for n in labels],color=["tab:red" if "danger" in n else "tab:blue" for n in labels])
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right",fontsize=8)
        ax.set_ylim(0,1.1)
        ax.set_title("Ideal receptor stimuli (before action); numbers = receptor spikes")
        for k,n in enumerate(labels):
            ax.text(k,1.03,str(c.last_receptor_spikes[k]),ha="center",fontsize=7)
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
                 "danger_contact","EnergyLow","satiety","effort","turn_feedback",*c.rings,*MOD_NAMES]
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

def save_recording(path, controller, trajectory, metadata, scene_grids=None):
    """Portable NPZ, no pickle. Arrays use ms/mV; metadata is a JSON string.

    step_columns describes trajectory; positions/sensors are before action.
    Spike neuron IDs index ring_bounds / mod_names / receptor_names / inhibitory_names.
    Files contain neural traces and compact grid snapshots for video replay.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = controller
    columns = ["step", "x", "y", "direction", "energy_before", "energy_after", "action",
               "food_collected", "danger_contacts", *c.input_names, *c.rings, *MOD_NAMES]
    data = dict(metadata_json=np.asarray(json.dumps(metadata)),
        step_columns=np.asarray(columns), steps=np.asarray(trajectory,dtype=float).reshape(-1,len(columns)),
        ring_names=np.asarray(list(c.rings)),
        ring_bounds=np.asarray([[r.start,r.end] for r in c.rings.values()]),
        mod_names=np.asarray(MOD_NAMES),
        receptor_names=np.asarray(c.input_names),
        inhibitory_names=np.asarray(c.inh_names),
        audit_json=np.asarray(json.dumps(c.audit())),
        state_ge=np.asarray(c.state_mon.ge,dtype=np.float32),
        state_gi=np.asarray(c.state_mon.gi,dtype=np.float32),
        receptor_v_mv=np.asarray(c.receptor_mon.v/mV,dtype=np.float32),
        inhibitory_v_mv=np.asarray(c.inh_mon.v/mV,dtype=np.float32),
        state_t_ms=np.asarray(c.state_mon.t/ms),
        state_v_mv=np.asarray(c.state_mon.v/mV,dtype=np.float32),
        state_adaptation_mv=np.asarray(c.state_mon.a/mV,dtype=np.float32),
        chemical_t_ms=np.asarray(c.chem_mon.t/ms),
        chemical_c=np.asarray(c.chem_mon.c,dtype=np.float32),
        mod_v_mv=np.asarray(c.chem_mon.v/mV,dtype=np.float32))
    for name, mon in (("ring",c.spikes),("mod",c.mod_spikes),("receptor",c.receptor_spikes),("inhibitory",c.inh_spikes)):
        data[name+"_spike_t_ms"] = np.asarray(mon.t/ms)
        data[name+"_spike_i"] = np.asarray(mon.i,dtype=np.int32)
    if scene_grids is not None:
        data["scene_grids"] = np.asarray(scene_grids,dtype=np.uint8)
    tmp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f,**data)
        os.replace(str(tmp),str(path))
    finally:
        if tmp.exists(): tmp.unlink()
    print("\nRecording saved: " + str(path.resolve()),flush=True)


class VideoWriter:
    """Stream frames to MP4; wall-clock calculation time never enters the video."""
    def __init__(self,path,fps):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video FPS must be finite and positive")
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("video path must end in .mp4")
        if self.path.exists():
            raise FileExistsError("Video already exists: " + str(self.path))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("Video requires: pip install imageio imageio-ffmpeg pillow") from exc
        self.writer = imageio.get_writer(str(self.path),fps=fps,codec="libx264",macro_block_size=2)
        self.count = 0

    def append(self,frame,step,energy,action,acts):
        from PIL import Image,ImageDraw
        w = max(640,frame.shape[1])
        canvas = Image.new("RGB",(w,frame.shape[0]+104),"white")
        canvas.paste(Image.fromarray(frame),((w-frame.shape[1])//2,0))
        d = ImageDraw.Draw(canvas)
        lines = [f"Step {int(step)} | energy {energy:.3f} | action {int(action)} (scene BEFORE action)",
                 " | ".join(f"{n}: {acts.get(n,0):.3f}" for n in MOD_NAMES),
                 " | ".join(f"{n}: {acts.get(n,0):.2f}" for n in ("MotorLeft","MotorRight","MotorForward","MotorRest"))]
        for i,line in enumerate(lines):
            d.text((8,frame.shape[0]+8+26*i),line,fill="black")
        self.writer.append_data(np.asarray(canvas))
        self.count += 1

    def close(self):
        self.writer.close()
        print(f"Video saved: {self.path.resolve()} ({self.count} frames)",flush=True)


def replay_recording(path,output,fps=10,model_path=None):
    with np.load(path,allow_pickle=False) as z:
        if "scene_grids" not in z:
            raise ValueError("Replay requires v40+ scene_grids snapshots")
        meta = json.loads(str(z["metadata_json"]))
        rows,grids = z["steps"],z["scene_grids"]
        if len(rows)!=len(grids) or not len(rows):
            raise ValueError("Missing or inconsistent scene snapshots")
        columns = list(z["step_columns"])
        env = RingsHomeostasisEnv(size=meta["grid_size"],sensor_range=meta["params"]["sensor_range"],render_mode="rgb_array")
        env.reset(seed=meta["seed"])
        writer = VideoWriter(output,fps)
        try:
            for row,grid in zip(rows,grids):
                r = dict(zip(columns,row))
                env.grid,_ = Grid.decode(grid)
                env.agent_pos = (int(r["x"]),int(r["y"]))
                env.agent_dir = int(r["direction"])
                writer.append(env.render(),r["step"],r["energy_before"],r["action"],r)
        finally:
            writer.close()
            env.close()


@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    final_energy: float
    food_collected: int
    danger_contacts: int
    rest_steps: int
    active_rest_steps: int = 0
    silent_steps: int = 0
    wall_bumps: int = 0
    unique_cells: int = 0
    forward_steps: int = 0
    turn_steps: int = 0
    moved_steps: int = 0


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
    audit_path: Optional[str] = None,
    record_path: Optional[str] = None,
    video: bool = False,
    video_path: Optional[str] = None,
    video_fps: float = 10.0,
) -> EpisodeResult:
    if record_path and not record:
        raise ValueError("record_path requires record=True / --record on")
    if record:
        record_path = record_path or str(Path("recordings") / ("rings_v043_seed%d_%s.npz" % (seed,uuid.uuid4().hex)))
        if Path(record_path).suffix.lower() != ".npz":
            raise ValueError("record_path must end in .npz")
        Path(record_path).parent.mkdir(parents=True,exist_ok=True)
        if Path(record_path).exists():
            raise FileExistsError("Recording already exists: " + str(record_path))
    params = params or RingsParams()
    env = RingsHomeostasisEnv(size=grid_size, max_steps=max_steps, sensor_range=params.sensor_range, danger_speed=danger_speed, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    video_writer = VideoWriter(video_path or "rings_v043.mp4", video_fps) if video else None
    controller = RingsSNNController(params=params, seed=seed, record=record)
    if audit_path:
        Path(audit_path).parent.mkdir(parents=True,exist_ok=True)
        Path(audit_path).write_text(json.dumps(controller.audit(),indent=2),encoding="utf-8")
    vis = RingsVisualizer(controller) if visualize else None

    debug_file = None
    debug_writer = None
    if debug_log:
        debug_path = Path(debug_log)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path.open("w", newline="", encoding="utf-8")
        debug_fields = ["step", "x_before", "y_before", "dir_before", "x_after", "y_after", "dir_after", "wall_after", "energy_before", "energy_after", *controller.input_names,
                        *controller.rings, *MOD_NAMES, "EnergyLow",
                        *[n+"_spikes" for n in MOD_NAMES],
                        "decision_kind", "wall_bump", *[n+"_receptor_spikes" for n in controller.input_names],
                        "action", "food_collected", "danger_contacts"]
        debug_writer = csv.DictWriter(debug_file, fieldnames=debug_fields)
        debug_writer.writeheader()

    total_reward = 0.0
    terminated = truncated = False
    step_idx = 0
    rest_steps = 0
    active_rest_steps = silent_steps = wall_bumps = 0
    forward_steps = turn_steps = moved_steps = 0
    visited = {tuple(map(int,env.agent_pos))}
    trajectory = []
    scene_grids = []

    interrupted = False
    try:
        while not (terminated or truncated):
            energy_before = float(obs["rstate"][0])
            pos_before = tuple(map(int,env.agent_pos))
            dir_before = int(env.agent_dir)
            frame_before = env.render().copy() if vis is not None or video_writer is not None else None
            grid_before = env.grid.encode().copy() if record else None
            info_before = dict(energy=energy_before,food_collected=env.food_collected,
                               danger_contacts=env.danger_contacts)
            action, acts = controller.step(obs, env)
            if action == 6:
                rest_steps += 1
            forward_steps += int(action == 2)
            turn_steps += int(action in (0,1))
            active_rest_steps += int(controller.last_decision == "active_rest")
            silent_steps += int(controller.last_decision == "silent")
            wall_bump = int(action == 2 and controller.last_inputs["wall_ahead"] > 0)
            wall_bumps += wall_bump
            obs, reward, terminated, truncated, info = env.step(action)
            moved_steps += int(tuple(map(int,env.agent_pos)) != pos_before)
            total_reward += reward
            visited.add(tuple(map(int,env.agent_pos)))
            if video_writer is not None:
                video_writer.append(frame_before, step_idx, energy_before, action, acts)
            if record:
                scene_grids.append(grid_before)
                trajectory.append([step_idx,*pos_before,dir_before,energy_before,float(info["energy"]),action,
                    info["food_collected"],info["danger_contacts"],
                    *[controller.last_inputs[n] for n in controller.input_names],
                    *[acts[n] for n in controller.rings],*[acts[n] for n in MOD_NAMES]])

            if debug_writer is not None:
                debug_writer.writerow(dict(step=step_idx, x_before=pos_before[0], y_before=pos_before[1], dir_before=dir_before,
                    x_after=int(env.agent_pos[0]), y_after=int(env.agent_pos[1]), dir_after=int(env.agent_dir), wall_after=env._wall_ahead(), energy_before=energy_before,
                    energy_after=float(info.get("energy",0)), **controller.last_inputs, **acts,
                    **{n+"_spikes":int(controller.last_mod_spikes[k]) for k,n in enumerate(MOD_NAMES)},
                    decision_kind=controller.last_decision,wall_bump=wall_bump,
                    **{n+"_receptor_spikes":int(controller.last_receptor_spikes[k]) for k,n in enumerate(controller.input_names)},
                    action=action, food_collected=info.get("food_collected",0),
                    danger_contacts=info.get("danger_contacts",0)))
                debug_file.flush()

            if vis is not None:
                vis.update(env, acts, step_idx, dict(info_before,last_action=action), frame=frame_before)
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

    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if debug_file is not None:
            debug_file.close()
        env.close()
        if video_writer is not None:
            video_writer.close()
        if record:
            save_recording(record_path,controller,trajectory,dict(
                agent_version=AGENT_VERSION,seed=seed,params=asdict(params),
                grid_size=grid_size,danger_speed=danger_speed,max_steps=max_steps,
                completed_steps=len(trajectory),interrupted=interrupted,
                completed=bool(terminated or truncated),trace_dt_ms=1.0),scene_grids=scene_grids)

    result = EpisodeResult(
        steps=step_idx,
        total_reward=float(total_reward),
        final_energy=float(info.get("energy", 0.0)),
        food_collected=int(info.get("food_collected", 0)),
        danger_contacts=int(info.get("danger_contacts", 0)),
        rest_steps=int(rest_steps),active_rest_steps=active_rest_steps,silent_steps=silent_steps,
        wall_bumps=wall_bumps,unique_cells=len(visited),
        forward_steps=forward_steps,turn_steps=turn_steps,moved_steps=moved_steps,
    )

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

    # Explicit research objective, no reward for a hand-picked behavioral style.
    # Always report resources/harm/coverage too: passive survival may be long.
    score = float(np.mean(steps))
    return {
        "score": score,
        "mean_active_rest_steps": float(np.mean([r.active_rest_steps for r in results])),
        "mean_silent_steps": float(np.mean([r.silent_steps for r in results])),
        "mean_wall_bumps": float(np.mean([r.wall_bumps for r in results])),
        "mean_unique_cells": float(np.mean([r.unique_cells for r in results])),
        "mean_forward_steps": float(np.mean([r.forward_steps for r in results])),
        "mean_turn_steps": float(np.mean([r.turn_steps for r in results])),
        "mean_moved_steps": float(np.mean([r.moved_steps for r in results])),
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
    config = dict(objective="mean_survival_steps", agent_version=AGENT_VERSION, trials=trials, search_seed=search_seed,
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


def run_circuit_checks():
    """Reproducible default-parameter regression fixtures, NOT behavioral rules.

    Synthetic stimuli enter only the receptor currents used by the real encoder.
    Feedback represents the previous selected actuator command in an open arena.
    These tests verify specific failure modes, not physiology or general fitness.
    """
    report = {}
    for scenario, duration in (("threat_withdrawal",60),("hunger_without_food",60),
                               ("curiosity_without_stimulus",60),("satiety",60)):
        controller = RingsSNNController(RingsParams(), seed=19)
        signals = np.zeros(len(controller.input_names))
        # The normal world encoder is replaced ONLY inside this diagnostic fixture.
        controller.encode_observation = lambda obs,env: signals
        rows = []
        for step in range(duration):
            signals[:] = 0
            if scenario == "threat_withdrawal" and step < 2:
                signals[controller.input_names.index("danger_ahead")] = .092
                signals[controller.input_names.index("danger_right")] = .062
            elif scenario == "hunger_without_food":
                signals[controller.input_names.index("low_energy")] = .8
            elif scenario == "satiety":
                signals[controller.input_names.index("satiety")] = 1
            if rows:
                action = rows[-1]["action"]
                signals[controller.input_names.index("effort")] = int(action in (0,1,2))
                signals[controller.input_names.index("turn_feedback")] = int(action in (0,1))
            action, acts = controller.step(None,None)
            rows.append(dict(action=action,decision=controller.last_decision,**acts))
        if scenario == "threat_withdrawal":
            checks = dict(
                turns_to_threat=any(r["action"] in (0,1) for r in rows[:2]),
                forward_after_withdrawal=sum(r["action"]==2 for r in rows[2:12])>=5,
                escape_ends=all(r["EscapeSwitch"]==0 for r in rows[-10:]),
                defense_clears=rows[-1]["DefenseMod"]<.001,
                no_chemical_self_excitation=all(rows[i+1]["DefenseMod"]<=rows[i]["DefenseMod"]+1e-12
                                               for i in range(4,duration-1)))
        elif scenario == "satiety":
            checks = dict(active_rest=sum(r["decision"]=="active_rest" for r in rows)>=50)
        else:
            checks = dict(exploration_starts=any(r["action"]==2 for r in rows[:20]),
                          movement_persists=sum(r["action"]==2 for r in rows[-20:])>=10)
        report[scenario] = dict(checks=checks,forward_steps=sum(r["action"]==2 for r in rows),
            turn_steps=sum(r["action"] in (0,1) for r in rows),
            active_rest_steps=sum(r["decision"]=="active_rest" for r in rows),
            silent_steps=sum(r["decision"]=="silent" for r in rows),
            final_defense=rows[-1]["DefenseMod"],final_escape=rows[-1]["EscapeSwitch"])
        print(scenario + ": " + json.dumps(report[scenario],sort_keys=True),flush=True)
    if not all(all(r["checks"].values()) for r in report.values()):
        raise AssertionError("Circuit regression failed; inspect the printed checks.")
    print("All default-parameter circuit checks passed.",flush=True)
    return report


def load_params(path: str) -> RingsParams:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload.get("best_params", payload)
    if "defense_escape_drive_mv" not in data:
        raise ValueError("v0.43 requires fresh parameters/tuning (missing defense_escape_drive_mv). "
                         "Omit --params to use v0.43 defaults; use a new --output for tuning.")
    unknown = set(data)-set(RingsParams.__dataclass_fields__)
    if unknown:
        raise ValueError("Incompatible parameter schema (v0.43 requires conductance parameters): " + ", ".join(sorted(unknown)))
    return RingsParams(**data)


# -----------------------------------------------------------------------------
# 6. CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run synthetic default-circuit regression checks")
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
    parser.add_argument("--output", type=str, default="rings_v043_best.json")
    parser.add_argument("--visual", choices=["on", "off"], default="off")
    parser.add_argument("--record", choices=["on", "off"], default="off")
    parser.add_argument("--video", choices=["on","off"], default="off")
    parser.add_argument("--video-path", default="rings_v043.mp4")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--replay-model", help="Original v39 source, required only for legacy NPZ replay")
    parser.add_argument("--replay", help="Render v40 NPZ snapshots without running the SNN")
    parser.add_argument("--audit-path", default=None, help="Run mode: export full circuit/assumptions JSON")
    parser.add_argument("--record-path", default=None, help="Run-mode .npz output; default recordings/unique name")
    parser.add_argument("--debug-log", type=str, default=None, help="Write per-step diagnostic CSV log to this path")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--danger-speed", type=float, default=0.0)
    args = parser.parse_args()
    print(f"RINGS {AGENT_VERSION} | Python {sys.version.split()[0]} | {sys.executable}", flush=True)

    if args.self_test:
        if args.params or args.mode != "run":
            parser.error("--self-test tests defaults only; omit --params and --mode tune")
        run_circuit_checks()
        return

    if args.replay:
        replay_recording(args.replay,args.video_path,args.video_fps,args.replay_model)
        return
    if args.video == "on" and args.mode != "run":
        parser.error("--video on is supported in run mode only")
    record = args.record == "on"
    if args.audit_path and args.mode != "run":
        parser.error("--audit-path is for run mode; each recorded tune episode includes audit JSON")
    if args.record_path and (not record or args.mode != "run"):
        parser.error("--record-path requires --mode run --record on; tune uses unique per-episode filenames")

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
        result = run_episode(seed=args.seed, params=params, visualize=visualize, max_steps=args.max_steps, status=not visualize, record=record, debug_log=args.debug_log, grid_size=args.grid_size, danger_speed=args.danger_speed, record_path=args.record_path, video=args.video == "on", video_path=args.video_path, video_fps=args.video_fps, audit_path=args.audit_path)
        print(result)


if __name__ == "__main__":
    main()
