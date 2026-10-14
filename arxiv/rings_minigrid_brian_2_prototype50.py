"""
RINGS v0.50.0 -- persistent dynamical self-state research prototype.
Python 3.8+; dependencies: numpy, matplotlib, brian2, minigrid, gymnasium.
Optional video: imageio, imageio-ffmpeg, pillow.

ARCHITECTURE / EXPLICIT IDEALIZATIONS
1. Body/world are unchanged from v0.43: discrete 90-degree turns and grid moves;
   energy capacity 1, basal cost .002, additional motor cost .006, food +.55,
   danger -.25. Sensors are ideal labelled, local, egocentric and visibility-limited.
2. Body/sensory signals -> 12 LIF receptors -> distributed recurrent state.
   NO sensor/body/chemical -> motor connection. Four mixed-selectivity rings,
   24 excitatory neurons each, exchange delayed excitation and inhibitory feedback.
   Ring A/B/C/D are anatomical partitions, NOT hunger/fear/rest/action switches.
3. Persistent activity is supported by heterogeneous tonic depolarization,
   recurrence, inhibition and adaptation; it is NEVER clamped or restarted by
   Python. Spikes need not occur at every .1 ms tick. Continuity is measured by
   global inter-spike gaps and silent control windows, not assumed from topology.
4. Each ring contains overlapping innate sensor/body/action-related selectivity.
   Directional afferents and fixed readout weights are hand-designed priors,
   not learned representations. A mixed state is a candidate regulatory self-state,
   not established self-representation, consciousness or a proven attractor.
5. Four secretory LIF neurons release normalized well-mixed fields on spikes;
   dc/dt=-c/tau, release*(1-c). Receptor occupancy c/(Kd+c) shifts STATE membranes
   through heterogeneous signed currents. Motor membranes have no chemical terms.
   Chemical names are functional labels, not calibrated molecular identities.
6. A separate 4 x 8 LIF motor layer reads state spikes. Common-mode inhibition and
   competition operate within readout. The decoder counts motor spikes, chooses
   maximum (seeded random ties), and no-ops on silence. It has no sensor access,
   action veto, sequence, behavioral timer, or state reset. Motor silence is NOT
   self-state silence. Actual actuator effort/turn returns via body receptors.
7. Structural connectivity/heterogeneity use one fixed topology seed; environment,
   spiking noise and action ties use separate streams. No within-life learning.
8. --ablation no-state-recurrence zeros within/cross-ring E->E and E->state-I
   feedback while keeping cells, inputs, tonic drive, readout and RNG draws.
   --ablation no-chemistry zeros only chemical membrane coupling; secretors/fields
   remain recorded. 'both' combines these. These are matched mechanism removals,
   NOT separately optimized reflex baselines. Each condition needs multi-seed study.
9. --self-test checks continuity without sensory input, persistence without noise,
   input/history dependence and causal motor mediation. Tests do not prove stable
   attractors: perturbation recovery/basin convergence require additional experiments.
10. Tune selects only candidates with no silent STATE control windows in the
    evaluation episodes (explicit offline validity constraint, no runtime rescue).
    Among valid candidates it optimizes mean survival with food/harm/movement/coverage
    and STATE-vs-MOTOR silence diagnostics; survival alone is not the paper's claim.
    Failed candidate dynamics remain observable, not rescued by a Python rule.

NPZ schema 50 separates ring/state spikes from motor spikes, stores all membrane
traces, chemistry, full connection/weight audit and pre-action world snapshots.
Old external v42 neural-video scripts need schema-50 adaptation; use built-in
--video or --replay. Previous parameter/checkpoint schemas are incompatible.

Examples:
 python3 rings_minigrid_brian_2_prototype50.py --visual on --danger-speed .5 --max-steps 1500 --record on --record-path run50.npz --debug-log log50.csv
 python3 rings_minigrid_brian_2_prototype50.py --self-test
 python3 rings_minigrid_brian_2_prototype50.py --mode tune --workers 32 --output rings_v050_best.json
 python3 rings_minigrid_brian_2_prototype50.py --ablation no-state-recurrence --seed 7
 python3 rings_minigrid_brian_2_prototype50.py --replay run50.npz --video-path run50.mp4
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

AGENT_VERSION = "0.50.0"


# -----------------------------------------------------------------------------
# 1. Tunable parameter set
# -----------------------------------------------------------------------------

MOD_NAMES = ("ForagingMod", "DefenseMod", "FatigueMod", "CuriosityMod")

@dataclass
class RingsParams:
    # Fixed computational/sensory budgets; these do not vary during tuning.
    ring_size: int = 24
    state_rings: int = 4
    motor_size: int = 8
    topology_seed: int = 1701
    snn_step_ms: float = 30.0
    dt_ms: float = 0.1
    sensor_range: int = 3
    membrane_tau_ms: float = 15.0
    excitatory_tau_ms: float = 4.0
    inhibitory_tau_ms: float = 6.0
    ring_delay_ms: float = 5.0
    adaptation_tau_ms: float = 80.0
    adaptation_jump_mv: float = 2.0
    trace_tau_ms: float = 60.0
    state_tonic_mv: float = 17.0
    tonic_spread_mv: float = 1.5
    receptor_drive_mv: float = 50.0
    receptor_half: float = 0.12
    satiety_midpoint: float = 0.95
    mod_tau_ms: float = 40.0
    mod_refractory_ms: float = 15.0
    foraging_tau_ms: float = 450.0
    defense_tau_ms: float = 300.0
    fatigue_tau_ms: float = 600.0
    curiosity_tau_ms: float = 600.0
    foraging_release: float = 0.12
    defense_release: float = 0.14
    fatigue_release: float = 0.05
    curiosity_release: float = 0.14
    receptor_kd: float = 0.25
    curiosity_tonic_mv: float = 18.0
    foraging_drive_mv: float = 8.0
    defense_drive_mv: float = 14.0
    fatigue_drive_mv: float = 12.0
    curiosity_drive_mv: float = 6.0
    g_ring: float = 1.1
    g_support: float = 0.25
    g_cross: float = 0.15
    g_state_to_inh: float = 0.18
    g_state_inhibition: float = 0.35
    g_sensory: float = 1.5
    g_afferent_inhibition: float = 2.0
    g_interneuron_in: float = 1.8
    g_readout: float = 5.5
    g_pool_in: float = 0.025
    g_pool_out: float = 0.8
    g_competition_in: float = 0.4
    g_competition_out: float = 0.6
    g_afferent_mod: float = 1.0
    g_effort_mod: float = 0.55
    g_mod_inhibition: float = 1.0
    noise_rate_hz: float = 40.0
    noise_g: float = 0.08

    def __post_init__(self):
        for k,v in asdict(self).items():
            if not np.isfinite(v) or v<=0: raise ValueError(k+" must be finite and positive")
        for k in ("ring_size","state_rings","motor_size","sensor_range","topology_seed"):
            if int(getattr(self,k))!=getattr(self,k): raise ValueError(k+" must be integer")
        if self.ring_size<8 or self.state_rings!=4 or self.motor_size<2:
            raise ValueError("v0.50 requires 4 rings, >=8 cells/ring and >=2 cells/motor")
        if self.sensor_range>3 or not 0<self.satiety_midpoint<1:
            raise ValueError("Invalid receptor range/normalization")
        if self.dt_ms>.1 or self.ring_delay_ms<=3:
            raise ValueError("dt must be <=.1 ms and ring delay > refractory (3 ms)")
        if not np.isclose(self.snn_step_ms/self.dt_ms,round(self.snn_step_ms/self.dt_ms)):
            raise ValueError("Control window must be an integer multiple of dt")
        for name in MOD_NAMES:
            stem=name[:-3].lower()
            if getattr(self,stem+"_release")>1: raise ValueError("Release must be <=1")
            if getattr(self,stem+"_tau_ms")<=self.adaptation_tau_ms:
                raise ValueError("Chemical decay must be slower than spike adaptation")


def sample_params(rng):
    """Offline weight search. Topology, body, sensor and temporal budgets fixed."""
    return RingsParams(state_tonic_mv=rng.uniform(16,19),
        g_ring=rng.uniform(.85,1.3),g_support=rng.uniform(.15,.35),
        g_cross=rng.uniform(.1,.25),g_state_inhibition=rng.uniform(.25,.5),
        g_sensory=rng.uniform(1.1,2),g_readout=rng.uniform(4.5,6.5),
        g_afferent_inhibition=rng.uniform(1.5,3),
        foraging_drive_mv=rng.uniform(6,12),defense_drive_mv=rng.uniform(10,18),
        fatigue_drive_mv=rng.uniform(8,16),curiosity_drive_mv=rng.uniform(4,9))


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


MOTOR_NAMES = ("MotorLeft","MotorRight","MotorForward","MotorRest")
STATE_DIAGNOSTICS = ("StateRateHz","StateActiveFraction","StateTraceNorm","StatePatternChange")
ABLATIONS = ("full","no-state-recurrence","no-chemistry","both")


class RingsSNNController:
    """Receptors -> persistent mixed state -> motor readout -> body.

    Four rings are anatomical partitions, NOT four mutually exclusive modes.
    Each contains overlapping sensory, body and action-related selectivities.
    """
    def __init__(self, params, seed=1, record=False, ablation="full"):
        if ablation not in ABLATIONS: raise ValueError("Unknown ablation: "+ablation)
        start_scope()
        defaultclock.dt=params.dt_ms*ms
        brian_seed((int(seed)+2003)%2**32)
        self.action_rng=random.Random(int(seed)+3001)
        self.params,self.record,self.ablation=params,record,ablation
        self.snn_step_ms,self.ring_size=params.snn_step_ms,params.ring_size
        self.rings={n:RingSpec(n,params.ring_size,k*params.ring_size,(k+1)*params.ring_size)
                    for k,n in enumerate(("StateA","StateB","StateC","StateD"))}
        self.n_state=params.ring_size*4
        self.motor_names=MOTOR_NAMES
        self.input_names=["low_energy","food_left","food_right","food_ahead",
            "danger_left","danger_right","danger_ahead","wall_ahead","danger_contact",
            "satiety","effort","turn_feedback"]
        self.inh_names=["StateInhA","StateInhB","StateInhC","StateInhD",
            "CompetitionLeft","CompetitionRight","CompetitionForward","CompetitionRest",
            "MotorPool","DangerAhead","WallAhead","Satiety","TurnFeedback","Events"]
        recurrence=float(ablation not in ("no-state-recurrence","both"))
        chemistry=float(ablation not in ("no-chemistry","both"))
        common="""
            dv/dt=(-65*mV-v+drive+ge*(0*mV-v)+gi*(-80*mV-v))/tau_v : volt (unless refractory)
            dge/dt=-ge/tau_e : 1
            dgi/dt=-gi/tau_i : 1
            tau_v : second (constant)
        """
        ns=dict(tau_e=params.excitatory_tau_ms*ms,tau_i=params.inhibitory_tau_ms*ms)
        self.receptors=NeuronGroup(12,common+"drive : volt",threshold="v>-50*mV",
            reset="v=-65*mV",refractory=3*ms,method="euler",namespace=ns)
        self.receptors.v=-65*mV;self.receptors.tau_v=10*ms
        self.I=NeuronGroup(len(self.inh_names),common+"drive : volt",threshold="v>-50*mV",
            reset="v=-65*mV",refractory=2*ms,method="euler",namespace=ns)
        self.I.v=-65*mV;self.I.tau_v=10*ms
        self.modulators=NeuronGroup(4,common+"""
            drive=tonic : volt
            dc/dt=-c/tau_chem : 1
            tonic : volt (constant)
            tau_chem : second (constant)
            release : 1 (constant)
        """,threshold="v>-50*mV",reset="v=-65*mV; c+=release*(1-c)",
            refractory=params.mod_refractory_ms*ms,method="exponential_euler",namespace=ns)
        self.modulators.v=-65*mV;self.modulators.tau_v=params.mod_tau_ms*ms
        self.modulators.tonic=[0,0,0,params.curiosity_tonic_mv]*mV
        for k,n in enumerate(MOD_NAMES):
            self.modulators.tau_chem[k]=getattr(params,n[:-3].lower()+"_tau_ms")*ms
            self.modulators.release[k]=getattr(params,n[:-3].lower()+"_release")
        self.G=NeuronGroup(self.n_state,common+"""
            da/dt=-a/tau_a : volt
            dx/dt=-x/tau_x : 1
            drive=tonic-a+chemical_effect*(wf*cf/(kd+cf)+wd*cd/(kd+cd)
                  +wh*ch/(kd+ch)+wq*cq/(kd+cq)) : volt
            tonic : volt (constant)
            wf : volt (constant)
            wd : volt (constant)
            wh : volt (constant)
            wq : volt (constant)
            cf : 1 (linked)
            cd : 1 (linked)
            ch : 1 (linked)
            cq : 1 (linked)
        """,threshold="v>-50*mV",reset="v=-65*mV; a+=adapt; x+=1",
            refractory=3*ms,method="euler",namespace=dict(ns,
                tau_a=params.adaptation_tau_ms*ms,tau_x=params.trace_tau_ms*ms,
                kd=params.receptor_kd,chemical_effect=chemistry,adapt=params.adaptation_jump_mv*mV))
        for k,v in enumerate(("cf","cd","ch","cq")):
            setattr(self.G,v,linked_var(self.modulators,"c",index=np.full(self.n_state,k,dtype=int)))
        self.G.tau_v=params.membrane_tau_ms*ms
        # Fixed structural seed across episodes: world seeds do NOT redraw the agent.
        rng=np.random.RandomState(params.topology_seed)
        phi=np.concatenate([2*np.pi*np.arange(params.ring_size)/params.ring_size+k*np.pi/7 for k in range(4)])
        # Smooth, overlapping innate projection basis (L,R,F,Rest), present in EVERY ring.
        anchors=np.array([-np.pi/2,np.pi/2,0,np.pi])
        affinity=np.exp(2.0*np.cos(phi[None,:]-anchors[:,None]))
        affinity/=affinity.sum(axis=0,keepdims=True)
        self.affinity=affinity
        left,right,forward,rest=affinity
        mobility=left+right+forward
        # Independent mixed body component: representation is not merely four motor scores.
        mixed=np.array([np.sin((k+2)*phi+rng.uniform(-np.pi,np.pi,self.n_state)) for k in range(4)])
        self.chemical_weights=np.array([
            params.foraging_drive_mv*(mobility-.6*rest+.3*mixed[0]),
            params.defense_drive_mv*(forward+.45*(left+right)-rest+.3*mixed[1]),
            params.fatigue_drive_mv*(rest-.45*mobility+.3*mixed[2]),
            params.curiosity_drive_mv*(mobility-.4*rest+.3*mixed[3])])
        for k,n in enumerate(("wf","wd","wh","wq")):setattr(self.G,n,self.chemical_weights[k]*mV)
        self.G.tonic=(params.state_tonic_mv+params.tonic_spread_mv*rng.uniform(-1,1,self.n_state))*mV
        self.G.v=rng.uniform(-65,-50.1,self.n_state)*mV
        self.motor=NeuronGroup(4*params.motor_size,common+"drive : volt",
            threshold="v>-50*mV",reset="v=-65*mV",refractory=3*ms,method="euler",namespace=ns)
        self.motor.v=rng.uniform(-65,-60,4*params.motor_size)*mV
        self.motor.tau_v=15*ms
        self.connections=[];self.connection_manifest=[];self.recurrent_synapses=[]
        groups={id(self.receptors):"receptors",id(self.G):"state",id(self.I):"inhibitory",
                id(self.modulators):"modulators",id(self.motor):"motor"}
        def conn(label,src,dst,ii,jj,w,inhibitory=False,delay=1):
            if inhibitory!=(src is self.I):raise ValueError("Dale convention violated")
            ii=np.asarray(ii,dtype=int);jj=np.asarray(jj,dtype=int)
            weights=np.broadcast_to(np.asarray(w,dtype=float),ii.shape).copy()
            if np.any(weights<0):raise ValueError("Negative conductance")
            syn=Synapses(src,dst,"w : 1 (constant)",on_pre=("gi" if inhibitory else "ge")+"_post+=w")
            syn.connect(i=ii,j=jj);syn.w=weights;syn.delay=delay*ms
            self.connections.append(syn)
            self.connection_manifest.append(dict(label=label,source=groups[id(src)],target=groups[id(dst)],
                sign="inhibitory" if inhibitory else "excitatory",delay_ms=delay,
                i=ii.tolist(),j=jj.tolist(),weights=weights.tolist()))
            return syn
        n=self.n_state;rs=params.ring_size;ids=np.arange(n)
        for offset,g in ((1,params.g_ring),(2,params.g_support)):
            jj=(ids//rs)*rs+(ids%rs+offset)%rs
            self.recurrent_synapses.append(conn("within_ring_%d"%offset,self.G,self.G,ids,jj,g*recurrence,delay=params.ring_delay_ms))
        # Bidirectional cross-ring coupling, four outgoing cross links per neuron.
        ii=[];jj=[]
        for i in ids:
            ring,pos=divmod(int(i),rs)
            for dr in (-1,1):
                for shift in (0,3):
                    ii.append(i);jj.append(((ring+dr)%4)*rs+(pos+shift)%rs)
        self.recurrent_synapses.append(conn("cross_ring",self.G,self.G,ii,jj,params.g_cross*recurrence,delay=3))
        conn("state_feedback_to_I",self.G,self.I,ids,ids//rs,params.g_state_to_inh*recurrence)
        ii=np.repeat(np.arange(4),n);jj=np.tile(ids,4)
        conn("state_feedback_inhibition",self.I,self.G,ii,jj,
             params.g_state_inhibition*(1+.5*(ii==jj//rs)),True)
        # Sensor profiles are fixed projections into the MIXED recurrent state.
        profiles=[.5*forward+.2*mobility,left,right,forward,
                  right+.5*forward,left+.5*forward,left+right,left+right,
                  forward,rest,.7*rest,.7*forward+.2*rest]
        for k,profile in enumerate(profiles):
            # Distributed positive afference includes a small non-motor-specific component.
            weights=params.g_sensory*(.85*profile+.15*(.5+.5*mixed[k%4]))
            conn("afferent_"+self.input_names[k],self.receptors,self.G,np.full(n,k),ids,weights)
        def rec(s):return self.input_names.index(s)
        def inh(s):return self.inh_names.index(s)
        for sensor,iname,profile in (("danger_ahead","DangerAhead",forward),
                ("wall_ahead","WallAhead",forward),("satiety","Satiety",mobility),
                ("turn_feedback","TurnFeedback",left+right)):
            conn(sensor+"_to_I",self.receptors,self.I,[rec(sensor)],[inh(iname)],params.g_interneuron_in)
            conn(sensor+"_state_inhibition",self.I,self.G,np.full(n,inh(iname)),ids,
                 params.g_afferent_inhibition*profile,True)
        for sensor in ("food_left","food_right","food_ahead","low_energy"):
            conn(sensor+"_secretion",self.receptors,self.modulators,[rec(sensor)],[0],params.g_afferent_mod)
        for sensor in ("danger_left","danger_right","danger_ahead","danger_contact"):
            conn(sensor+"_secretion",self.receptors,self.modulators,[rec(sensor)],[1],params.g_afferent_mod)
        conn("effort_secretion",self.receptors,self.modulators,[rec("effort")],[2],params.g_effort_mod)
        conn("satiety_secretion_inhibition",self.I,self.modulators,[inh("Satiety")]*2,[0,3],params.g_mod_inhibition,True)
        event_ids=[rec(s) for s in ("food_left","food_right","food_ahead","danger_left","danger_right","danger_ahead","danger_contact")]
        conn("events_to_I",self.receptors,self.I,event_ids,[inh("Events")]*len(event_ids),params.g_interneuron_in)
        conn("events_curiosity",self.I,self.modulators,[inh("Events")],[3],params.g_mod_inhibition,True)
        # ONLY state spikes excite motor neurons. Readout has no sensory/chemical variables.
        for k in range(4):
            mids=np.arange(k*params.motor_size,(k+1)*params.motor_size)
            ii=np.repeat(ids,params.motor_size);jj=np.tile(mids,n)
            weights=np.repeat(params.g_readout*affinity[k]/affinity[k].sum(),params.motor_size)
            conn("state_readout_"+MOTOR_NAMES[k],self.G,self.motor,ii,jj,weights)
            conn("motor_competition_in_"+str(k),self.motor,self.I,mids,np.full(len(mids),4+k),params.g_competition_in)
            other=np.array([j for j in range(4*params.motor_size) if j//params.motor_size!=k])
            conn("motor_competition_out_"+str(k),self.I,self.motor,np.full(len(other),4+k),other,params.g_competition_out,True)
        conn("state_common_mode_to_I",self.G,self.I,ids,np.full(n,inh("MotorPool")),params.g_pool_in)
        conn("readout_common_mode_inhibition",self.I,self.motor,
             np.full(4*params.motor_size,inh("MotorPool")),np.arange(4*params.motor_size),params.g_pool_out,True)
        self.noise=PoissonGroup(n,rates=params.noise_rate_hz*Hz)
        groups[id(self.noise)]="background"
        conn("background",self.noise,self.G,ids,ids,params.noise_g)
        self.spikes=SpikeMonitor(self.G,record=record)
        self.motor_spikes=SpikeMonitor(self.motor,record=record)
        self.mod_spikes=SpikeMonitor(self.modulators,record=record)
        self.receptor_spikes=SpikeMonitor(self.receptors,record=record)
        self.inh_spikes=SpikeMonitor(self.I,record=record)
        self.state_mon=StateMonitor(self.G,["v","a","x","ge","gi"],record=True,dt=1*ms) if record else None
        self.motor_mon=StateMonitor(self.motor,["v","ge","gi"],record=True,dt=1*ms) if record else None
        self.chem_mon=StateMonitor(self.modulators,["c","v"],record=True,dt=1*ms) if record else None
        self.receptor_mon=StateMonitor(self.receptors,"v",record=True,dt=1*ms) if record else None
        self.inh_mon=StateMonitor(self.I,"v",record=True,dt=1*ms) if record else None
        self.net=Network(self.receptors,self.G,self.I,self.modulators,self.motor,self.noise,*self.connections,
            self.spikes,self.motor_spikes,self.mod_spikes,self.receptor_spikes,self.inh_spikes,
            *([self.state_mon,self.motor_mon,self.chem_mon,self.receptor_mon,self.inh_mon] if record else []))
        self.recent_spikes=np.zeros(n);self.recent_motor_spikes=np.zeros(4*params.motor_size)
        self.last_inputs=dict.fromkeys(self.input_names,0.0)
        self.last_mod_spikes=np.zeros(4,dtype=int);self.last_receptor_spikes=np.zeros(12,dtype=int)
        self.last_inh_spikes=np.zeros(len(self.inh_names),dtype=int)
        self.previous_pattern=np.zeros(n);self.last_decision="silent"
        self.step_state_trace=[] if record else None
        self.step_value_names=(*self.rings,*MOTOR_NAMES,*MOD_NAMES,*STATE_DIAGNOSTICS)

    def encode_observation(self,obs,env):
        energy,fl,fa,fr,dl,da,dr=obs["rstate"]
        hunger=1-float(energy)
        satiety=max(0,(float(energy)-self.params.satiety_midpoint)/(1-self.params.satiety_midpoint))
        return np.clip([hunger,fl,fr,fa,dl,dr,da,env._wall_ahead(),env.danger_contact_signal,
                        satiety,env.last_effort,env.last_turn],0,1)

    def step(self,obs,env):
        return self.advance(self.encode_observation(obs,env))

    def advance(self,signals):
        """One fixed neural window. Also used by explicit diagnostic stimuli."""
        signals=np.asarray(signals,dtype=float)
        if signals.shape!=(12,) or not np.isfinite(signals).all() or np.any((signals<0)|(signals>1)):
            raise ValueError("Receptor stimuli must be 12 finite values in [0,1]")
        self.last_inputs=dict(zip(self.input_names,map(float,signals)))
        self.receptors.drive=self.params.receptor_drive_mv*signals/(self.params.receptor_half+signals)*mV
        monitors=(self.spikes,self.motor_spikes,self.mod_spikes,self.receptor_spikes,self.inh_spikes)
        before=[np.array(m.count[:]) for m in monitors]
        self.net.run(self.snn_step_ms*ms,namespace={})
        counts=[np.array(m.count[:])-n for m,n in zip(monitors,before)]
        self.recent_spikes,self.recent_motor_spikes,self.last_mod_spikes,self.last_receptor_spikes,self.last_inh_spikes=counts
        acts=self.ring_activations()
        acts.update({n:float(self.recent_motor_spikes[k*self.params.motor_size:(k+1)*self.params.motor_size].mean()) for k,n in enumerate(MOTOR_NAMES)})
        acts.update(zip(MOD_NAMES,map(float,self.modulators.c[:])))
        acts["EnergyLow"]=self.last_inputs["low_energy"]
        x=np.array(self.G.x[:]);norm=float(np.linalg.norm(x));pattern=x/max(norm,1e-12)
        acts.update(StateRateHz=float(self.recent_spikes.sum()/self.n_state/(self.snn_step_ms/1000)),
                    StateActiveFraction=float(np.mean(self.recent_spikes>0)),StateTraceNorm=norm,
                    StatePatternChange=float(np.linalg.norm(pattern-self.previous_pattern)))
        self.previous_pattern=pattern
        if self.record:self.step_state_trace.append(x.copy())
        return self.decode_action(acts),acts

    def ring_activations(self):
        return {n:float(self.recent_spikes[r.start:r.end].mean()) for n,r in self.rings.items()}

    def decode_action(self,acts):
        """Motor spikes only; silence -> no-op, exact ties -> independent seeded RNG."""
        scores=np.array([acts[n] for n in MOTOR_NAMES])
        if scores.max()<=0:self.last_decision="silent";return 6
        chosen=self.action_rng.choice(np.flatnonzero(scores==scores.max()).tolist())
        self.last_decision="active_rest" if chosen==3 else "motor"
        return (0,1,2,6)[chosen]

    def audit(self):
        import hashlib
        from importlib.metadata import version
        return dict(agent_version=AGENT_VERSION,record_schema=50,ablation=self.ablation,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            python=sys.version,runtime={n:version(n) for n in ("brian2","minigrid","gymnasium","numpy")},
            params=asdict(self.params),assumptions=__doc__,
            populations=dict(receptors=self.input_names,inhibitory=self.inh_names,
                rings={n:[r.start,r.end] for n,r in self.rings.items()},
                motor={n:[k*self.params.motor_size,(k+1)*self.params.motor_size] for k,n in enumerate(MOTOR_NAMES)},
                modulators=list(MOD_NAMES)),connections=self.connection_manifest,
            state_equations=str(self.G.equations),motor_equations=str(self.motor.equations),
            chemical_weights_mv=self.chemical_weights.tolist(),readout_affinity=self.affinity.tolist(),
            tonic_mv=np.asarray(self.G.tonic/mV).tolist(),
            rng_offsets=dict(environment=1009,brian=2003,action=3001),
            ablation_definition="no-state-recurrence zeros E->E and E->state-I feedback; no-chemistry zeros chemical membrane coupling, secretors retained",
            conductance_units="relative to leak",reversal_mv=dict(excitatory=0,inhibitory=-80))


# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------

class RingsVisualizer:
    def __init__(self,controller):
        self.controller=controller
        self.fig=plt.figure(figsize=(22,10),layout="constrained")
        gs=self.fig.add_gridspec(3,4,width_ratios=[1,1,1.7,2.2])
        self.ax_env=self.fig.add_subplot(gs[:2,:2])
        self.ax_sensors=self.fig.add_subplot(gs[2,:2])
        self.ax_rings=self.fig.add_subplot(gs[0,2])
        self.ax_motor=self.fig.add_subplot(gs[1,2])
        self.ax_mods=self.fig.add_subplot(gs[2,2])
        self.ax_timeline=self.fig.add_subplot(gs[:,3]);self.timeline_history=[]
        plt.ion();plt.show(block=False)

    def update(self,env,acts,step_idx,info,frame=None):
        c=self.controller
        self.timeline_history=(self.timeline_history+[dict(c.last_inputs,**acts,step=step_idx)])[-60:]
        ax=self.ax_env;ax.clear();ax.imshow(env.render() if frame is None else frame);ax.axis("off")
        ax.set_title("RINGS %s | %s | step %d | energy %.3f\nfood %s | contacts %s | action %s | %s\nScene and inputs BEFORE action"%
            (AGENT_VERSION,c.ablation,step_idx,info.get('energy',0),info.get('food_collected',0),
             info.get('danger_contacts',0),info.get('last_action'),c.last_decision))
        ax=self.ax_sensors;ax.clear()
        ax.bar(range(12),[c.last_inputs[n] for n in c.input_names],color="slategray")
        ax.set_xticks(range(12),c.input_names,rotation=35,ha="right",fontsize=8);ax.set_ylim(0,1.1)
        ax.set_title("Receptor stimuli (not motor commands)")
        ax=self.ax_rings;ax.clear()
        x=np.asarray(c.G.x[:]).reshape(4,c.params.ring_size)
        ax.imshow(x,aspect="auto",vmin=0,vmax=6,cmap="viridis",interpolation="nearest")
        ax.set_yticks(range(4),list(c.rings));ax.set_xlabel("Neuron within anatomical ring; spike trace x (0..6 color scale)")
        ax.set_title("COUPLED SELF-STATE | %.1f Hz | active cells %.0f%%\nPattern change %.3f; zero-spike window: %s"%
            (acts['StateRateHz'],100*acts['StateActiveFraction'],acts['StatePatternChange'],acts['StateRateHz']==0))
        ax=self.ax_motor;ax.clear()
        ax.bar(range(4),[acts[n] for n in MOTOR_NAMES],color="steelblue")
        ax.set_xticks(range(4),[n.replace('Motor','') for n in MOTOR_NAMES]);ax.set_ylim(0,np.ceil(c.snn_step_ms/3))
        ax.set_title("MOTOR READOUT (separate population)\nspikes / cell / window; silence is not state silence")
        ax=self.ax_mods;ax.clear()
        ax.bar(range(4),[acts[n] for n in MOD_NAMES],color="darkorange")
        ax.set_xticks(range(4),[n.replace('Mod','') for n in MOD_NAMES],fontsize=9);ax.set_ylim(0,1)
        ax.set_title("NEUROMODULATORS | normalized concentration\nchemical effects enter self-state only")
        ax=self.ax_timeline;ax.clear()
        lanes=["low_energy","satiety","effort","food_ahead","danger_ahead","wall_ahead",*c.rings,*MOTOR_NAMES,*MOD_NAMES]
        xs=[h['step'] for h in self.timeline_history]
        for k,n in enumerate(lanes):
            base=len(lanes)-k-1
            scale=np.ceil(c.snn_step_ms/3) if n in c.rings or n in MOTOR_NAMES else 1
            vals=np.array([h[n] for h in self.timeline_history])/scale
            color="teal" if n in c.rings else "steelblue" if n in MOTOR_NAMES else "darkorange" if n in MOD_NAMES else "gray"
            ax.plot(xs,base+.75*np.clip(vals,0,1),color=color,lw=1.4);ax.axhline(base,color="gray",alpha=.2,lw=.5)
        ax.set_yticks(range(len(lanes)),list(reversed(lanes)),fontsize=8)
        ax.set_xlim(max(0,step_idx-59),max(59,step_idx));ax.set_ylim(-.3,len(lanes))
        ax.set_title("ONE SHARED TIMELINE\ninputs / self-state / motor readout / chemistry")
        ax.set_xlabel("Environment step")
        self.fig.canvas.draw();self.fig.canvas.flush_events();plt.pause(.001)


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
               "food_collected", "danger_contacts", *c.input_names, *c.step_value_names]
    data = dict(metadata_json=np.asarray(json.dumps(metadata)),
        step_columns=np.asarray(columns), steps=np.asarray(trajectory,dtype=float).reshape(-1,len(columns)),
        ring_names=np.asarray(list(c.rings)),
        ring_bounds=np.asarray([[r.start,r.end] for r in c.rings.values()]),
        motor_names=np.asarray(MOTOR_NAMES),
        motor_bounds=np.asarray([[k*c.params.motor_size,(k+1)*c.params.motor_size] for k in range(4)]),
        motor_v_mv=np.asarray(c.motor_mon.v/mV,dtype=np.float32),
        motor_ge=np.asarray(c.motor_mon.ge,dtype=np.float32),
        motor_gi=np.asarray(c.motor_mon.gi,dtype=np.float32),
        state_trace=np.asarray(c.state_mon.x,dtype=np.float32),
        step_state_trace=np.asarray(c.step_state_trace[:len(trajectory)],dtype=np.float32).reshape(-1,c.n_state),
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
    for name, mon in (("ring",c.spikes),("motor",c.motor_spikes),("mod",c.mod_spikes),("receptor",c.receptor_spikes),("inhibitory",c.inh_spikes)):
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
    """Unified world/state/motor/chemical view with a common timeline, schema 50."""
    def __init__(self,path,fps,ablation="full"):
        if not np.isfinite(fps) or fps<=0:raise ValueError("FPS must be finite and positive")
        self.path=Path(path)
        if self.path.suffix.lower()!='.mp4':raise ValueError("Video path must end in .mp4")
        if self.path.exists():raise FileExistsError(str(self.path))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg
        except ImportError as exc:raise RuntimeError("Video requires imageio imageio-ffmpeg pillow") from exc
        self.writer=imageio.get_writer(str(self.path),fps=fps,codec='libx264',macro_block_size=2)
        self.count=0;self.history=[];self.ablation=ablation

    def append(self,frame,step,energy,action,acts,state_trace=None):
        from PIL import Image,ImageDraw,ImageFont
        canvas=Image.new('RGB',(1280,1050),(15,23,36));d=ImageDraw.Draw(canvas)
        try:
            font=ImageFont.truetype('DejaVuSans.ttf',17);small=ImageFont.truetype('DejaVuSans.ttf',13)
        except OSError:font=small=ImageFont.load_default()
        fg=(225,233,245);teal=(67,207,194);blue=(97,164,245);orange=(242,175,79)
        d.text((20,14),'RINGS '+AGENT_VERSION+' | '+self.ablation+' | persistent dynamical self-state',fill=fg,font=font)
        d.text((20,42),'Step %d | energy %.3f | action %d | scene BEFORE action'%(step,energy,action),fill=fg,font=small)
        im=Image.fromarray(frame);im.thumbnail((555,475));canvas.paste(im,(20,82))
        d.text((610,78),'COUPLED STATE RINGS (trace x; fixed color scale 0..6)',fill=teal,font=small)
        ring_names=[n for n in acts if n.startswith('State') and n in ('StateA','StateB','StateC','StateD')]
        if state_trace is not None:
            values=np.asarray(state_trace).reshape(4,-1)
            for k,name in enumerate(('StateA','StateB','StateC','StateD')):
                cx,cy=735+280*(k%2),195+175*(k//2);radius=60
                d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=(55,80,95),width=2)
                for j,x in enumerate(values[k]):
                    angle=2*np.pi*j/len(values[k]);px=cx+radius*np.cos(angle);py=cy+radius*np.sin(angle)
                    f=min(1,max(0,float(x)/6));color=(int(35+170*f),int(70+165*f),int(100+75*f))
                    d.ellipse((px-6,py-6,px+6,py+6),fill=color)
                d.text((cx-34,cy-13),name,fill=fg,font=small)
                d.text((cx-25,cy+8),'%.2f'%acts.get(name,0),fill=teal,font=small)
        else:
            d.text((650,230),'Per-neuron state traces unavailable in this recording',fill=fg,font=small)
        d.text((610,442),'Centers: spikes/cell/window; node colors: exponential spike traces',fill=fg,font=small)
        d.text((610,466),'State %.1f Hz | active %.0f%% | pattern change %.3f'%
            (acts.get('StateRateHz',0),100*acts.get('StateActiveFraction',0),acts.get('StatePatternChange',0)),fill=fg,font=small)
        d.text((20,566),'MOTOR READOUT: spikes/cell/window (0..10)',fill=blue,font=small)
        d.text((655,566),'CHEMICAL CONCENTRATIONS (0..1)',fill=orange,font=small)
        for k,(motor,mod) in enumerate(zip(MOTOR_NAMES,MOD_NAMES)):
            y=594+27*k
            d.text((20,y),motor,fill=fg,font=small);d.rectangle((160,y+2,440,y+16),fill=(37,50,70))
            d.rectangle((160,y+2,160+280*min(1,acts.get(motor,0)/10),y+16),fill=blue)
            d.text((450,y),'%.2f'%acts.get(motor,0),fill=fg,font=small)
            d.text((655,y),mod,fill=fg,font=small);d.rectangle((790,y+2,1190,y+16),fill=(37,50,70))
            d.rectangle((790,y+2,790+400*min(1,acts.get(mod,0)),y+16),fill=orange)
            d.text((1200,y),'%.2f'%acts.get(mod,0),fill=fg,font=small)
        self.history=(self.history+[dict(acts)])[-60:]
        d.text((20,716),'SHARED TIMELINE: state / motor / chemistry | last 60 control windows',fill=fg,font=small)
        lanes=[*ring_names,*MOTOR_NAMES,*MOD_NAMES]
        for k,n in enumerate(lanes):
            y=745+24*k;d.text((20,y),n,fill=fg,font=small)
            d.line((175,y+19,1245,y+19),fill=(50,65,82))
            scale=1 if n in MOD_NAMES else 10
            color=orange if n in MOD_NAMES else blue if n in MOTOR_NAMES else teal
            points=[(175+1070*(60-len(self.history)+j)/59,y+19-18*min(1,max(0,h.get(n,0)/scale))) for j,h in enumerate(self.history)]
            if len(points)>1:d.line(points,fill=color,width=2)
        self.writer.append_data(np.asarray(canvas));self.count+=1

    def close(self):
        self.writer.close();print('Video saved: %s (%d frames)'%(self.path.resolve(),self.count),flush=True)


def replay_recording(path,output,fps=10):
    with np.load(path,allow_pickle=False) as z:
        if "audit_json" not in z or json.loads(str(z["audit_json"])).get("record_schema")!=50:
            raise ValueError("Use the matching model for older recordings; v0.50 replay requires schema 50")
        if "scene_grids" not in z:
            raise ValueError("Scene snapshots are missing")
        meta = json.loads(str(z["metadata_json"]))
        rows,grids = z["steps"],z["scene_grids"]
        if len(rows)!=len(grids) or not len(rows):
            raise ValueError("Missing or inconsistent scene snapshots")
        columns = list(z["step_columns"])
        env = RingsHomeostasisEnv(size=meta["grid_size"],sensor_range=meta["params"]["sensor_range"],render_mode="rgb_array")
        env.reset(seed=meta["seed"])
        writer = VideoWriter(output,fps,meta.get("ablation","full"))
        try:
            for row,grid in zip(rows,grids):
                r = dict(zip(columns,row))
                env.grid,_ = Grid.decode(grid)
                env.agent_pos = (int(r["x"]),int(r["y"]))
                env.agent_dir = int(r["direction"])
                trace=None
                if "step_state_trace" in z and len(z["step_state_trace"])==len(rows):
                    trace=z["step_state_trace"][int(r["step"])]
                elif "state_trace" in z and len(z["state_t_ms"]):
                    target=(r["step"]+1)*meta["params"]["snn_step_ms"]
                    ix=max(0,int(np.searchsorted(z["state_t_ms"],target,side="left"))-1)
                    trace=z["state_trace"][:,ix]
                writer.append(env.render(),r["step"],r["energy_before"],r["action"],r,trace)
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
    state_silent_steps: int = 0
    mean_state_rate_hz: float = 0.0


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
    ablation: str = "full",
) -> EpisodeResult:
    if record_path and not record:
        raise ValueError("record_path requires record=True / --record on")
    if record:
        record_path = record_path or str(Path("recordings") / ("rings_v050_seed%d_%s.npz" % (seed,uuid.uuid4().hex)))
        if Path(record_path).suffix.lower() != ".npz":
            raise ValueError("record_path must end in .npz")
        Path(record_path).parent.mkdir(parents=True,exist_ok=True)
        if Path(record_path).exists():
            raise FileExistsError("Recording already exists: " + str(record_path))
    params = params or RingsParams()
    env = RingsHomeostasisEnv(size=grid_size, max_steps=max_steps, sensor_range=params.sensor_range, danger_speed=danger_speed, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    video_writer = VideoWriter(video_path or "rings_v050.mp4", video_fps, ablation) if video else None
    controller = RingsSNNController(params=params, seed=seed, record=record, ablation=ablation)
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
                        *controller.step_value_names, "EnergyLow",
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
    state_silent_steps = 0
    state_rate_sum = 0.0
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
            state_silent_steps += int(acts["StateRateHz"] == 0)
            state_rate_sum += acts["StateRateHz"]
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
                video_writer.append(frame_before, step_idx, energy_before, action, acts, np.asarray(controller.G.x[:]))
            if record:
                scene_grids.append(grid_before)
                trajectory.append([step_idx,*pos_before,dir_before,energy_before,float(info["energy"]),action,
                    info["food_collected"],info["danger_contacts"],
                    *[controller.last_inputs[n] for n in controller.input_names],
                    *[acts[n] for n in controller.step_value_names]])

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
                agent_version=AGENT_VERSION,seed=seed,params=asdict(params),ablation=ablation,
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
        state_silent_steps=state_silent_steps,mean_state_rate_hz=state_rate_sum/max(1,step_idx),
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
    ablation: str = "full",
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
        results.append(run_episode(seed=s, params=params, visualize=False, max_steps=max_steps, status=status, record=record, grid_size=grid_size, danger_speed=danger_speed, debug_log=episode_debug_log, ablation=ablation, progress_callback=telemetry if _TUNE_QUEUE is not None else None))
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
        "state_continuity_valid": int(all(r.state_silent_steps==0 for r in results)),
        "mean_state_silent_steps": float(np.mean([r.state_silent_steps for r in results])),
        "mean_state_rate_hz": float(np.mean([r.mean_state_rate_hz for r in results])),
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
    output_path: str = "rings_v050_best.json",
    record: bool = False,
    grid_size: int = 15,
    danger_speed: float = 0.0,
    debug_log_dir: Optional[str] = None,
    workers: int = 32,
    resume: bool = False,
    progress_interval: float = 5.0,
    ablation: str = "full",
) -> Tuple[RingsParams, Dict[str, float]]:
    """Parallel search, with an explicit empirical state-continuity validity gate.

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
                    debug_log_dir=debug_log_dir, ablation=ablation)
    config = dict(objective="mean_survival_steps_subject_to_no_silent_state_windows", agent_version=AGENT_VERSION, trials=trials, search_seed=search_seed,
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
        eligible=[r for r in completed.values() if r["metrics"]["state_continuity_valid"]]
        best = max(eligible, key=lambda row: (row["metrics"]["score"], -row["trial"])) if eligible else None
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
                          f"danger={metrics['mean_danger']:.2f}; best_trial={best['trial'] if best else 'none (state continuity)'}", flush=True)
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
        raise RuntimeError("No candidate passed the explicit state-continuity constraint. All trials remain in the checkpoint history.")
    print(f"Saved {len(completed)} candidates to {output_path}; best trial={best['trial']}", flush=True)
    return RingsParams(**best["params"]), best["metrics"]


def run_circuit_checks():
    """Explicit experimental interventions, never called during run/tune.

    Synthetic receptor currents, input removal, pathway cuts and equalization of
    peripheral/readout states isolate mechanisms. No runtime rescue is installed.
    Reported history distances are diagnostic, not a proof of an attractor.
    """
    report={}
    def require(name,condition):
        if not condition:raise AssertionError(name)
    def emit(name,value):
        report[name]=value;print(name+": "+json.dumps(value,sort_keys=True),flush=True)
    def normalize(x):return x/max(float(np.linalg.norm(x)),1e-12)
    p=RingsParams();zero=np.zeros(12)
    c=RingsSNNController(p,seed=19,record=True)
    rates=[]
    for _ in range(30):
        _,a=c.advance(zero);rates.append(a['StateRateHz'])
    c.noise.rates=0*Hz
    noise_off=[]
    for _ in range(20):
        _,a=c.advance(zero);noise_off.append(a['StateRateHz'])
    ts=np.asarray(c.spikes.t/ms)
    gaps=np.diff(np.r_[0,ts,float(c.net.t/ms)])
    require('Global state became silent without afference',min(rates+noise_off)>0)
    require('Global inter-spike gap exceeds a control window',float(gaps.max())<p.snn_step_ms)
    emit('continuity',dict(silent_windows=0,mean_rate_hz=float(np.mean(rates)),
        mean_rate_without_noise_hz=float(np.mean(noise_off)),max_global_gap_ms=float(gaps.max()),
        duration_ms=float(c.net.t/ms)))
    # Readout cannot be driven by afferents if the state's output synapses are cut.
    stimulus=zero.copy();stimulus[3]=1
    before=[]
    for _ in range(4):
        _,a=c.advance(stimulus);before.append(sum(a[n] for n in MOTOR_NAMES))
    for syn,path in zip(c.connections,c.connection_manifest):
        if path['source']=='state' and path['target']=='motor':syn.w=0
    after=[]
    for _ in range(6):
        _,a=c.advance(stimulus);after.append(sum(a[n] for n in MOTOR_NAMES))
    require('Readout never responded before intervention',max(before)>0)
    require('Motor activity bypasses state projection',max(after[-3:])==0)
    require('State was killed instead of its readout',a['StateRateHz']>0)
    emit('causal_readout_cut',dict(motor_before=before,motor_after=after,state_rate_after_hz=a['StateRateHz']))
    for path in c.connection_manifest:
        require('Non-state source drives motor',path['target']!='motor' or path['source'] in ('state','inhibitory'))
        if path['target']=='motor' and path['source']=='inhibitory':
            require('Afferent interneuron bypasses state',set(path['i']).issubset(set(range(4,9))))
        require('Negative synaptic conductance',min(path['weights'])>=0)
    emit('topology',dict(no_direct_sensor_or_chemical_motor_path=True,
                         state_cells=c.n_state,motor_cells=4*p.motor_size))
    # Two histories, identical future stimuli, noise and chemical concentrations.
    # Peripheral/motor populations and event queues start fresh. Only the state
    # membrane/adaptation/conductance/feedback-I snapshot is transferred.
    for mode in ('full','no-state-recurrence'):
        c=RingsSNNController(p,seed=19,ablation=mode)
        for _ in range(10):c.advance(zero)
        c.net.store('common');rng=c.action_rng.getstate();pattern=c.previous_pattern.copy();branches=[]
        for side in ('food_left','food_right'):
            c.net.restore('common',restore_random_state=True)
            c.action_rng.setstate(rng);c.previous_pattern=pattern.copy()
            s=zero.copy();s[c.input_names.index(side)]=.5
            for _ in range(6):c.advance(s)
            for _ in range(8):c.advance(zero)
            state=np.array(c.G.x[:]);chem=np.array(c.modulators.c[:])
            # Transfer ONLY the internal state to a fresh, identical probe network.
            # This removes peripheral/motor refractory memory AND pending readout
            # synaptic events; those cannot masquerade as state-dependent behavior.
            # Pending recurrent spikes are intentionally not transferred either.
            state_vars={n:getattr(c.G,n)[:].copy() for n in ("v","a","x","ge","gi","not_refractory")}
            state_vars["lastspike"]=c.G.lastspike[:].copy()-c.net.t
            inh_vars={n:getattr(c.I,n)[:4].copy() for n in ("v","ge","gi","not_refractory")}
            inh_vars["lastspike"]=c.I.lastspike[:4].copy()-c.net.t
            probe=RingsSNNController(p,seed=1019,ablation=mode)
            for n,value in state_vars.items():setattr(probe.G,n,value)
            for n,value in inh_vars.items():getattr(probe.I,n)[:4]=value
            probe.modulators.c=chem
            s=zero.copy();s[3]=.08;response=[]
            for _ in range(5):
                action,a=probe.advance(s);response.append([action,*[a[n] for n in MOTOR_NAMES]])
            branches.append((state,chem,response))
        l,r=branches
        distance=float(np.linalg.norm(normalize(l[0])-normalize(r[0])))
        equal_chem=bool(np.allclose(l[1],r[1]))
        delta=float(np.linalg.norm(np.asarray(l[2])[:,1:]-np.asarray(r[2])[:,1:]))
        if mode=='full':
            require('Distinct histories left no neural-state difference',distance>1e-3)
            require('Chemical mismatch confounds history test',equal_chem)
            require('History has no motor effect',delta>1e-3)
        emit('history_'+mode,dict(sensory_gap_ms=8*p.snn_step_ms,state_distance=distance,
            chemical_concentrations_equal=equal_chem,motor_response_distance=delta,
            probe_protocol="state snapshot into fresh identical periphery/readout; no delayed events transferred",
            response_after_left=l[2],response_after_right=r[2]))
    # Body-state influence must survive relocation from direct motor paths to state.
    for case in ('satiety','low_energy','danger_ahead'):
        c=RingsSNNController(p,seed=19);rows=[]
        for t in range(30):
            s=zero.copy()
            if case!='danger_ahead' or t<4:
                s[c.input_names.index(case)]=.8 if case=='low_energy' else .3 if case=='danger_ahead' else 1
            if rows:
                s[10]=float(rows[-1][0]!=6);s[11]=float(rows[-1][0] in (0,1))
            action,a=c.advance(s);rows.append((action,a,c.last_decision))
        require('Global state silence during body/sensory scenario',all(a['StateRateHz']>0 for _,a,_ in rows))
        if case=='satiety':require('Satiety did not recruit active rest',sum(k=='active_rest' for _,_,k in rows)>=20)
        if case=='low_energy':require('Hunger failed to evoke locomotion',sum(a==2 for a,_,_ in rows)>=5)
        if case=='danger_ahead':require('No movement after threat disappeared',sum(a==2 for a,_,_ in rows[4:14])>=2)
        emit(case,dict(forward_steps=sum(a==2 for a,_,_ in rows),
            turn_steps=sum(a in (0,1) for a,_,_ in rows),active_rest_steps=sum(k=='active_rest' for _,_,k in rows),
            motor_silent_steps=sum(k=='silent' for _,_,k in rows),state_silent_steps=0))
    print('All v0.50 circuit checks passed. Attractor stability and population-level fitness remain separate experiments.',flush=True)
    return report


def load_params(path: str) -> RingsParams:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload.get("best_params", payload)
    if "state_tonic_mv" not in data:
        raise ValueError("v0.50 requires fresh parameters/tuning (missing state_tonic_mv). "
                         "Omit --params to use v0.50 defaults; use a new --output for tuning.")
    unknown = set(data)-set(RingsParams.__dataclass_fields__)
    if unknown:
        raise ValueError("Incompatible parameter schema (v0.50 requires conductance parameters): " + ", ".join(sorted(unknown)))
    return RingsParams(**data)


# -----------------------------------------------------------------------------
# 6. CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--test-report", help="With --self-test: save diagnostic results and source hash as JSON")
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
    parser.add_argument("--params", type=str, default=None, help="Path to v0.50 parameter JSON")
    parser.add_argument("--output", type=str, default="rings_v050_best.json")
    parser.add_argument("--visual", choices=["on", "off"], default="off")
    parser.add_argument("--record", choices=["on", "off"], default="off")
    parser.add_argument("--video", choices=["on","off"], default="off")
    parser.add_argument("--video-path", default="rings_v050.mp4")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--replay", help="Replay schema-50 state/motor/chemical recording without running the SNN")
    parser.add_argument("--audit-path", default=None, help="Run mode: export full circuit/assumptions JSON")
    parser.add_argument("--record-path", default=None, help="Run-mode .npz output; default recordings/unique name")
    parser.add_argument("--debug-log", type=str, default=None, help="Write per-step diagnostic CSV log to this path")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--danger-speed", type=float, default=0.0)
    args = parser.parse_args()
    print(f"RINGS {AGENT_VERSION} | Python {sys.version.split()[0]} | {sys.executable}", flush=True)

    if args.self_test:
        if args.params or args.mode != "run" or args.ablation != "full":
            parser.error("--self-test tests fixed default fixtures; omit --params, --mode tune and --ablation")
        report=run_circuit_checks()
        if args.test_report:
            import hashlib
            from importlib.metadata import version
            payload=dict(agent_version=AGENT_VERSION,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         params=asdict(RingsParams()),python=sys.version,
                         runtime={n:version(n) for n in ("brian2","minigrid","gymnasium","numpy")},results=report)
            _save_tune_payload(args.test_report,payload)
            print("Test report saved: "+str(Path(args.test_report).resolve()))
        return

    if args.test_report:
        parser.error("--test-report requires --self-test")
    if args.replay:
        replay_recording(args.replay,args.video_path,args.video_fps)
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
            ablation=args.ablation,
        )
    else:
        params = load_params(args.params) if args.params else RingsParams()
        visualize = args.visual == "on"
        result = run_episode(seed=args.seed, params=params, visualize=visualize, max_steps=args.max_steps, status=not visualize, record=record, debug_log=args.debug_log, grid_size=args.grid_size, danger_speed=args.danger_speed, record_path=args.record_path, video=args.video == "on", video_path=args.video_path, video_fps=args.video_fps, audit_path=args.audit_path, ablation=args.ablation)
        print(result)


if __name__ == "__main__":
    main()
