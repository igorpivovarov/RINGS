#!/usr/bin/env python3
"""RINGS Model C, baseline version 1.3. Python >=3.8.
Standalone minimal feedforward spiking controller; same plant as v0.51.2.
Requires numpy, brian2, gymnasium, minigrid (matplotlib for --visual on).
"""
import json, random, csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List
import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv
from brian2 import NeuronGroup, Synapses, SpikeMonitor, StateMonitor, Network, defaultclock, ms, mV, prefs, start_scope, linked_var
prefs.codegen.target='numpy'
MODEL='C'
AGENT_VERSION='RC-1.3'
CHEM_NAMES=('foraging','defense','fatigue','curiosity')
ENV_SOURCE_SHA256='99a5da2b23029e0b18c44276c03de217cb8f469280f331ed8eb4c6613ceab210'

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

@dataclass
class RingsParams:
    sensor_range: int = 3
    snn_step_ms: float = 30.0
    dt_ms: float = 0.1
    membrane_tau_ms: float = 3.0
    synapse_tau_ms: float = 2.0
    satiety_midpoint: float = 0.95
    receptor_drive_mv: float = 50.0
    receptor_half: float = 0.12
    food_weight: float = 90.0
    threat_weight: float = 100.0
    wall_weight: float = 100.0
    hunger_weight: float = 50.0
    satiety_weight: float = 20.0
    rest_bias_mv: float = 17.0
    foraging_drive_mv: float = 15.0
    defense_drive_mv: float = 65.0
    fatigue_drive_mv: float = 24.0
    curiosity_drive_mv: float = 20.0
    receptor_kd: float = 0.25
    foraging_tau_ms: float = 450.0
    defense_tau_ms: float = 300.0
    fatigue_tau_ms: float = 600.0
    curiosity_tau_ms: float = 600.0
    foraging_release: float = 0.12
    defense_release: float = 0.14
    fatigue_release: float = 0.05
    curiosity_release: float = 0.14

    def __post_init__(self):
        for k,v in asdict(self).items():
            if not np.isfinite(v) or v <= 0: raise ValueError(k+' must be finite and positive')
        if self.sensor_range != 3: raise ValueError('Fixed sensor range is 3')
        if not 0 < self.satiety_midpoint < 1: raise ValueError('Invalid satiety midpoint')
        if self.dt_ms > .1 or not np.isclose(self.snn_step_ms/self.dt_ms,round(self.snn_step_ms/self.dt_ms)):
            raise ValueError('Invalid integration step')
        if max(self.membrane_tau_ms,self.synapse_tau_ms)>self.snn_step_ms/5:
            raise ValueError('Fast neural dynamics must be shorter than a control window')
        for n in CHEM_NAMES:
            if getattr(self,n+'_release') > 1 or getattr(self,n+'_tau_ms') <= self.snn_step_ms:
                raise ValueError('Invalid chemical time scale/release')


class RingsSNNController:
    """Feedforward LIF circuit; no recurrent synapses or adaptive neurons.

    R has only fast membrane/synapse transients, not mathematically zero memory.
    C adds four spike-secreted, slowly decaying concentrations. No per-step reset.
    """
    input_names=('low_energy','food_left','food_right','food_ahead',
                 'danger_left','danger_right','danger_ahead','wall_ahead',
                 'danger_contact','satiety','effort','turn_feedback')
    motor_names=('MotorLeft','MotorRight','MotorForward','MotorRest')

    def __init__(self,params,seed=1,record=False,ablation='full'):
        if ablation not in ('full','no-chemistry'):
            raise ValueError('R/C have no state recurrence; only full or no-chemistry applies')
        self.params=params; self.record=record
        self.chemistry=MODEL=='C' and ablation=='full'
        self.action_rng=random.Random(int(seed)+3001)
        start_scope(); defaultclock.dt=params.dt_ms*ms
        eq='''dv/dt=(-65*mV-v+drive+j)/tau_v : volt (unless refractory)
              dj/dt=-j/tau_s : volt
              drive : volt'''
        ns=dict(tau_v=params.membrane_tau_ms*ms,tau_s=params.synapse_tau_ms*ms)
        self.receptors=NeuronGroup(12,eq,threshold='v>-50*mV',reset='v=-65*mV',
                                  refractory=2*ms,method='euler',namespace=ns)
        motor_eq=eq.replace('drive : volt','''drive = bias + chemical : volt
            bias : volt
            chemical : volt''')
        if self.chemistry:
            motor_eq=motor_eq.replace('            chemical : volt','''chemical = wf*cf/(kd+cf)+wd*cd/(kd+cd)+wt*ct/(kd+ct)+wc*cc/(kd+cc) : volt
                cf : 1 (linked)
                cd : 1 (linked)
                ct : 1 (linked)
                cc : 1 (linked)
                wf : volt (constant)
                wd : volt (constant)
                wt : volt (constant)
                wc : volt (constant)''')
        self.motor=NeuronGroup(4,motor_eq,threshold='v>-50*mV',reset='v=-65*mV',
                             refractory=2*ms,method='euler',namespace=dict(ns,kd=params.receptor_kd))
        self.receptors.v=-65*mV;self.motor.v=-65*mV
        self.motor.bias=[0,0,0,params.rest_bias_mv]*mV
        # Signed feedforward current synapses. No motor-to-motor connections.
        w=np.zeros((12,4))
        w[0]=params.hunger_weight*np.array([.25,.25,1,-.6])
        for src,target in [(1,0),(2,1),(3,2)]:
            w[src,:3]=-params.food_weight*.35;w[src,target]=params.food_weight
        w[4]=params.threat_weight*np.array([-1,1,.65,-1.5])
        w[5]=params.threat_weight*np.array([1,-1,.65,-1.5])
        w[6]=params.threat_weight*np.array([1,1,-1.5,-1.5])
        w[7]=params.wall_weight*np.array([1,1,-2,0])
        w[8]=params.threat_weight*np.array([.4,.4,1,-1.5])
        w[9]=params.satiety_weight*np.array([-.5,-.5,-.5,1])
        self.connections=[];self.edge_catalog=[]
        self._connect(self.receptors,self.motor,w,'sensory_to_motor')
        self.modulators=None
        objects=[self.receptors,self.motor]
        if self.chemistry:
            self.modulators=NeuronGroup(4,eq+'''
                dc/dt=-c/tau_c : 1
                tau_c : second (constant)
                release : 1 (constant)''',threshold='v>-50*mV',
                reset='v=-65*mV; c+=release*(1-c)',refractory=15*ms,
                method='euler',namespace=ns)
            self.modulators.v=-65*mV;self.modulators.drive=[0,0,0,18]*mV
            self.modulators.tau_c=[getattr(params,n+'_tau_ms') for n in CHEM_NAMES]*ms
            self.modulators.release=[getattr(params,n+'_release') for n in CHEM_NAMES]
            for k,key in enumerate(('cf','cd','ct','cc')):
                setattr(self.motor,key,linked_var(self.modulators,'c',index=np.full(4,k)))
            self.motor.wf=params.foraging_drive_mv*np.array([.35,.35,1,-.5])*mV
            self.motor.wd=params.defense_drive_mv*np.array([.2,.2,1,-1])*mV
            self.motor.wt=params.fatigue_drive_mv*np.array([-.5,-.5,-1,1])*mV
            self.motor.wc=params.curiosity_drive_mv*np.array([.65,.65,1,-.5])*mV
            u=np.zeros((12,4));u[0,0]=80;u[9,0]=-80
            u[[4,5,6,8],1]=100;u[10,2]=65
            u[[1,2,3,4,5,6,8],3]=-100
            self._connect(self.receptors,self.modulators,u,'sensory_to_secretor')
            objects.append(self.modulators)
        self.spikes=SpikeMonitor(self.motor,record=record);objects.append(self.spikes)
        self.trace=StateMonitor(self.motor,['v','j'],record=True,dt=1*ms) if record else None
        self.chem_trace=StateMonitor(self.modulators,'c',record=True,dt=1*ms) if record and self.chemistry else None
        objects += [x for x in (self.trace,self.chem_trace) if x is not None]
        self.net=Network(*objects,*self.connections)
        self.last_decision='silent';self.last_inputs={};self.recent_counts=np.zeros(4)

    def enable_visual_monitoring(self):
        if hasattr(self,'visual_monitors'): return
        self.visual_monitors={'receptors':SpikeMonitor(self.receptors,record=False)}
        if self.chemistry: self.visual_monitors['secretors']=SpikeMonitor(self.modulators,record=False)
        self.visual_counts={n:np.zeros(len(m.count),dtype=int) for n,m in self.visual_monitors.items()}
        self.net.add(*self.visual_monitors.values())

    def _connect(self,source,target,weights,name):
        i,j=np.nonzero(weights)
        syn=Synapses(source,target,'w : volt',on_pre='j_post+=w')
        syn.connect(i=i,j=j);syn.w=weights[i,j]*mV
        self.connections.append(syn)
        self.edge_catalog += [dict(family=name,source=int(a),target=int(b),weight_mv=float(weights[a,b])) for a,b in zip(i,j)]

    def encode_observation(self,obs,env):
        e,fl,fa,fr,dl,da,dr=obs['rstate']
        sat=max(0,(float(e)-self.params.satiety_midpoint)/(1-self.params.satiety_midpoint))
        return np.clip([1-float(e),fl,fr,fa,dl,dr,da,env._wall_ahead(),
                        env.danger_contact_signal,sat,env.last_effort,env.last_turn],0,1)

    def step(self,obs,env): return self.advance(self.encode_observation(obs,env))

    def advance(self,signals):
        s=np.asarray(signals,dtype=float)
        if s.shape!=(12,) or not np.isfinite(s).all() or np.any((s<0)|(s>1)):
            raise ValueError('Expected 12 finite receptor signals in [0,1]')
        self.last_inputs=dict(zip(self.input_names,map(float,s)))
        self.receptors.drive=self.params.receptor_drive_mv*s/(self.params.receptor_half+s)*mV
        before=np.array(self.spikes.count[:])
        visual_before={n:np.asarray(m.count[:]).copy() for n,m in getattr(self,'visual_monitors',{}).items()}
        self.net.run(self.params.snn_step_ms*ms,namespace={})
        for n,m in getattr(self,'visual_monitors',{}).items():
            self.visual_counts[n]=np.asarray(m.count[:])-visual_before[n]
        self.recent_counts=np.array(self.spikes.count[:])-before
        scores=self.recent_counts
        if scores.max()==0: action=6;self.last_decision='silent'
        else:
            k=self.action_rng.choice(np.flatnonzero(scores==scores.max()).tolist())
            action=(0,1,2,6)[k];self.last_decision='active_rest' if k==3 else 'motor'
        values=dict(zip(self.motor_names,map(int,scores)))
        values.update(zip(CHEM_NAMES,map(float,self.modulators.c[:] if self.chemistry else np.zeros(4))))
        return action,values

    def audit(self):
        return dict(version=AGENT_VERSION,model=MODEL,chemical_effect=self.chemistry,
                    neurons=20 if self.chemistry else 16,receptors=12,motors=4,
                    secretors=4 if self.chemistry else 0,recurrent_synapses=0,
                    adaptation=False,neural_reset_between_steps=False,
                    fast_transient_ms=self.params.membrane_tau_ms,
                    edges=self.edge_catalog,params=asdict(self.params))


class RingsVisualizer:
    """Bounded live diagnostics; no full-episode recording is required."""
    def __init__(self,controller,history=20):
        import matplotlib.pyplot as plt
        from collections import deque
        self.plt=plt;self.c=controller;self.history=deque(maxlen=history);self.window=history
        controller.enable_visual_monitoring()
        self.fig=plt.figure(figsize=(21,11),constrained_layout=True)
        gs=self.fig.add_gridspec(3,4,width_ratios=[1.25,1,1,2.1],height_ratios=[1,1,1])
        self.ax_env=self.fig.add_subplot(gs[:2,0])
        self.ax_sensors=self.fig.add_subplot(gs[2,0])
        self.ax_voltage=self.fig.add_subplot(gs[0,1:3])
        self.ax_motor=self.fig.add_subplot(gs[1,1])
        self.ax_chem=self.fig.add_subplot(gs[1,2])
        self.ax_spikes=self.fig.add_subplot(gs[2,1:3])
        self.ax_history=self.fig.add_subplot(gs[:,3])
        self.labels=['Hunger','Food L','Food R','Food F','Danger L','Danger R','Danger F',
                     'Wall','Contact','Satiety','Effort','Turn','M Left','M Right','M Fwd','M Rest']
        self.colors=['#64748b']*12+['#2563eb']*4
        if controller.chemistry:
            self.labels+=['Forage','Defense','Fatigue','Curiosity'];self.colors+=['#d97706']*4
        self.was_interactive=plt.isinteractive()
        plt.ioff()
        self.ready=False
        plt.show(block=False)

    def update(self,frame,values,step,info,action):
        if not self.plt.fignum_exists(self.fig.number): return False
        c=self.c;p=c.params
        row=dict(c.last_inputs,**values,step=step,energy=info['energy_after'])
        self.history.append(row)
        action_name={0:'LEFT',1:'RIGHT',2:'FORWARD',6:'REST'}[action]
        self.fig.suptitle('RINGS %s | Model %s | step %d | action %s (%s) | energy %.3f -> %.3f | food %d | contacts %d'%(
            AGENT_VERSION,MODEL,step,action_name,c.last_decision,info['energy_before'],info['energy_after'],
            info['food_collected'],info['danger_contacts']),fontsize=13)
        if self.ready:
            self._refresh(frame,values,step,action)
            return True
        ax=self.ax_env;ax.clear();ax.imshow(frame);ax.axis('off')
        ax.set_title('Scene BEFORE action\nInputs correspond to this scene',fontsize=10)
        ax=self.ax_sensors;ax.clear()
        ax.bar(range(12),[c.last_inputs[n] for n in c.input_names],color=self.colors[:12])
        ax.set_xticks(range(12));ax.set_xticklabels(self.labels[:12],rotation=65,ha='right',fontsize=8)
        ax.set_ylim(0,1.05);ax.set_title('Receptor stimuli [0, 1]');ax.grid(axis='y',alpha=.2)
        voltage=np.concatenate([np.asarray(c.receptors.v/mV),np.asarray(c.motor.v/mV)]+
                               ([np.asarray(c.modulators.v/mV)] if c.chemistry else []))
        ax=self.ax_voltage;ax.clear()
        ax.scatter(range(len(voltage)),voltage,c=self.colors,s=55,zorder=3)
        ax.axhline(-50,color='#dc2626',ls='--',lw=1,label='Spike threshold (-50 mV)')
        ax.axhline(-65,color='gray',ls=':',lw=1,label='Reset / rest (-65 mV)')
        ax.set_xticks(range(len(voltage)));ax.set_xticklabels(self.labels,rotation=65,ha='right',fontsize=8)
        ax.set_ylim(min(-85,float(voltage.min())-5),max(-45,float(voltage.max())+5))
        ax.set_ylabel('mV');ax.set_title('All neuron voltages at END of window (not peak voltages)')
        ax.legend(loc='lower left',fontsize=7,ncol=2);ax.grid(axis='y',alpha=.2)
        counts=np.concatenate([c.visual_counts['receptors'],c.recent_counts]+
                              ([c.visual_counts['secretors']] if c.chemistry else []))
        ax=self.ax_spikes;ax.clear();ax.bar(range(len(counts)),counts,color=self.colors)
        ax.set_xticks(range(len(counts)));ax.set_xticklabels(self.labels,rotation=65,ha='right',fontsize=8)
        ax.set_ylim(0,np.ceil(p.snn_step_ms/2)+1);ax.set_ylabel('Spikes / neuron / window')
        ax.set_title('Feedforward circuit: receptors -> motors'+(' + receptors -> secretors' if c.chemistry else ''))
        ax.grid(axis='y',alpha=.2)
        ax=self.ax_motor;ax.clear()
        bars=ax.bar(range(4),c.recent_counts,color='#2563eb')
        selected=(0,1,2,6).index(action)
        if c.last_decision!='silent':bars[selected].set_edgecolor('#dc2626');bars[selected].set_linewidth(3)
        for k,count in enumerate(c.recent_counts):ax.text(k,float(count)+.15,str(int(count)),ha='center',fontsize=9)
        ax.set_xticks(range(4));ax.set_xticklabels(['Left','Right','Forward','Rest'],rotation=20,fontsize=9)
        ax.set_ylim(0,np.ceil(p.snn_step_ms/2)+1);ax.set_title('Motor spikes | red outline = action',fontsize=10)
        ax=self.ax_chem;ax.clear()
        if c.chemistry:
            chem=[values[n] for n in CHEM_NAMES]
            ax.bar(range(4),chem,color='#d97706')
            for k,v in enumerate(chem):ax.text(k,v+.025,'%.3f'%v,ha='center',fontsize=8)
            ax.set_xticks(range(4));ax.set_xticklabels(['Foraging','Defense','Fatigue','Curiosity'],rotation=35,ha='right',fontsize=8)
            ax.set_ylim(0,1.1);ax.set_title('Neuromodulator concentrations',fontsize=10);ax.set_ylabel('Normalized c [0, 1]')
        else:
            ax.axis('off');ax.text(.5,.5,'No chemical block\n'+('Model R' if MODEL=='R' else 'C: no-chemistry ablation'),
                                  ha='center',va='center',transform=ax.transAxes,fontsize=12,color='#64748b')
        ax=self.ax_history;ax.clear()
        lanes=['energy',*c.input_names,*c.motor_names]+(list(CHEM_NAMES) if c.chemistry else [])
        labels=(['Energy',*self.labels[:12],'Motor Left','Motor Right','Motor Forward','Motor Rest']+
                ([] if not c.chemistry else ['Foraging','Defense','Fatigue','Curiosity']))
        xs=[h['step']-step for h in self.history]
        for k,n in enumerate(lanes):
            base=len(lanes)-1-k;motor=n in c.motor_names
            scale=np.ceil(p.snn_step_ms/2) if motor else 1
            vals=np.asarray([h[n] for h in self.history])/scale
            color='#16a34a' if n=='energy' else '#2563eb' if motor else '#d97706' if n in CHEM_NAMES else '#64748b'
            ax.plot(xs,base+.8*vals,color=color,lw=1.5,marker='.' if len(xs)==1 else None)
            ax.axhline(base,color='gray',alpha=.2,lw=.5)
        ax.set_yticks(range(len(lanes)));ax.set_yticklabels(list(reversed(labels)),fontsize=8)
        ax.set_xlim(1-self.window,0);ax.set_ylim(-.2,len(lanes))
        ax.set_xlabel('Steps relative to current step (0 = now)');ax.set_title('Last %d steps | shared time axis\nInputs [0,1]; motor spikes / %.0f; concentrations [0,1]'%(self.window,np.ceil(p.snn_step_ms/2)),fontsize=10)
        self._capture_artists()
        self._paint()
        return True

    def _capture_artists(self):
        self.image=self.ax_env.images[0]
        self.sensor_bars=list(self.ax_sensors.patches)
        self.voltage_points=self.ax_voltage.collections[0]
        self.spike_bars=list(self.ax_spikes.patches)
        self.motor_bars=list(self.ax_motor.patches)
        self.motor_texts=list(self.ax_motor.texts)
        self.chem_bars=list(self.ax_chem.patches) if self.c.chemistry else []
        self.chem_texts=list(self.ax_chem.texts) if self.c.chemistry else []
        self.history_lines=list(self.ax_history.lines)[::2]
        self.lanes=['energy',*self.c.input_names,*self.c.motor_names]+(list(CHEM_NAMES) if self.c.chemistry else [])
        self.artists=[self.image,*self.sensor_bars,self.voltage_points,*self.spike_bars,
                      *self.motor_bars,*self.motor_texts,*self.chem_bars,*self.chem_texts,
                      *self.history_lines,self.fig._suptitle]
        self.use_blit=bool(self.fig.canvas.supports_blit)
        for artist in self.artists: artist.set_animated(self.use_blit)
        # Solve the expensive layout once, then retain the existing axes/artists.
        self.fig.canvas.draw()
        if hasattr(self.fig,'set_layout_engine'): self.fig.set_layout_engine(None)
        else: self.fig.set_constrained_layout(False)
        self.background=None
        self.layout_dirty=False
        self.fig.canvas.mpl_connect('draw_event',self._on_draw)
        self.fig.canvas.mpl_connect('resize_event',self._on_resize)
        self.ready=True

    def _on_resize(self,event):
        self.background=None
        self.layout_dirty=True

    def _on_draw(self,event):
        if self.use_blit and not event.canvas.is_saving():
            self.background=event.canvas.copy_from_bbox(self.fig.bbox)
            for artist in self.artists: self.fig.draw_artist(artist)

    def _refresh(self,frame,values,step,action):
        c=self.c
        self.image.set_data(frame)
        for bar,n in zip(self.sensor_bars,c.input_names):bar.set_height(c.last_inputs[n])
        voltage=np.concatenate([np.asarray(c.receptors.v/mV),np.asarray(c.motor.v/mV)]+
                               ([np.asarray(c.modulators.v/mV)] if c.chemistry else []))
        self.voltage_points.set_offsets(np.column_stack([np.arange(len(voltage)),voltage]))
        lo,hi=self.ax_voltage.get_ylim()
        if voltage.min()<lo or voltage.max()>hi:
            self.ax_voltage.set_ylim(min(lo,float(voltage.min())-5),max(hi,float(voltage.max())+5))
            self.background=None
        counts=np.concatenate([c.visual_counts['receptors'],c.recent_counts]+
                              ([c.visual_counts['secretors']] if c.chemistry else []))
        for bar,v in zip(self.spike_bars,counts):bar.set_height(v)
        selected=(0,1,2,6).index(action)
        for k,(bar,txt,v) in enumerate(zip(self.motor_bars,self.motor_texts,c.recent_counts)):
            bar.set_height(v);bar.set_edgecolor('#dc2626' if k==selected and c.last_decision!='silent' else 'none')
            bar.set_linewidth(3);txt.set_text(str(int(v)));txt.set_position((k,float(v)+.15))
        for k,(bar,txt,n) in enumerate(zip(self.chem_bars,self.chem_texts,CHEM_NAMES)):
            v=values[n];bar.set_height(v);txt.set_text('%.3f'%v);txt.set_position((k,v+.025))
        xs=[h['step']-step for h in self.history]
        for k,(line,n) in enumerate(zip(self.history_lines,self.lanes)):
            scale=np.ceil(c.params.snn_step_ms/2) if n in c.motor_names else 1
            line.set_data(xs,len(self.lanes)-1-k+.8*np.asarray([h[n] for h in self.history])/scale)
            line.set_marker('.' if len(xs)==1 else '')
        self._paint()

    def _paint(self):
        canvas=self.fig.canvas
        if self.layout_dirty:
            if hasattr(self.fig,'set_layout_engine'):self.fig.set_layout_engine('constrained')
            else:self.fig.set_constrained_layout(True)
            canvas.draw()
            if hasattr(self.fig,'set_layout_engine'):self.fig.set_layout_engine(None)
            else:self.fig.set_constrained_layout(False)
            self.layout_dirty=False
        if self.use_blit:
            if self.background is None:canvas.draw()
            if self.background is not None:
                canvas.restore_region(self.background)
                for artist in self.artists:self.fig.draw_artist(artist)
                canvas.blit(self.fig.bbox)
        else:
            canvas.draw()  # Synchronous fallback: no deferred idle redraw.
        # Service the GUI explicitly. pyplot.pause also calls show on every
        # frame on some backends; neither show nor an idle repaint is needed here.
        canvas.flush_events()

    def close(self):
        self.plt.close(self.fig)
        if self.was_interactive:self.plt.ion()


@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    final_energy: float
    food_collected: int
    danger_contacts: int
    rest_steps: int
    active_rest_steps: int
    silent_steps: int
    wall_bumps: int
    unique_cells: int
    forward_steps: int
    turn_steps: int
    moved_steps: int


def run_episode(seed,params=None,visualize=False,max_steps=1000,status=False,record=False,
                debug_log=None,grid_size=15,danger_speed=0.0,ablation='full',
                record_path=None,audit_path=None,progress_callback=None,visual_history=20):
    if max_steps<1: raise ValueError('max_steps must be positive')
    if visual_history<2 or int(visual_history)!=visual_history: raise ValueError('visual_history must be an integer >=2')
    if record_path and not record: raise ValueError('record_path requires --record on')
    p=params or RingsParams()
    env=RingsHomeostasisEnv(size=grid_size,max_steps=max_steps,sensor_range=p.sensor_range,danger_speed=danger_speed)
    obs,info=env.reset(seed=seed)
    c=RingsSNNController(p,seed=seed,record=record,ablation=ablation)
    if audit_path: save_json(audit_path,c.audit())
    rows=[];total=0.;rest=active=silent=bumps=forward=turns=moved=0
    visited={tuple(env.agent_pos)}
    vis=RingsVisualizer(c,history=visual_history) if visualize else None
    try:
        for step in range(1,max_steps+1):
            pos=tuple(env.agent_pos);energy=env.energy
            frame=env.render().copy() if vis is not None else None
            action,values=c.step(obs,env)
            rest+=int(action==6);active+=int(c.last_decision=='active_rest');silent+=int(c.last_decision=='silent')
            forward+=int(action==2);turns+=int(action in (0,1));bumps+=int(action==2 and env._wall_ahead())
            obs,reward,terminated,truncated,info=env.step(action)
            total+=float(reward);moved+=int(tuple(env.agent_pos)!=pos);visited.add(tuple(env.agent_pos))
            if record or debug_log:
                rows.append(dict(step=step,action=action,energy_before=float(energy),energy_after=float(env.energy),
                                 food_collected=env.food_collected,danger_contacts=env.danger_contacts,
                                 **c.last_inputs,**values))
            if vis is not None:
                opened=vis.update(frame,values,step,dict(energy_before=energy,energy_after=float(env.energy),
                    food_collected=env.food_collected,danger_contacts=env.danger_contacts),action)
                if not opened: vis=None  # Closing diagnostics does not change the episode.
            if status: print('\rModel %s step %d energy %.3f food %d danger %d'%(MODEL,step,env.energy,env.food_collected,env.danger_contacts),end='',flush=True)
            if progress_callback: progress_callback(step,max_steps)
            if terminated or truncated: break
    finally:
        env.close()
        if vis is not None: vis.close()
        if status: print()
    if debug_log:
        Path(debug_log).parent.mkdir(parents=True,exist_ok=True)
        with open(debug_log,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if record:
        path=Path(record_path or ('rings_%s_seed%d.npz'%(MODEL,seed)))
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,metadata_json=json.dumps(c.audit()),rows_json=json.dumps(rows),
                            t_ms=np.asarray(c.trace.t/ms),motor_v_mv=np.asarray(c.trace.v/mV),
                            spike_t_ms=np.asarray(c.spikes.t/ms),spike_i=np.asarray(c.spikes.i),
                            chemistry=np.asarray(c.chem_trace.c) if c.chem_trace is not None else np.empty((0,0)))
    return EpisodeResult(step,total,float(env.energy),env.food_collected,env.danger_contacts,
                         rest,active,silent,bumps,len(visited),forward,turns,moved)


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def load_params(path):
    d=json.loads(Path(path).read_text())
    if 'model' in d and d['model']!=MODEL: raise ValueError('Parameter file belongs to another model')
    if 'best_params' in d: d=d['best_params']
    elif 'params' in d: d=d['params']
    return RingsParams(**d)


def sample_params(rng,ablation='full'):
    d=asdict(RingsParams())
    names=['food_weight','threat_weight','wall_weight','hunger_weight','satiety_weight','rest_bias_mv']
    if MODEL=='C' and ablation=='full': names+=['foraging_drive_mv','defense_drive_mv','fatigue_drive_mv','curiosity_drive_mv']
    for n in names: d[n]*=rng.uniform(.65,1.4)
    return RingsParams(**d)


def evaluate_params(params,seeds,max_steps=1000,grid_size=15,danger_speed=0.0,ablation='full'):
    results=[asdict(run_episode(seed=s,params=params,max_steps=max_steps,grid_size=grid_size,
                               danger_speed=danger_speed,ablation=ablation)) for s in seeds]
    return dict(mean_steps=float(np.mean([x['steps'] for x in results])),
                mean_food=float(np.mean([x['food_collected'] for x in results])),
                mean_danger=float(np.mean([x['danger_contacts'] for x in results])),episodes=results)


def _trial(job):
    trial,d,settings=job
    return dict(trial=trial,params=d,metrics=evaluate_params(RingsParams(**d),**settings))


def tune(args):
    import concurrent.futures as cf
    import time
    if not np.isfinite(args.progress_interval) or args.progress_interval <= 0:
        raise ValueError("progress-interval must be finite and positive")
    if args.trials<1 or args.workers<1: raise ValueError('Positive trials/workers required')
    rng=random.Random(args.search_seed)
    settings=dict(seeds=args.seeds,max_steps=args.max_steps,grid_size=args.grid_size,
                  danger_speed=args.danger_speed,ablation=args.ablation)
    jobs=[(i+1,asdict(RingsParams() if i==0 else sample_params(rng,args.ablation)),settings) for i in range(args.trials)]
    payload=dict(version=AGENT_VERSION,model=MODEL,selection='mean_steps; ties: lower trial',
                 tune_config=dict(search_seed=args.search_seed,settings=settings),all_results=[])
    started=time.monotonic()
    def duration(seconds):
        seconds=max(0,int(round(seconds)))
        h,rem=divmod(seconds,3600);m,s=divmod(rem,60)
        return '%02d:%02d:%02d'%(h,m,s)
    def progress(label='Status'):
        elapsed=time.monotonic()-started
        done=len(payload['all_results'])
        # Wall-time throughput includes worker parallelism. No estimate before
        # a complete candidate; later estimates fluctuate with episode lengths.
        remaining=elapsed*(args.trials-done)/done if done else None
        eta=duration(remaining) if remaining is not None else 'estimating (no completed candidates)'
        best=('trial %d mean_steps %.2f food %.2f danger %.2f'%(
            payload['best_trial'],payload['best_metrics']['mean_steps'],
            payload['best_metrics']['mean_food'],payload['best_metrics']['mean_danger'])) if done else 'none yet'
        print('%s | completed %d/%d (%.1f%%) | elapsed %s | Estimated time remaining %s | best %s'%(
            label,done,args.trials,100.0*done/args.trials,duration(elapsed),eta,best),flush=True)
    def retain(r):
        payload['all_results'].append(r)
        ranked=sorted(payload['all_results'],key=lambda x:(-x['metrics']['mean_steps'],x['trial']))
        payload['best_trial']=ranked[0]['trial'];payload['best_params']=ranked[0]['params']
        payload['best_metrics']=ranked[0]['metrics'];payload['reserve_candidates']=ranked[1:3]
        save_json(args.output,payload)
        metrics=r['metrics']
        progress('Finished trial %d: mean_steps %.2f food %.2f danger %.2f'%(
            r['trial'],metrics['mean_steps'],metrics['mean_food'],metrics['mean_danger']))
    progress('Starting with %d workers'%min(args.workers,args.trials))
    # A separate process also for workers=1 keeps the parent available to
    # print heartbeats while a long candidate is still being simulated.
    with cf.ProcessPoolExecutor(max_workers=min(args.workers,args.trials)) as pool:
        pending={pool.submit(_trial,j) for j in jobs}
        next_update=time.monotonic()+args.progress_interval
        while pending:
            done,pending=cf.wait(pending,timeout=max(0,next_update-time.monotonic()),
                                 return_when=cf.FIRST_COMPLETED)
            for future in done: retain(future.result())
            if time.monotonic()>=next_update:
                if pending: progress()
                next_update=time.monotonic()+args.progress_interval


def main():
    import argparse
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--mode',choices=['run','tune','evaluate'],default='run')
    a.add_argument('--params');a.add_argument('--seed',type=int,default=7)
    a.add_argument('--seeds',type=int,nargs='+',default=list(range(1,13)))
    a.add_argument('--max-steps',type=int,default=1000);a.add_argument('--grid-size',type=int,default=15)
    a.add_argument('--danger-speed',type=float,default=0.)
    a.add_argument('--visual',choices=['on','off'],default='off')
    a.add_argument('--visual-history',type=int,default=20,help='Live history length in environment steps (default: 20)')
    a.add_argument('--record',choices=['on','off'],default='off');a.add_argument('--record-path')
    a.add_argument('--debug-log');a.add_argument('--audit-path')
    a.add_argument('--ablation',choices=['full','no-chemistry'],default='full')
    a.add_argument('--trials',type=int,default=100);a.add_argument('--workers',type=int,default=1)
    a.add_argument('--progress-interval',type=float,default=30.0,help='Tune status interval in seconds (default: 30)')
    a.add_argument('--search-seed',type=int,default=20261012)
    a.add_argument('--output',default='rings_%s_results.json'%MODEL)
    args=a.parse_args();print('RINGS '+AGENT_VERSION+' Model '+MODEL,flush=True)
    if args.mode=='tune':
        if args.params: a.error('tune uses defaults plus random search; --params is for run/evaluate')
        tune(args);return
    p=load_params(args.params) if args.params else RingsParams()
    if args.mode=='evaluate':
        result=evaluate_params(p,args.seeds,args.max_steps,args.grid_size,args.danger_speed,args.ablation)
        save_json(args.output,dict(model=MODEL,params=asdict(p),metrics=result));print(json.dumps(result,indent=2));return
    result=run_episode(args.seed,p,visualize=args.visual=='on',max_steps=args.max_steps,
        status=args.visual=='off',record=args.record=='on',debug_log=args.debug_log,
        grid_size=args.grid_size,danger_speed=args.danger_speed,ablation=args.ablation,
        record_path=args.record_path,audit_path=args.audit_path,visual_history=args.visual_history)
    print(json.dumps(asdict(result),indent=2))


if __name__=='__main__': main()
