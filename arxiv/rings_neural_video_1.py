"""Render measured RINGS v40/v41 NPZ traces as a synchronized 2D MP4.

No Brian2 simulation or model import. Dependencies: numpy, pillow, imageio,
imageio-ffmpeg. Python 3.8-compatible. Layout is schematic, not a connectome.
Each environment snapshot precedes its action; neural time advances within it.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = '#0b1220'
PANEL = '#131e30'
TEXT = '#edf2fa'
MUTED = '#a6b5cb'
LINE = '#29394f'
GOLD = '#ffe49c'
MOD_COLORS = ['#69d5ae', '#f38585', '#bdabf2', '#68c7ee']
ACTIONS = {0: 'поворот налево', 1: 'поворот направо', 2: 'вперёд', 6: 'отдых / нет команды'}


def rounded_panel(draw, bounds, radius, fill):
    """Filled rounded rectangle, including Pillow without rounded_rectangle."""
    native = getattr(draw, 'rounded_rectangle', None)
    if callable(native):
        native(bounds, radius=radius, fill=fill)
        return
    x0, y0, x1, y1 = bounds
    r = min(radius, (x1-x0)/2, (y1-y0)/2)
    if r <= 0:
        draw.rectangle(bounds, fill=fill)
        return
    draw.rectangle((x0+r, y0, x1-r, y1), fill=fill)
    draw.rectangle((x0, y0+r, x1, y1-r), fill=fill)
    for x in (x0, x1-2*r):
        for y in (y0, y1-2*r):
            draw.ellipse((x, y, x+2*r, y+2*r), fill=fill)


def font(size, bold=False, path=None):
    candidates = [path] if path else []
    candidates += ['DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
                   '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
                   else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                   'C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    raise ValueError('No Cyrillic-capable font found. Supply --font /path/to/font.ttf')


def voltage_color(value):
    # Fixed visualization scale, not a clamp applied to the recorded signal.
    stops = [(-80, (31, 49, 87)), (-65, (65, 130, 199)),
             (-50, (226, 127, 160)), (-40, (255, 208, 135))]
    v = float(np.clip(value, -80, -40))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            return tuple(int(x + (y-x)*(v-a)/(b-a)) for x, y in zip(ca, cb))
    return stops[-1][1]


class Recording:
    def __init__(self, path):
        required = ['metadata_json', 'step_columns', 'steps', 'scene_grids',
                    'ring_names', 'ring_bounds', 'mod_names', 'state_t_ms',
                    'state_v_mv', 'chemical_t_ms', 'chemical_c', 'mod_v_mv',
                    'ring_spike_t_ms', 'ring_spike_i', 'mod_spike_t_ms', 'mod_spike_i']
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in required if k not in z]
            if missing:
                raise ValueError('Missing recorded arrays: ' + ', '.join(missing) +
                                 '. Use v40/v41 --record on; CSV alone is insufficient.')
            self.data = {k: z[k] for k in required}
        d = self.data
        self.meta = json.loads(str(d['metadata_json']))
        self.rows = d['steps']
        self.columns = list(d['step_columns'])
        self.names = list(d['ring_names'])
        self.mods = list(d['mod_names'])
        self.bounds = np.asarray(d['ring_bounds'], dtype=int)
        self.dt = float(self.meta['params']['snn_step_ms'])
        self.size = int(self.meta['grid_size'])
        self.view = max(3, 2*min(3, int(self.meta['params']['sensor_range']))-1)
        if self.rows.ndim != 2 or len(self.rows) == 0 or self.rows.shape[1] != len(self.columns):
            raise ValueError('Empty or malformed trajectory')
        if len(d['scene_grids']) != len(self.rows) or d['scene_grids'].shape[1:] != (self.size, self.size, 3):
            raise ValueError('Grid snapshots do not match the trajectory')
        if not np.isfinite(self.dt) or self.dt <= 0 or not np.all(np.isfinite(self.rows)):
            raise ValueError('Invalid timing or trajectory values')
        if set(self.names) != {'EscapeSwitch', 'MotorLeft', 'MotorRight', 'MotorForward', 'MotorRest'}:
            raise ValueError('This layout expects the five v40/v41 rings')
        if self.mods != ['ForagingMod', 'DefenseMod', 'FatigueMod', 'CuriosityMod']:
            raise ValueError('Unexpected modulator ordering')
        for k in ['step', 'x', 'y', 'direction', 'energy_before', 'action',
                  'food_left', 'food_ahead', 'food_right', 'danger_left', 'danger_ahead',
                  'danger_right', 'wall_ahead']:
            if k not in self.columns:
                raise ValueError('Missing trajectory column: ' + k)
        step_ids = self.rows[:, self.columns.index('step')]
        if not np.array_equal(step_ids, np.arange(len(self.rows))):
            raise ValueError('Expected contiguous, zero-based recorded step IDs')
        if self.bounds.shape != (5, 2) or np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError('Invalid ring bounds')
        n = int(self.bounds[:, 1].max())
        if not np.array_equal(np.concatenate([np.arange(a, b) for a, b in self.bounds]), np.arange(n)):
            raise ValueError('Ring bounds must partition the recorded neurons')
        for tk, vk, count in [('state_t_ms', 'state_v_mv', n), ('chemical_t_ms', 'chemical_c', 4),
                              ('chemical_t_ms', 'mod_v_mv', 4)]:
            t, values = d[tk], d[vk]
            if t.ndim != 1 or len(t) < 1 or values.shape != (count, len(t)):
                raise ValueError('Invalid trace shape: ' + vk)
            if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(values)):
                raise ValueError('Invalid trace values: ' + vk)
            tolerance = float(np.max(np.diff(t))) if len(t) > 1 else 1.0
            if t[0] > 1e-6 or t[-1]+tolerance+1e-5 < len(self.rows)*self.dt:
                raise ValueError('Trace does not cover all completed steps: ' + vk)
        for prefix, count in [('ring', n), ('mod', 4)]:
            t, ids = d[prefix+'_spike_t_ms'], d[prefix+'_spike_i']
            if t.ndim != 1 or ids.shape != t.shape or np.any(np.diff(t) < 0):
                raise ValueError('Invalid spike array: ' + prefix)
            if not np.all(np.isfinite(t)) or np.any(ids < 0) or np.any(ids >= count):
                raise ValueError('Invalid spike values: ' + prefix)
            if not np.all(ids == ids.astype(int)):
                raise ValueError('Non-integer neuron IDs')
        self.n = n

    def row(self, index):
        return dict(zip(self.columns, self.rows[index]))

    def counts(self, prefix, start, end):
        times = self.data[prefix+'_spike_t_ms']
        a, b = np.searchsorted(times, [start, end], side='left')
        ids = self.data[prefix+'_spike_i'][a:b].astype(int)
        return np.bincount(ids, minlength=self.n if prefix == 'ring' else 4)

    def state(self, end):
        # StateMonitor samples at the beginning of a dt. Never use next-step data.
        ti = max(0, np.searchsorted(self.data['state_t_ms'], end, side='left')-1)
        ci = max(0, np.searchsorted(self.data['chemical_t_ms'], end, side='left')-1)
        return (self.data['state_v_mv'][:, ti], self.data['chemical_c'][:, ci],
                self.data['mod_v_mv'][:, ci], ci)


class Renderer:
    width, height = 1600, 1000

    def __init__(self, recording, font_path=None):
        self.r = recording
        self.fonts = {s: font(s, path=font_path) for s in (15, 17, 19, 21, 24, 28, 34)}
        self.heading = font(34, bold=True, path=font_path)
        self.centres = {'EscapeSwitch': (1060, 242), 'MotorLeft': (755, 457),
                        'MotorForward': (1060, 457), 'MotorRight': (1365, 457),
                        'MotorRest': (1060, 670)}

    def text(self, draw, xy, value, size=19, color=TEXT, anchor=None):
        draw.text(xy, str(value), font=self.fonts[size], fill=color, anchor=anchor)

    def world(self, draw, row, grid):
        ox, oy, extent = 44, 152, 460
        cell = extent/self.r.size
        ax, ay, direction = int(row['x']), int(row['y']), int(row['direction'])
        forward = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
        right = (-forward[1], forward[0])
        # Same rectangular view as MiniGrid (including see_through_walls=True).
        for x in range(self.r.size):
            for y in range(self.r.size):
                dx, dy = x-ax, y-ay
                f = dx*forward[0]+dy*forward[1]
                side = dx*right[0]+dy*right[1]
                visible = 0 <= f < self.r.view and abs(side) <= self.r.view//2
                kind, color, _ = map(int, grid[x, y])
                fill = '#344053' if kind == 2 else ('#243e50' if visible else '#101a27')
                x0, y0 = ox+x*cell, oy+y*cell
                draw.rectangle((x0, y0, x0+cell, y0+cell), fill=fill, outline='#415064')
                if kind == 6:  # RINGS encodes food/danger as green/red MiniGrid balls.
                    cc = '#70e3a6' if color == 1 else '#ff7a83' if color == 0 else '#e8d38c'
                    pad = cell*.23
                    draw.ellipse((x0+pad, y0+pad, x0+cell-pad, y0+cell-pad), fill=cc)
                elif kind not in (0, 1, 2):
                    self.text(draw, (x0+cell/2, y0+cell/2), str(kind), 15, anchor='mm')
        cx, cy = ox+(ax+.5)*cell, oy+(ay+.5)*cell
        tip = (cx+forward[0]*cell*.40, cy+forward[1]*cell*.40)
        back = (cx-forward[0]*cell*.30, cy-forward[1]*cell*.30)
        draw.polygon([tip, (back[0]+right[0]*cell*.29, back[1]+right[1]*cell*.29),
                      (back[0]-right[0]*cell*.29, back[1]-right[1]*cell*.29)], fill='#f5dd93')

    def frame(self, step, part, subframes):
        r = self.r
        row = r.row(step)
        begin = step*r.dt+part*r.dt/subframes
        end = step*r.dt+(part+1)*r.dt/subframes
        volts, chemical, modvolts, ci = r.state(end)
        spikes = r.counts('ring', begin, end)
        modspikes = r.counts('mod', begin, end)
        image = Image.new('RGB', (self.width, self.height), BG)
        d = ImageDraw.Draw(image)
        d.text((26, 20), 'RINGS / нейронная активность', fill=TEXT, font=self.heading)
        self.text(d, (27, 66), 'Записанные состояния Brian2 • 2D-схема групп, не полная карта синапсов', 19, MUTED)
        self.text(d, (1574, 30), 'v%s  |  seed %s' % (r.meta['agent_version'], r.meta['seed']), 21, anchor='ra')
        self.text(d, (1574, 65), 'Шаг %d / %d  •  окно сети %.1f–%.1f ms' % (step, len(r.rows)-1, begin, end), 19, MUTED, 'ra')
        rounded_panel(d, (24, 106, 526, 790), radius=16, fill=PANEL)
        rounded_panel(d, (546, 106, 1576, 790), radius=16, fill=PANEL)
        self.text(d, (44, 119), 'СРЕДА — до выполнения действия', 19)
        self.world(d, row, r.data['scene_grids'][step])
        self.text(d, (44, 625), 'Энергия до шага: %.3f' % row['energy_before'], 21)
        if part == subframes-1:
            decision = ACTIONS.get(int(row['action']), str(int(row['action'])))
            if int(row['action']) == 6:
                decision = ('отдых (MotorRest)' if row.get('MotorRest', 0) > 0
                            else 'нет моторной команды')
            self.text(d, (44, 657), 'Решение: '+decision, 19, GOLD)
        else:
            self.text(d, (44, 657), 'Сеть рассчитывает действие…', 19, MUTED)
        self.text(d, (44, 690), 'Сенсоры          слева   впереди   справа', 17, MUTED)
        for offset, prefix, color, label in [(0, 'food', '#70e3a6', 'Еда'), (1, 'danger', '#ff7a83', 'Опасность')]:
            y = 718+offset*27
            self.text(d, (44, y), label, 19, color)
            for x, side in zip([253, 349, 449], ['left', 'ahead', 'right']):
                self.text(d, (x, y), '%.2f' % row[prefix+'_'+side], 19, color)
        self.text(d, (44, 769), 'Стена впереди: %s' % ('ДА' if row['wall_ahead'] else 'нет'), 15, MUTED)
        self.text(d, (567, 119), 'НЕЙРОНЫ: цвет — потенциал, светлый ободок — спайк в этом окне', 19)
        selected = {0: 'MotorLeft', 1: 'MotorRight', 2: 'MotorForward', 6: 'MotorRest'}.get(int(row['action']))
        if selected == 'MotorRest' and row.get('MotorRest', 0) <= 0:
            selected = None
        for name, (lo, hi) in zip(r.names, r.bounds):
            cx, cy = self.centres[name]
            radius = 61
            d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=LINE, width=2)
            chosen = name == selected and part == subframes-1
            self.text(d, (cx, cy-90), name, 21, GOLD if chosen else TEXT, 'mm')
            for i in range(lo, hi):
                angle = -math.pi/2+2*math.pi*(i-lo)/(hi-lo)
                x, y = cx+radius*math.cos(angle), cy+radius*math.sin(angle)
                if spikes[i]:
                    d.ellipse((x-14, y-14, x+14, y+14), outline=GOLD, width=3)
                d.ellipse((x-9, y-9, x+9, y+9), fill=voltage_color(volts[i]))
            self.text(d, (cx, cy-8), str(int(spikes[lo:hi].sum())), 28, GOLD, 'mm')
            self.text(d, (cx, cy+19), 'спайков', 15, MUTED, 'mm')
            self.text(d, (cx, cy+91), 'V: %.0f…%.0f mV' % (volts[lo:hi].min(), volts[lo:hi].max()), 17, MUTED, 'mm')
        # Modulator secretory neuron plus its measured chemical concentration.
        times = r.data['chemical_t_ms']
        first = np.searchsorted(times, max(0, end-20*r.dt))
        indices = np.unique(np.linspace(first, ci, min(160, ci-first+1)).astype(int))
        for k, name in enumerate(r.mods):
            x, y = 24+k*394, 808
            color = MOD_COLORS[k]
            rounded_panel(d, (x, y, x+370, 939), radius=12, fill=PANEL)
            self.text(d, (x+16, y+10), name, 21)
            nx, ny = x+335, y+27
            if modspikes[k]:
                d.ellipse((nx-15, ny-15, nx+15, ny+15), outline=GOLD, width=3)
            d.ellipse((nx-10, ny-10, nx+10, ny+10), fill=voltage_color(modvolts[k]))
            c = float(chemical[k])
            self.text(d, (x+16, y+43), 'c = %.3f' % c, 21, color)
            self.text(d, (x+150, y+46), 'нейрон: %d сп.' % modspikes[k], 17, MUTED)
            d.rectangle((x+16, y+77, x+352, y+85), fill=LINE)
            d.rectangle((x+16, y+77, x+16+336*np.clip(c, 0, 1), y+85), fill=color)
            if len(indices) > 1:
                left_time = max(0, end-20*r.dt)
                points = [(x+16+336*(times[i]-left_time)/max(1, end-left_time),
                           y+119-25*np.clip(r.data['chemical_c'][k, i], 0, 1)) for i in indices]
                d.line(points, fill=color, width=2)
        # Legend always uses the same scale, regardless of model/parameter set.
        self.text(d, (26, 953), 'Цвет V:', 17, MUTED)
        for j, value in enumerate([-80, -65, -50, -40]):
            x = 105+j*86
            d.ellipse((x, 954, x+16, 970), fill=voltage_color(value))
            self.text(d, (x+23, 952), str(value), 17)
        self.text(d, (461, 953), 'mV; вне шкалы цвет насыщается', 17, MUTED)
        self.text(d, (1573, 953), 'Модуляторы: c от 0 до 1 • линии: последние 20 шагов', 17, MUTED, 'ra')
        return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', help='v40/v41 NPZ created with --record on')
    p.add_argument('--output', required=True, help='New MP4 path')
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--frames-per-step', type=int, default=3)
    p.add_argument('--start-step', type=int, default=0, help='Inclusive zero-based step')
    p.add_argument('--end-step', type=int, help='Exclusive step; default all completed steps')
    p.add_argument('--preview', help='Optional new PNG showing the last neural window of a step')
    p.add_argument('--preview-step', type=int, help='Default: first selected step')
    p.add_argument('--font', help='TTF font with Cyrillic support')
    a = p.parse_args()
    if not np.isfinite(a.fps) or a.fps <= 0 or a.frames_per_step < 1:
        p.error('FPS and frames-per-step must be positive')
    r = Recording(a.recording)
    end = len(r.rows) if a.end_step is None else a.end_step
    if not 0 <= a.start_step < end <= len(r.rows):
        p.error('Invalid step range')
    preview_step = a.start_step if a.preview_step is None else a.preview_step
    if not 0 <= preview_step < len(r.rows):
        p.error('Invalid preview-step')
    output = Path(a.output)
    if output.suffix.lower() != '.mp4' or output.exists():
        p.error('Output must be a new .mp4 file; existing files are never overwritten')
    if a.preview and (Path(a.preview).suffix.lower() != '.png' or Path(a.preview).exists()):
        p.error('Preview must be a new .png file')
    # Too many subframes would imply temporal detail absent from StateMonitor.
    sample_dt = max(float(np.diff(r.data[k]).max()) if len(r.data[k]) > 1
                    else float(r.meta.get('trace_dt_ms', 1.0))
                    for k in ('state_t_ms', 'chemical_t_ms'))
    if r.dt/a.frames_per_step < sample_dt-1e-5:
        p.error('frames-per-step exceeds the temporal resolution of the saved traces')
    renderer = Renderer(r, a.font)
    import imageio.v2 as imageio
    import imageio_ffmpeg  # Ensure the bundled ffmpeg backend is available before rendering.
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem+'.partial-'+uuid.uuid4().hex+'.mp4')
    total = (end-a.start_step)*a.frames_per_step
    writer = None
    count = 0
    started = last_update = time.monotonic()
    print('Rendering %d frames; %.1f s video; %.2f environment steps/s' %
          (total, total/a.fps, a.fps/a.frames_per_step), flush=True)
    try:
        writer = imageio.get_writer(str(temp), fps=a.fps, codec='libx264',
                                    quality=8, macro_block_size=2, ffmpeg_log_level='error')
        for step in range(a.start_step, end):
            for part in range(a.frames_per_step):
                writer.append_data(np.asarray(renderer.frame(step, part, a.frames_per_step)))
                count += 1
                now = time.monotonic()
                if now-last_update >= 5 or count == total:
                    print('%d/%d frames (%.1f%%), elapsed %.1fs' %
                          (count, total, 100*count/total, now-started), flush=True)
                    last_update = now
        writer.close()
        writer = None
        if output.exists():
            raise FileExistsError(str(output))
        os.rename(str(temp), str(output))
    except BaseException:
        if writer is not None:
            writer.close()
        if temp.exists():
            print('Incomplete video retained at: '+str(temp), flush=True)
        raise
    if a.preview:
        preview = Path(a.preview)
        preview.parent.mkdir(parents=True, exist_ok=True)
        with preview.open('xb') as f:
            renderer.frame(preview_step, a.frames_per_step-1, a.frames_per_step).save(f, format='PNG')
    print('Saved: '+str(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
