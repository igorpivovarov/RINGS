#!/usr/bin/env python3
"""Parallel RINGS benchmark. Python >=3.10; see README_RU.md."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

VERSION = 1

def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(path):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location('benchmark_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def worker(request):
    """Each invocation has a fresh interpreter, RNG state and Brian network."""
    import dataclasses
    import random
    import numpy as np
    q = json.loads(Path(request).read_text())
    start = time.monotonic()
    result = {k: q[k] for k in ('env_seed', 'repeat', 'agent_seed')}
    try:
        random.seed(q['agent_seed'])
        np.random.seed(q['agent_seed'])
        m = load_module(q['model'])
        if q.get('minigrid_compat'):
            # Custom Ball types retain their semantics; only image encoding changes.
            from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
            def ball_encode(self):
                return (OBJECT_TO_IDX['ball'], COLOR_TO_IDX[self.color], 0)
            for name in ('Food', 'Danger'):
                getattr(m, name).encode = ball_encode
        # run_episode seeds the environment; only the controller gets a new seed.
        original = m.RingsSNNController
        def controller(*args, **kwargs):
            if len(args) >= 2:
                args = list(args)
                args[1] = q['agent_seed']
            else:
                kwargs['seed'] = q['agent_seed']
            return original(*args, **kwargs)
        m.RingsSNNController = controller
        # Capture actual termination flags, including death exactly at the horizon.
        env_step = m.RingsHomeostasisEnv.step
        ending = {}
        def step(self, action):
            out = env_step(self, action)
            ending.update(terminated=bool(out[2]), truncated=bool(out[3]))
            return out
        m.RingsHomeostasisEnv.step = step
        params = m.load_params(q['params']) if q['params'] else m.RingsParams()
        r = m.run_episode(seed=q['env_seed'], params=params, visualize=False,
                          max_steps=q['max_steps'], status=False, record=False,
                          grid_size=q['grid_size'], danger_speed=q['danger_speed'])
        metrics = dataclasses.asdict(r) if dataclasses.is_dataclass(r) else dict(r)
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        if not ending or not 1 <= metrics['steps'] <= q['max_steps']:
            raise ValueError('Invalid episode result / missing termination flags')
        for value in metrics.values():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Episode metrics must be finite numbers')
        result.update(status='ok', metrics=metrics, **ending)
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    result['wall_seconds'] = time.monotonic() - start
    dump(q['result'], result)
    return 0 if result['status'] == 'ok' else 1

def launch(q, timeout):
    request = Path(q['result']).with_suffix('.request.json')
    dump(request, q)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env.update(MPLBACKEND='Agg', PYTHONHASHSEED=str(q['agent_seed']))
    log = Path(q['result']).with_suffix('.log')
    try:
        with log.open('w') as output:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_worker', str(request)],
                               env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        if p.returncode and not Path(q['result']).exists():
            raise RuntimeError(f'Worker exit {p.returncode}; see {log}')
        return json.loads(Path(q['result']).read_text())
    except Exception as e:
        r = {k: q[k] for k in ('env_seed', 'repeat', 'agent_seed')}
        r.update(status='error', error=str(e))
        dump(q['result'], r)
        return r

def bootstrap(values, samples, seed=8301):
    """Percentile CI on independent environment-seed means, not pseudo-replicates."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    return np.quantile(means, [.025, .975]).tolist()

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def collect(folder):
    folder = Path(folder)
    config = json.loads((folder / 'manifest.json').read_text())
    paths = sorted((folder / 'runs').glob('*.result.json'))
    if paths:
        rows = [json.loads(p.read_text()) for p in paths]
    elif (folder / 'runs.csv').exists():
        print(f'INFO: {folder}: loading episodes from runs.csv (no per-run JSON files)', file=sys.stderr)
        rows = []
        metadata = {'env_seed', 'repeat', 'agent_seed', 'status', 'terminated',
                    'truncated', 'wall_seconds', 'error'}
        def boolean(value, key):
            if value.lower() in ('true', '1'): return True
            if value.lower() in ('false', '0'): return False
            raise ValueError(f'{folder}/runs.csv: invalid {key}={value!r}')
        with (folder / 'runs.csv').open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required = {'env_seed', 'repeat', 'status', 'steps'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{folder}/runs.csv: missing required columns {required}')
            for item in reader:
                r = dict(env_seed=int(item['env_seed']), repeat=int(item['repeat']), status=item['status'])
                for key in ('agent_seed', 'wall_seconds'):
                    if item.get(key): r[key] = int(item[key]) if key == 'agent_seed' else float(item[key])
                if item.get('error'): r['error'] = item['error']
                if r['status'] == 'ok':
                    r['metrics'] = {k: float(v) for k, v in item.items() if k not in metadata and v not in ('', None)}
                    for key in ('terminated', 'truncated'):
                        r[key] = boolean(item.get(key, ''), key)
                rows.append(r)
    else:
        raise ValueError(f'{folder}: no run results found. Supply runs/*.result.json or runs.csv; '
                         'summary.json alone does not contain per-seed observations.')
    seen = set()
    for r in rows:
        identity = (r['env_seed'], r['repeat'])
        if identity in seen:
            raise ValueError(f'{folder}: duplicate episode {identity}')
        seen.add(identity)
        if not 0 <= r['repeat'] < config['repeats']:
            raise ValueError(f'{folder}: repeat index outside manifest range: {identity}')
        if r['status'] == 'ok':
            steps = r['metrics']['steps']
            if not math.isfinite(steps) or not 1 <= steps <= config['max_steps']:
                raise ValueError(f'{folder}: invalid steps for {identity}: {steps}')
    good = [r for r in rows if r['status'] == 'ok']
    groups = {}
    for r in good:
        groups.setdefault(r['env_seed'], []).append(r)
    complete = {s: g for s, g in groups.items() if len(g) == config['repeats']}
    return config, rows, good, complete

def report(folder, samples=5000):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    c, rows, good, groups = collect(folder)
    write_csv(folder / 'runs.csv', [dict({k: v for k, v in r.items() if k != 'metrics'}, **r.get('metrics', {})) for r in rows])
    summary = dict(expected_runs=c['seeds'] * c['repeats'], successful_runs=len(good),
                   failed_runs=sum(r['status'] != 'ok' for r in rows), complete_seeds=len(groups),
                   inference='Bootstrap of complete environment-seed means; capped lifetime', metrics={})
    if not good:
        dump(folder / 'summary.json', summary)
        raise RuntimeError('No successful episodes; inspect runs/*.log and runs.csv')
    keys = sorted(set.intersection(*(set(r['metrics']) for r in good)))
    seed_rows = []
    for s, g in sorted(groups.items()):
        seed_rows.append(dict(env_seed=s, **{k: float(np.mean([r['metrics'][k] for r in g])) for k in keys}))
    write_csv(folder / 'seed_means.csv', seed_rows)
    for k in keys:
        vals = [r[k] for r in seed_rows]
        all_vals = [r['metrics'][k] for r in good]
        summary['metrics'][k] = dict(mean_all_successful=float(np.mean(all_vals)),
            mean_complete_seeds=float(np.mean(vals)) if vals else None,
            median_all_successful=float(np.median(all_vals)),
            sd_all_successful=float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else None,
            ci95_complete_seeds=bootstrap(vals, samples))
    summary['censored_runs'] = sum(r['truncated'] and not r['terminated'] for r in good)
    summary['reached_horizon_fraction'] = float(np.mean([r['metrics']['steps'] == c['max_steps'] for r in good]))
    dump(folder / 'summary.json', summary)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    times = np.array([r['metrics']['steps'] for r in good])
    ax[0, 0].hist(times, bins=min(40, max(5, int(np.sqrt(len(times))))), color='#3178a5')
    ax[0, 0].set(xlabel='Observed lifetime (steps, capped)', ylabel='Episodes')
    # Administrative censoring only: fraction alive beyond each t, strictly below horizon.
    t = np.unique(np.r_[0, times[times < c['max_steps']], c['max_steps'] - 1])
    ax[0, 1].step(t, [(times > x).mean() for x in t], where='post')
    ax[0, 1].set(xlabel='Step (< observation horizon)', ylabel='Fraction surviving', ylim=(0, 1.02))
    vals = np.array([r['steps'] for r in seed_rows])
    if len(vals):
        ax[1, 0].plot(np.arange(1, len(vals)+1), np.cumsum(vals)/np.arange(1, len(vals)+1))
        ax[1, 1].scatter([r['env_seed'] for r in seed_rows], vals, s=15)
    ax[1, 0].set(xlabel='Complete seeds (seed order)', ylabel='Running mean capped lifetime')
    ax[1, 1].set(xlabel='Environment seed', ylabel='Mean lifetime over repeats')
    fig.suptitle(f"{Path(c['model']).name}: {len(good)} episodes, {len(groups)} complete seeds\n"
                 f"Horizon {c['max_steps']}; censored {summary['censored_runs']}; errors {summary['failed_runs']}")
    fig.tight_layout()
    fig.savefig(folder / 'performance.png', dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))

def run(a):
    import numpy as np
    if min(a.workers, a.seeds, a.repeats, a.max_steps, a.bootstrap) < 1 or a.timeout <= 0:
        raise ValueError('Counts, timeout and horizon must be positive')
    if a.start_seed < 0 or a.start_seed + a.seeds > 2**32 or not 0 <= a.master_seed < 2**32:
        raise ValueError('Seeds must fit uint32')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = Path(a.model).resolve()
    params = str(Path(a.params).resolve()) if a.params else None
    c = dict(version=VERSION, model=str(model), model_sha256=digest(model), params=params,
             params_sha256=digest(params) if params else None, seeds=a.seeds, repeats=a.repeats,
             start_seed=a.start_seed, master_seed=a.master_seed, max_steps=a.max_steps,
             grid_size=a.grid_size, danger_speed=a.danger_speed, minigrid_compat=a.minigrid_compat)
    manifest = out / 'manifest.json'
    if manifest.exists():
        if not a.resume:
            raise ValueError('Output already exists; use --resume or a new output directory')
        if json.loads(manifest.read_text()) != c:
            raise ValueError('Resume configuration/model hash mismatch')
    else:
        if any(out.iterdir()):
            raise ValueError('New output directory must be empty')
        dump(manifest, c)
    (out / 'runs').mkdir(exist_ok=True)
    pending = []
    for s in range(a.start_seed, a.start_seed+a.seeds):
        for r in range(a.repeats):
            target = out / 'runs' / f'seed{s:010d}_repeat{r:04d}.result.json'
            if target.exists():
                old = json.loads(target.read_text())
                if old['status'] == 'ok':
                    continue
                target.unlink()  # failed runs are retried on resume
            agent_seed = int(np.random.SeedSequence([a.master_seed, s, r]).generate_state(1)[0])
            pending.append(dict(c, env_seed=s, repeat=r, agent_seed=agent_seed, result=str(target)))
    total = c['seeds'] * c['repeats']
    done = total - len(pending)
    print(f'{total} episodes; {len(pending)} pending; up to {a.workers} model processes', flush=True)
    # Threads only supervise subprocesses: all model computation uses separate CPU processes.
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(launch, q, a.timeout) for q in pending]
        for f in cf.as_completed(futures):
            r = f.result()
            done += 1
            print(f"[{done}/{total}] seed={r['env_seed']} repeat={r['repeat']} {r['status']} "
                  f"steps={r.get('metrics', {}).get('steps', '-')}", flush=True)
    report(out, a.bootstrap)
    _, rows, _, _ = collect(out)
    return 1 if any(r['status'] != 'ok' for r in rows) else 0

def compare(a):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if a.labels is not None and len(a.labels) != len(a.folders):
        raise ValueError('--labels requires exactly one label per result folder')
    loaded = [collect(p) for p in a.folders]
    if len(loaded) < 2:
        raise ValueError('Provide at least two result folders')
    if a.bootstrap < 1:
        raise ValueError('--bootstrap must be positive')
    keys = ('seeds', 'repeats', 'start_seed', 'master_seed', 'max_steps',
            'grid_size', 'danger_speed', 'minigrid_compat')
    differences = {key: [c.get(key) for c, _, _, _ in loaded]
                   for key in keys
                   if any(c.get(key) != loaded[0][0].get(key) for c, _, _, _ in loaded[1:])}
    if 'max_steps' in differences:
        raise ValueError('Different max_steps: capped lifetimes have different horizons; '
                         '--allow-different-settings cannot override this. Use a common horizon.')
    if differences and not a.allow_different_settings:
        raise ValueError('Different comparison settings: ' + ', '.join(differences) +
                         '. To compare these experimental conditions explicitly, add --allow-different-settings')
    messages = []
    if differences:
        messages.append('Experimental settings differ; differences in performance may reflect '
                        'environment/RNG settings as well as the model.')
        for key, values in differences.items():
            messages.append(key + ': ' + '; '.join(f'{folder}={value}'
                            for folder, value in zip(a.folders, values)))
    for folder, (config, rows, good, groups) in zip(a.folders, loaded):
        expected = config['seeds'] * config['repeats']
        if len(good) != expected:
            messages.append(f'{folder}: {len(good)}/{expected} successful episodes; '
                            'only complete seed groups are used. Missing/error runs may bias results.')
    common = sorted(set.intersection(*(set(g) for _, _, _, g in loaded)))
    if len(common) < 2:
        raise ValueError('Comparison needs at least two common complete seeds')
    for folder, (_, _, _, groups) in zip(a.folders, loaded):
        if len(groups) != len(common):
            messages.append(f'{folder}: using {len(common)} common complete seeds out of {len(groups)} complete seeds')
    for message in messages:
        print('WARNING: ' + message, file=sys.stderr, flush=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'comparison_settings.json', dict(
        folders=[str(Path(p).resolve()) for p in a.folders],
        differing_settings=differences, warnings=messages, common_seed_ids=common,
        configurations=[c for c, _, _, _ in loaded],
        analysis='Paired bootstrap of means over common environment seed IDs; '
                 'all successful repeats within each complete seed group are averaged. '
                 'Same seed IDs do not imply identical trajectories across conditions.'))
    results, means, intervals = [], [], []
    baseline = None
    labels = []
    for index, (folder, (c, _, _, groups)) in enumerate(zip(a.folders, loaded)):
        v = np.array([np.mean([r['metrics']['steps'] for r in groups[s]]) for s in common])
        if baseline is None:
            baseline = v
        label = a.labels[index] if a.labels is not None else Path(folder).name
        ci = bootstrap(v, a.bootstrap)
        results.append(dict(model=label, common_seeds=len(common), mean=float(v.mean()), ci95=ci,
                            difference_from_first=float((v-baseline).mean()),
                            paired_difference_ci95=bootstrap(v-baseline, a.bootstrap)))
        means.append(v.mean()); intervals.append(ci); labels.append(label)
    dump(out / 'comparison.json', results)
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(labels)), 6))
    for i, (mean, ci) in enumerate(zip(means, intervals)):
        ax.plot([i, i], ci, color='#3178a5', linewidth=2)
        ax.scatter([i], [mean], color='#3178a5')
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel='Mean capped lifetime (steps)',
           title=f'95% seed-bootstrap CI; {len(common)} common environment seeds' +
                 ('\nDifferent settings: ' + ', '.join(differences) if differences else ''))
    ax.tick_params(axis='x', labelrotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment('right')
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=170); plt.close(fig)
    print(json.dumps(results, indent=2))
    return 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('_worker'); w.add_argument('request')
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--params')
    r.add_argument('--workers', type=int, default=30)
    r.add_argument('--seeds', type=int, default=100)
    r.add_argument('--repeats', type=int, default=10)
    r.add_argument('--start-seed', type=int, default=1)
    r.add_argument('--master-seed', type=int, default=20261005)
    r.add_argument('--max-steps', type=int, default=1000)
    r.add_argument('--grid-size', type=int, default=15)
    r.add_argument('--danger-speed', type=float, default=0.0)
    r.add_argument('--timeout', type=float, default=3600, help='Wall seconds per episode')
    r.add_argument('--resume', action='store_true')
    r.add_argument('--minigrid-compat', action='store_true', help='Encode custom Food/Danger as colored balls')
    r.add_argument('--bootstrap', type=int, default=5000)
    s = sub.add_parser('report'); s.add_argument('folder'); s.add_argument('--bootstrap', type=int, default=5000)
    c = sub.add_parser('compare'); c.add_argument('folders', nargs='+', help='Two or more result directories, in plotting order; first is the baseline'); c.add_argument('--output', required=True)
    c.add_argument('--bootstrap', type=int, default=5000)
    c.add_argument('--labels', nargs='+', help='One quoted plot label per directory, in the same order')
    c.add_argument('--allow-different-settings', action='store_true',
                   help='Allow different settings except max_steps; warn and use common complete seed groups')
    a = p.parse_args()
    if a.command == '_worker': return worker(a.request)
    if a.command == 'run': return run(a)
    if a.command == 'report': report(a.folder, a.bootstrap); return 0
    return compare(a)

if __name__ == '__main__':
    sys.exit(main())
