# RINGS

**RINGS** is an experimental project for studying how progressively richer internal neural dynamics can support adaptive behaviour and minimal self-representation in a simple embodied agent.

The agent is controlled by a **spiking neural network (SNN)** implemented in [Brian2](https://brian2.readthedocs.io/) and acts in a **Gymnasium/MiniGrid** environment containing food, danger and an internal energy variable. To survive, it must collect food, avoid danger and regulate its own energy. The main behavioural benchmark is lifetime in environment steps, capped by `max_steps`, while the neural dynamics themselves are also a primary object of study.

[Русская версия](README_RU.md)

## Research idea

RINGS is motivated by **TEVSER (Theory of Evolving Self-Representations)**. In TEVSER, increasingly capable behaviour emerges as a system acquires internal representations that distinguish and integrate information about its own state, the environment and the relation between them.

The project asks a deliberately minimal question:

> **What changes when an agent progresses from direct stimulus–response mappings to persistent internal neural states that participate in action selection?**

Rather than beginning with a complex cognitive architecture, RINGS builds this transition step by step in a small, inspectable SNN. The goal is not to claim that a particular recurrent pattern is already a complete biological self-model. The goal is to test whether mechanisms that are candidates for elementary self-representation produce identifiable changes in neural dynamics and behaviour.

A central hypothesis is that **recurrent attractor-like activity can provide a persistent internal state that is not reducible to the current sensory input alone**. Such a state may integrate sensory, homeostatic and action-related information over time and thereby serve as a minimal substrate for self-related regulation.

## Model progression

The repository preserves a sequence of increasingly complex models. This progression is intentional: simpler models make the contribution of each added mechanism easier to interpret.

1. **Reactive model (R)** — feed-forward stimulus–response control. Sensory inputs directly drive behaviour through fast neural dynamics.
2. **Chemically modulated model (C)** — adds slow internal modulatory variables representing longer-lived physiological context, while keeping the neural controller feed-forward.
3. **Recurrent ring models** — introduce recurrent neural populations capable of maintaining structured activity beyond the immediate input.
4. **Coupled recurrent state (v0.51)** — separates persistent internal state from motor readout and allows the state to evolve continuously while influencing behaviour.

This sequence supports both **mechanistic comparison** and **ablation experiments**. The same environment can therefore be used to ask whether behavioural differences come from slow chemical modulation, recurrent state dynamics, or their interaction.

## Why this project

RINGS is intended as a small computational testbed rather than a performance-maximising agent. Its main value is that the internal mechanisms remain explicit enough to inspect, record, visualise and selectively disable.

The project therefore combines three kinds of analysis:

- **Behavioural performance:** survival time, food collection, danger encounters and related episode statistics.
- **Mechanistic interventions:** fixed-parameter ablations and matched comparisons between model variants.
- **Neural dynamics:** recordings, population activity, recurrent-state trajectories and PCA-based visualisation of state-space dynamics.

This makes it possible to study not only **whether** an agent behaves better, but also **what internal dynamical mechanism changed** when the architecture was modified.

The current codebase should be treated as an active research prototype. Model versions are retained because the historical progression itself is useful for understanding how persistent recurrent dynamics emerge from simpler controllers.

## Documentation scope

This reference covers the source snapshot available on 2026-10-14: prototype models, R/C baselines, benchmark, recording renderers and R/C test scripts. It is not an inventory fetched from the GitHub repository. Archive files are documented separately; missing version numbers are not assumed to exist. All explicit command-line options and all declared `RingsParams` fields in the inspected scripts are listed below. Internal Python methods are not a public CLI.


## Contents

- [Installation](#installation)
- [Main scripts](#main-scripts)
- [Quick start](#quick-start)
- [Workflow](#workflow-and-experimental-controls)
- [Parallel benchmark](#parallel-benchmark)
- [Video and network views](#recordings-video-and-network-views)
- [Complete CLI reference](#complete-cli-reference)
- [Model parameter JSON](#model-parameter-json-reference)
- [Historical models](#historical-models-per-script-reference)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy brian2 gymnasium minigrid matplotlib pygame pillow imageio imageio-ffmpeg
```

These are dependency names, not a tested version lock. Keep a separate environment and record installed versions for reproducibility. The recent scripts target Python 3.8-compatible syntax; this does not guarantee that the newest releases of all dependencies support Python 3.8. Legacy notebook exports may additionally need IPython. Simulation visualisation needs a graphical display; benchmark and offline video rendering can run headlessly. Neural recordings can consume substantial memory.

## Main scripts

| Script | Purpose |
| --- | --- |
| `rings_minigrid_brian_2_prototype51.py` | v0.51.3: coupled recurrent state, separate motor readout, ablations, tune/evaluate/select. |
| `rings_minigrid_brian_2_prototype43.py` | v0.43.0: recurrent motor rings, tune/evaluate/select. |
| `rings_model_R.py` | RC-1.3 reactive feedforward baseline; fast neural transients, no slow chemistry. |
| `rings_model_C.py` | RC-1.3 feedforward baseline plus slow chemical modulation; no recurrent synapses. |
| `rings_benchmark.py` | v1.2.1: parallel episodes, aggregate reports and paired multi-model comparison. |
| `rings_neural_video.py` | Preserved ring-based NPZ replay for v40–v43; no simulation. |
| `rings_neural_video_attr.py` | v3.1: v50/v51 population heatmap, 3D PCA and shared timeline, 1920×1080; legacy fallback. |
| `test_rings_RC.py` | R/C mechanism and interface checks; writes RINGS_RC_validation.json. |
| `test_rings_RC_visual.py` | Headless rendering and visual on/off non-interference checks; writes RINGS_RC_visual_validation.json. |


## Quick start

```bash
# Single recorded episode
python3 rings_minigrid_brian_2_prototype51.py --mode run --seed 7 \
  --max-steps 1000 --danger-speed 0.5 --visual on \
  --record on --record-path logs/run51.npz --debug-log logs/log51.csv

# Parallel tuning
python3 rings_minigrid_brian_2_prototype51.py --mode tune \
  --trials 100 --workers 30 --num-seeds 12 --start-seed 1 \
  --max-steps 1000 --danger-speed 0.5 --output config/rings_best_params_51.json

# Independent evaluation
python3 rings_minigrid_brian_2_prototype51.py --mode evaluate \
  --params best51.json --seeds 1001 1002 1003 1004 \
  --max-steps 1000 --danger-speed 0.5 --eval-report logs/evaluation51.json

# Offline 16:9 video
python3 rings_neural_video_attr.py logs/run51.npz --output video/run51_attr.mp4
```
## Workflow and experimental controls

1. Tune on development scenarios, then freeze parameters.
2. Evaluate on held-out scenarios. Prototype51 tuning excludes validation seeds from training and also reserves fixture seeds 4 and 7 from validation.
3. Benchmark versions under the same environment, horizon and seed protocol. Use a new output directory for each condition.
4. Compare complete seed groups, then inspect selected recordings to understand mechanisms.

For prototype51, `--params` does not initialise the random search in tune mode; run/evaluate load it, and select interprets it as a search checkpoint. Resume uses the checkpoint named by `--output`. `--workers 30` explicitly fits a 30-core server; prototype51 defaults to 32 workers (allowed 1–100), whereas benchmark defaults to 30 and R/C tune defaults to 1. Evaluation mode is not automatically a parallel benchmark.

`--mode select --params SEARCH.json --output SELECTED.json` reselects from `all_results` without repeating the parameter search; it can run candidate checks, so it is not necessarily instantaneous. Preserve the original checkpoint. A checkpoint without `best_params` is not a ready parameter file. v0.51 parameter JSON is not interchangeable with earlier schemas.

The v0.51 and R/C CLIs and benchmark default to **static dangers (`--danger-speed 0.0`)**. Pass `0.5` explicitly for moving-danger experiments. In v0.51, 0.5 schedules one danger-movement update per two environment steps; blocked proposals need not produce displacement. An environment step is not a second: `snn_step_ms` is simulated Brian2 time, while video speed is controlled independently.

A fixed-parameter ablation isolates the effect of removing a mechanism more directly than retuning each ablated model. Retuned comparisons answer a different question. Neither approach should silently mix development and held-out scenarios.

### Ablation modes

| Model | Values | Meaning |
| --- | --- | --- |
| v0.51 | `full`, `no-state-recurrence`, `no-chemistry`, `both` | Intact circuit; state-recurrence ablation; chemical effects disabled; both ablations. This does not remove every feedback loop in the agent/environment. |
| R/C | `full`, `no-chemistry` | C without chemistry reproduces the R circuit mechanism under the tested conditions. R already has no chemistry. |
| Historical models | See each CLI below | Do not assume every version supports ablations. |

## Parallel benchmark

```bash
python3 rings_benchmark.py run \
  --model rings_minigrid_brian_2_prototype51.py --params best51.json \
  --output results_v051 --workers 30 --seeds 100 --repeats 10 \
  --start-seed 1001 --max-steps 1000 --danger-speed 0.5

python3 rings_benchmark.py run \
  --model rings_minigrid_brian_2_prototype51.py --params best51.json \
  --ablation no-chemistry --output results_v051_Ac \
  --workers 30 --seeds 100 --repeats 10 --start-seed 1001 \
  --max-steps 1000 --danger-speed 0.5

python3 rings_benchmark.py report results_v051
python3 rings_benchmark.py compare results_v051 results_v051_Ac \
  --output comparison --allow-different-settings \
  --labels "v0.51 full" "v0.51 no chemistry"
```

The default 100 seeds × 10 repeats gives 1,000 episodes. Repeats share an environment seed but use different agent seeds. Uncertainty is bootstrapped over **environment-seed means**, not over 1,000 independent environments. More repeats do not replace a broader scenario sample, and no fixed run count guarantees statistical power.

`compare` accepts two or more directories. It uses the intersection of **complete** seed groups across every directory and requires at least two common groups. `--allow-different-settings` does not remove this requirement. Different `max_steps` is always rejected. If one set uses seeds 1–100 and another 1001–1100, this implementation cannot compare them; rerun matched scenarios, or implement an explicitly independent analysis. Never relabel seed IDs to manufacture pairing.

`manifest.json` plus either `runs/*.result.json` or `runs.csv` is required. Per-run JSON takes precedence when present. `summary.json` alone is insufficient. CSV metadata such as `ablation=full` must not be treated as a numeric metric. Resume validates the experiment configuration, including the ablation.

Outputs: `manifest.json`, per-run result/request/log files, `runs.csv`, `seed_means.csv`, `summary.json`, `performance.png`; comparison produces `comparison.json`, `comparison_settings.json`, `comparison.png`. The first folder is the baseline for paired differences. Missing/failed runs can change which complete seeds remain and may bias results.

The benchmark imports model Python code rather than forwarding arbitrary CLI arguments. It requires `RingsParams`, `load_params`, `RingsSNNController`, `RingsHomeostasisEnv.step`, and `run_episode` accepting the supplied arguments (`seed`, `params`, `visualize`, `max_steps`, `status`, `record`, `grid_size`, `danger_speed`). Results must be numeric metrics including `steps`; ablation is passed only when supported. Early prototypes do not all satisfy this interface. `--minigrid-compat` addresses object encoding, not general interface incompatibility.

## Recordings, video and network views

`--record on` saves neural data; `--debug-log` is a per-step CSV; `--video on` saves a model-provided video during execution. They serve different purposes. A CSV cannot reconstruct all individual neuron traces. Replay also depends on the recording schema: R/C NPZ files are not accepted by the ring/state video renderers described here.

The standalone video scripts need no model import or Brian2 simulation. Use `rings_neural_video.py` for the preserved ring view and `rings_neural_video_attr.py` for the v50/v51 heatmap/PCA view. The latter also has a legacy fallback, whose old layout is not converted to 16:9. In the new layout the scene is before the action; the decision is displayed only at the last subframe. The timeline shows 60 steps, all 12 inputs, state groups, support, motors and chemistry. The fixed scale maximum is shown for each row.

At the default 30 FPS and 3 frames/step the movie advances 10 environment steps/s. `--frames-per-step 6` makes it twice as slow at the same FPS. Existing MP4/PNG files are not overwritten. On interruption an incomplete video may remain under a `.partial-…mp4` filename.

PCA is centred, unstandardised PCA of the recorded state spike traces. It is fitted once to the whole recording with fixed axes, including when rendering a subset. Grey is the full retrospective trajectory, blue the recent tail and yellow the current point. It is not an estimated potential landscape. Independent recordings obtain independent PCA bases; their coordinates cannot be compared directly. `--phase-export` writes the basis and coordinates for further analysis.

The optional `rings_v043_3d.html` and `rings_v051_3d.html` are self-contained browser viewers of **static anatomy**. Rotate, zoom, inspect neurons, filter connections and use their matching JSON/GraphML exports. They have no CLI and do not play NPZ activity. Static architecture PNG/SVG files are illustrations, not executable scripts.

## Diagnostics and common problems

```bash
python3 rings_minigrid_brian_2_prototype51.py --self-test --test-report checks51.json
python3 test_rings_RC.py
python3 test_rings_RC_visual.py
```

Keep R/C test scripts beside both R/C model files. They accept no CLI options and write their named JSON reports; the visual test also writes `/tmp/RC_visual_*.png`. These checks assess mechanisms/interfaces, not statistical performance or attractor stability. `--self-test` in prototype51 uses default fixtures: omit `--params`, tuning mode and non-full ablations.

- Unknown CLI option: check the exact script’s `--help`; benchmark options do not automatically pass through to the model.
- No common complete seeds: inspect seed ranges, expected repeats, failed runs and manifests.
- `could not convert string to float: 'full'`: use benchmark v1.2.1 or later with metadata-aware CSV parsing.
- Video schema mismatch: use the appropriate recording/model generation, not only a matching filename suffix.
- Avoid assuming Ctrl-C cleanup behaviour is identical across historical tune scripts and benchmark versions; confirm that workers stopped before starting another large batch.

Do not commit large recordings, generated videos, checkpoints or result directories accidentally. Licensing terms are provided in the repository `LICENSE` file.

## Complete CLI reference

`-h` / `--help` is supplied automatically by argparse for every script with an argument parser. `None` means not supplied; Python expressions in default cells are source-level expressions, not shell syntax. A flag has no value argument.

### `rings_minigrid_brian_2_prototype51.py`

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--ablation` | `str; ('full', 'no-state-recurrence', 'no-chemistry', 'both')` | `'full'` | Disable specified mechanisms; accepted values are model-specific. |
| `--test-report` | `str` | `None` | JSON diagnostic report; requires --self-test. |
| `--self-test` | `flag` | `False` | Run built-in circuit checks, not a population benchmark. |
| `--mode` | `str; ['run', 'tune', 'evaluate', 'select']` | `'run'` | Operation: run one episode, tune parameters; evaluate/select where listed. |
| `--reserve-count` | `int` | `2` | Number of reserve candidates retained during selection. |
| `--validation-seeds` | `int × '+'` | `list(range(101, 109))` | Held-out evaluation seeds; not used for tune selection. |
| `--skip-validation` | `flag` | `False` | Skip held-out evaluation after tuning. |
| `--eval-report` | `str` | `'rings_v051_evaluation.json'` | Output JSON for --mode evaluate. |
| `--seed` | `int` | `7` | Episode RNG seed; in prototype tuning, the search RNG seed. |
| `--trials` | `int` | `50` | Number of random-search parameter candidates. |
| `--workers` | `int` | `32` | Parallel worker count; model tune and benchmark run are separate controls. |
| `--num-seeds` | `int` | `12` | Number of consecutive tuning scenarios. |
| `--start-seed` | `int` | `1` | First environment/scenario seed. |
| `--seeds` | `int × '+'` | `None` | Scenario seed list in models; NUMBER of seeds in benchmark run. |
| `--progress-interval` | `float` | `5.0` | Progress-message interval, in wall-clock seconds. |
| `--resume` | `flag` | `False` | Resume the compatible output/checkpoint rather than start again. |
| `--max-steps` | `int` | `250` | Episode horizon, in environment steps. |
| `--params` | `str` | `None` | Parameter JSON; select mode instead reads a tuning checkpoint. |
| `--output` | `str` | `'rings_v051_best.json'` | Output file or directory, depending on the script/subcommand. |
| `--visual` | `str; ['on', 'off']` | `'off'` | Enable the live GUI during simulation. |
| `--record` | `str; ['on', 'off']` | `'off'` | Save neural recording; format depends on model version. |
| `--video` | `str; ['on', 'off']` | `'off'` | Save video during run; independent from NPZ recording. |
| `--video-path` | `str` | `'rings_v051.mp4'` | Output video filename for simulation or replay. |
| `--video-fps` | `float` | `10.0` | Video frames per second. |
| `--replay` | `str` | `None` | Replay an existing recording instead of simulating an episode. |
| `--audit-path` | `str` | `None` | Export circuit topology/assumptions as JSON in run mode. |
| `--record-path` | `str` | `None` | Run-mode recording filename; enable --record on. |
| `--debug-log` | `str` | `None` | Per-step diagnostic CSV; tuning may interpret this as a log directory. |
| `--grid-size` | `int` | `15` | Square environment side length in cells. |
| `--danger-speed` | `float` | `0.0` | Danger movement-update budget per environment step; 0 means static. |


### `rings_model_R.py`

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--mode` | `str; ['run', 'tune', 'evaluate']` | `'run'` | Operation: run one episode, tune parameters; evaluate/select where listed. |
| `--params` | `str` | `None` | Parameter JSON; select mode instead reads a tuning checkpoint. |
| `--seed` | `int` | `7` | Episode RNG seed; in prototype tuning, the search RNG seed. |
| `--seeds` | `int × '+'` | `list(range(1, 13))` | Scenario seed list in models; NUMBER of seeds in benchmark run. |
| `--max-steps` | `int` | `1000` | Episode horizon, in environment steps. |
| `--grid-size` | `int` | `15` | Square environment side length in cells. |
| `--danger-speed` | `float` | `0.0` | Danger movement-update budget per environment step; 0 means static. |
| `--visual` | `str; ['on', 'off']` | `'off'` | Enable the live GUI during simulation. |
| `--visual-history` | `int` | `20` | Live history length, in environment steps. |
| `--record` | `str; ['on', 'off']` | `'off'` | Save neural recording; format depends on model version. |
| `--record-path` | `str` | `None` | Run-mode recording filename; enable --record on. |
| `--debug-log` | `str` | `None` | Per-step diagnostic CSV; tuning may interpret this as a log directory. |
| `--audit-path` | `str` | `None` | Export circuit topology/assumptions as JSON in run mode. |
| `--ablation` | `str; ['full', 'no-chemistry']` | `'full'` | Disable specified mechanisms; accepted values are model-specific. |
| `--trials` | `int` | `100` | Number of random-search parameter candidates. |
| `--workers` | `int` | `1` | Parallel worker count; model tune and benchmark run are separate controls. |
| `--progress-interval` | `float` | `30.0` | Progress-message interval, in wall-clock seconds. |
| `--search-seed` | `int` | `20261012` | Random-search RNG seed, separate from scenario seeds. |
| `--output` | `str` | `'rings_R_results.json'` | Output file or directory, depending on the script/subcommand. |


### `rings_model_C.py`

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--mode` | `str; ['run', 'tune', 'evaluate']` | `'run'` | Operation: run one episode, tune parameters; evaluate/select where listed. |
| `--params` | `str` | `None` | Parameter JSON; select mode instead reads a tuning checkpoint. |
| `--seed` | `int` | `7` | Episode RNG seed; in prototype tuning, the search RNG seed. |
| `--seeds` | `int × '+'` | `list(range(1, 13))` | Scenario seed list in models; NUMBER of seeds in benchmark run. |
| `--max-steps` | `int` | `1000` | Episode horizon, in environment steps. |
| `--grid-size` | `int` | `15` | Square environment side length in cells. |
| `--danger-speed` | `float` | `0.0` | Danger movement-update budget per environment step; 0 means static. |
| `--visual` | `str; ['on', 'off']` | `'off'` | Enable the live GUI during simulation. |
| `--visual-history` | `int` | `20` | Live history length, in environment steps. |
| `--record` | `str; ['on', 'off']` | `'off'` | Save neural recording; format depends on model version. |
| `--record-path` | `str` | `None` | Run-mode recording filename; enable --record on. |
| `--debug-log` | `str` | `None` | Per-step diagnostic CSV; tuning may interpret this as a log directory. |
| `--audit-path` | `str` | `None` | Export circuit topology/assumptions as JSON in run mode. |
| `--ablation` | `str; ['full', 'no-chemistry']` | `'full'` | Disable specified mechanisms; accepted values are model-specific. |
| `--trials` | `int` | `100` | Number of random-search parameter candidates. |
| `--workers` | `int` | `1` | Parallel worker count; model tune and benchmark run are separate controls. |
| `--progress-interval` | `float` | `30.0` | Progress-message interval, in wall-clock seconds. |
| `--search-seed` | `int` | `20261012` | Random-search RNG seed, separate from scenario seeds. |
| `--output` | `str` | `'rings_C_results.json'` | Output file or directory, depending on the script/subcommand. |


### `rings_benchmark.py`

#### Global

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--version` | `flag` | `None` | Print script version and exit. |


#### run

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | `str` | `required` | Python model script imported by each benchmark worker. |
| `--output` | `str` | `required` | Output file or directory, depending on the script/subcommand. |
| `--params` | `str` | `None` | Parameter JSON; select mode instead reads a tuning checkpoint. |
| `--ablation` | `str; ('full', 'no-state-recurrence', 'no-chemistry', 'both')` | `'full'` | Disable specified mechanisms; accepted values are model-specific. |
| `--workers` | `int` | `30` | Parallel worker count; model tune and benchmark run are separate controls. |
| `--seeds` | `int` | `100` | Scenario seed list in models; NUMBER of seeds in benchmark run. |
| `--repeats` | `int` | `10` | Episodes per environment seed, with separate agent seeds. |
| `--start-seed` | `int` | `1` | First environment/scenario seed. |
| `--master-seed` | `int` | `20261005` | Seed controlling benchmark agent-seed generation. |
| `--max-steps` | `int` | `1000` | Episode horizon, in environment steps. |
| `--grid-size` | `int` | `15` | Square environment side length in cells. |
| `--danger-speed` | `float` | `0.0` | Danger movement-update budget per environment step; 0 means static. |
| `--timeout` | `float` | `3600` | Wall-clock timeout per benchmark episode, seconds. |
| `--resume` | `flag` | `False` | Resume the compatible output/checkpoint rather than start again. |
| `--minigrid-compat` | `flag` | `False` | Encode custom Food/Danger objects as coloured MiniGrid balls. |
| `--bootstrap` | `int` | `5000` | Number of bootstrap resamples over environment-seed means. |


#### report

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `folder` | `str` | `required` | Result directory containing manifest and episode data. |
| `--bootstrap` | `int` | `5000` | Number of bootstrap resamples over environment-seed means. |


#### compare

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `folders` | `str × '+'` | `required` | Two or more result directories; the first is the baseline. |
| `--output` | `str` | `required` | Output file or directory, depending on the script/subcommand. |
| `--bootstrap` | `int` | `5000` | Number of bootstrap resamples over environment-seed means. |
| `--labels` | `str × '+'` | `None` | One label per comparison folder, in the same order. |
| `--allow-different-settings` | `flag` | `False` | Allow setting differences except max_steps; still requires common complete seeds. |


#### _worker (internal)

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `request` | `str` | `required` | Internal worker request JSON; not a public user workflow. |


### `rings_neural_video.py`

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `recording` | `str` | `required` | Input NPZ recording, not a CSV log. |
| `--output` | `str` | `required` | Output file or directory, depending on the script/subcommand. |
| `--fps` | `float` | `30` | Output frames per second. |
| `--frames-per-step` | `int` | `3` | Video frames per environment step; limited by recorded temporal resolution. |
| `--start-step` | `int` | `0` | First rendered step, inclusive, zero-based. |
| `--end-step` | `int` | `None` | Stop before this step; omitted means all completed steps. |
| `--preview` | `str` | `None` | Optional new PNG preview filename. |
| `--preview-step` | `int` | `None` | Step for preview; omitted means the first selected step. |
| `--font` | `str` | `None` | TTF font with Cyrillic support; auto-detected when omitted. |


### `rings_neural_video_attr.py`

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--version` | `flag` | `None` | Print script version and exit. |
| `recording` | `str` | `required` | Input NPZ recording, not a CSV log. |
| `--output` | `str` | `required` | Output file or directory, depending on the script/subcommand. |
| `--fps` | `float` | `30` | Output frames per second. |
| `--frames-per-step` | `int` | `3` | Video frames per environment step; limited by recorded temporal resolution. |
| `--start-step` | `int` | `0` | First rendered step, inclusive, zero-based. |
| `--end-step` | `int` | `None` | Stop before this step; omitted means all completed steps. |
| `--preview` | `str` | `None` | Optional new PNG preview filename. |
| `--preview-step` | `int` | `None` | Step for preview; omitted means the first selected step. |
| `--phase-tail-ms` | `float` | `600` | Recent PCA trajectory length in Brian2 milliseconds. |
| `--phase-elev` | `float` | `24` | Fixed PCA camera elevation, degrees. |
| `--phase-azim` | `float` | `-60` | Fixed PCA camera azimuth, degrees. |
| `--phase-export` | `str` | `None` | Save PCA mean, basis, explained variance, scores and time to a new NPZ. |
| `--font` | `str` | `None` | TTF font with Cyrillic support; auto-detected when omitted. |


## Model parameter JSON reference

CLI controls the experiment; `RingsParams` controls the circuit. These keys are JSON fields, **not extra CLI flags**. The defaults below are constructor defaults, not tuned values. Current loaders accept a flat parameter object or a tuning payload with `best_params`; inspect the version-specific loader before migrating files. Missing keys may use constructor defaults, but v0.51 additionally requires `g_food` and `selectivity` when loading a file. Unknown v0.51 keys are rejected. Fields ending in `_ms`, `_mv`, `_hz` use ms, mV and Hz respectively; `g_*` are dimensionless increments/scales in the model conductance equations, not SI conductance values.

### `rings_minigrid_brian_2_prototype51.py`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ring_size` | `int` | `24` | Neurons per state ring. |
| `state_rings` | `int` | `4` | Number of state rings (v0.51 requires four). |
| `motor_size` | `int` | `8` | Neurons per motor population. |
| `topology_seed` | `int` | `1701` | Seed of fixed neuronal selectivity/heterogeneity. |
| `snn_step_ms` | `float` | `30.0` | Simulated neural time per environment step, ms. |
| `dt_ms` | `float` | `0.1` | Brian2 integration time step, ms. |
| `sensor_range` | `int` | `3` | Sensor distance range in cells. |
| `membrane_tau_ms` | `float` | `15.0` | Membrane time constant, ms. |
| `excitatory_tau_ms` | `float` | `4.0` | Excitatory conductance decay time, ms. |
| `inhibitory_tau_ms` | `float` | `6.0` | Inhibitory conductance decay time, ms. |
| `ring_delay_ms` | `float` | `5.0` | Within-ring recurrent transmission delay, ms. |
| `adaptation_tau_ms` | `float` | `80.0` | Adaptation decay time, ms. |
| `adaptation_jump_mv` | `float` | `2.0` | Adaptation increment per state-neuron spike, mV. |
| `trace_tau_ms` | `float` | `60.0` | Spike-trace x decay time, ms. |
| `state_tonic_mv` | `float` | `17.0` | Baseline tonic state drive, mV. |
| `tonic_spread_mv` | `float` | `1.5` | Tonic-drive heterogeneity amplitude, mV. |
| `receptor_drive_mv` | `float` | `50.0` | Receptor neuron drive scale, mV. |
| `receptor_half` | `float` | `0.12` | Half-saturation parameter for receptor input. |
| `satiety_midpoint` | `float` | `0.95` | Energy midpoint of the satiety signal. |
| `mod_tau_ms` | `float` | `40.0` | Secretory-neuron membrane time constant, ms. |
| `mod_refractory_ms` | `float` | `15.0` | Secretory-neuron refractory time, ms. |
| `foraging_tau_ms` | `float` | `450.0` | Foraging concentration decay time, ms. |
| `defense_tau_ms` | `float` | `300.0` | Defense concentration decay time, ms. |
| `fatigue_tau_ms` | `float` | `600.0` | Fatigue concentration decay time, ms. |
| `curiosity_tau_ms` | `float` | `600.0` | Curiosity concentration decay time, ms. |
| `foraging_release` | `float` | `0.12` | Foraging secretion increment per secretory spike. |
| `defense_release` | `float` | `0.14` | Defense secretion increment per secretory spike. |
| `fatigue_release` | `float` | `0.05` | Fatigue secretion increment per secretory spike. |
| `curiosity_release` | `float` | `0.14` | Curiosity secretion increment per secretory spike. |
| `receptor_kd` | `float` | `0.25` | Chemical receptor saturation constant. |
| `curiosity_tonic_mv` | `float` | `18.0` | Tonic drive of the curiosity secretory neuron, mV. |
| `foraging_drive_mv` | `float` | `8.0` | Foraging chemical modulation drive scale, mV. |
| `defense_drive_mv` | `float` | `14.0` | Defense chemical modulation drive scale, mV. |
| `fatigue_drive_mv` | `float` | `12.0` | Fatigue chemical modulation drive scale, mV. |
| `curiosity_drive_mv` | `float` | `6.0` | Curiosity chemical modulation drive scale, mV. |
| `g_ring` | `float` | `1.1` | Nearest-neighbour recurrent excitation scale. |
| `g_support` | `float` | `0.25` | Second-neighbour recurrent excitation scale. |
| `g_cross` | `float` | `0.15` | Cross-ring recurrent excitation scale. |
| `g_state_to_inh` | `float` | `0.18` | State-to-feedback-interneuron conductance. |
| `g_state_inhibition` | `float` | `0.35` | Feedback inhibition of state neurons. |
| `g_food` | `float` | `4.0` | Food afferent excitation of state. |
| `g_threat` | `float` | `3.0` | Danger afferent excitation of state. |
| `g_body` | `float` | `1.2` | Internal-body afferent excitation of state. |
| `g_proprioception` | `float` | `1.5` | Wall/effort/turn-feedback afferent gain. |
| `g_food_competition` | `float` | `3.0` | Food-induced inhibitory contrast between directions. |
| `g_threat_rest_inhibition` | `float` | `5.0` | Threat-rest inhibitory pathway scale. |
| `defense_gate_mv` | `float` | `90.0` | Defense chemical drive into threat-rest gating, mV. |
| `selectivity` | `float` | `4.0` | Sharpness of directional tuning profiles. |
| `g_afferent_inhibition` | `float` | `2.0` | Input-triggered inhibition of state. |
| `g_interneuron_in` | `float` | `1.8` | Afferent excitation of routing interneurons. |
| `g_readout` | `float` | `5.5` | State-to-motor readout excitation scale. |
| `g_pool_in` | `float` | `0.025` | State excitation of the common motor inhibitory pool. |
| `g_pool_out` | `float` | `0.8` | Common-pool inhibition of motor neurons. |
| `g_competition_in` | `float` | `0.4` | Motor excitation of competition interneurons. |
| `g_competition_out` | `float` | `0.6` | Cross-motor inhibition via competition interneurons. |
| `g_afferent_mod` | `float` | `1.0` | Sensory excitation of modulator secretion. |
| `g_effort_mod` | `float` | `0.55` | Effort excitation of fatigue secretion. |
| `g_mod_inhibition` | `float` | `1.0` | Inhibition of modulator secretion. |
| `noise_rate_hz` | `float` | `40.0` | Poisson source firing rate, Hz. |
| `noise_g` | `float` | `0.08` | Noise synaptic conductance increment. |


### `rings_model_R.py` / `rings_model_C.py`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sensor_range` | `int` | `3` | Sensor distance range in cells. |
| `snn_step_ms` | `float` | `30.0` | Simulated neural time per environment step, ms. |
| `dt_ms` | `float` | `0.1` | Brian2 integration time step, ms. |
| `membrane_tau_ms` | `float` | `3.0` | Membrane time constant, ms. |
| `synapse_tau_ms` | `float` | `2.0` | Fast synaptic current decay time, ms. |
| `satiety_midpoint` | `float` | `0.95` | Energy midpoint of the satiety signal. |
| `receptor_drive_mv` | `float` | `50.0` | Receptor neuron drive scale, mV. |
| `receptor_half` | `float` | `0.12` | Half-saturation parameter for receptor input. |
| `food_weight` | `float` | `90.0` | Food feedforward synaptic weight scale. |
| `threat_weight` | `float` | `100.0` | Threat feedforward synaptic weight scale. |
| `wall_weight` | `float` | `100.0` | Wall feedforward synaptic weight scale. |
| `hunger_weight` | `float` | `50.0` | Hunger feedforward synaptic weight scale. |
| `satiety_weight` | `float` | `20.0` | Satiety feedforward synaptic weight scale. |
| `rest_bias_mv` | `float` | `17.0` | Baseline rest-neuron drive, mV. |
| `foraging_drive_mv` | `float` | `15.0` | Foraging chemical modulation drive scale, mV. |
| `defense_drive_mv` | `float` | `65.0` | Defense chemical modulation drive scale, mV. |
| `fatigue_drive_mv` | `float` | `24.0` | Fatigue chemical modulation drive scale, mV. |
| `curiosity_drive_mv` | `float` | `20.0` | Curiosity chemical modulation drive scale, mV. |
| `receptor_kd` | `float` | `0.25` | Chemical receptor saturation constant. |
| `foraging_tau_ms` | `float` | `450.0` | Foraging concentration decay time, ms. |
| `defense_tau_ms` | `float` | `300.0` | Defense concentration decay time, ms. |
| `fatigue_tau_ms` | `float` | `600.0` | Fatigue concentration decay time, ms. |
| `curiosity_tau_ms` | `float` | `600.0` | Curiosity concentration decay time, ms. |
| `foraging_release` | `float` | `0.12` | Foraging secretion increment per secretory spike. |
| `defense_release` | `float` | `0.14` | Defense secretion increment per secretory spike. |
| `fatigue_release` | `float` | `0.05` | Fatigue secretion increment per secretory spike. |
| `curiosity_release` | `float` | `0.14` | Curiosity secretion increment per secretory spike. |


R and C share the same parameter schema; chemical fields do not activate chemistry in R. Validators enforce additional constraints (for example R/C sensor_range=3 and fast neural time constants shorter than the control window); accepting JSON syntax does not imply that the parameter set is valid.

### `rings_minigrid_brian_2_prototype43.py` 

Archived model implementation — '0.43.0'.

| Option | Type / choices | Default | Meaning |
| --- | --- | --- | --- |
| `--self-test` | `flag` | `False` | Run built-in circuit checks, not a population benchmark. |
| `--mode` | `str; ['run', 'tune']` | `'run'` | Operation: run one episode, tune parameters; evaluate/select where listed. |
| `--seed` | `int` | `7` | Episode RNG seed; in prototype tuning, the search RNG seed. |
| `--trials` | `int` | `50` | Number of random-search parameter candidates. |
| `--workers` | `int` | `32` | Parallel worker count; model tune and benchmark run are separate controls. |
| `--num-seeds` | `int` | `3` | Number of consecutive tuning scenarios. |
| `--start-seed` | `int` | `1` | First environment/scenario seed. |
| `--seeds` | `int × '+'` | `None` | Scenario seed list in models; NUMBER of seeds in benchmark run. |
| `--progress-interval` | `float` | `5.0` | Progress-message interval, in wall-clock seconds. |
| `--resume` | `flag` | `False` | Resume the compatible output/checkpoint rather than start again. |
| `--max-steps` | `int` | `250` | Episode horizon, in environment steps. |
| `--params` | `str` | `None` | Parameter JSON; select mode instead reads a tuning checkpoint. |
| `--output` | `str` | `'rings_v043_best.json'` | Output file or directory, depending on the script/subcommand. |
| `--visual` | `str; ['on', 'off']` | `'off'` | Enable the live GUI during simulation. |
| `--record` | `str; ['on', 'off']` | `'off'` | Save neural recording; format depends on model version. |
| `--video` | `str; ['on', 'off']` | `'off'` | Save video during run; independent from NPZ recording. |
| `--video-path` | `str` | `'rings_v043.mp4'` | Output video filename for simulation or replay. |
| `--video-fps` | `float` | `10.0` | Video frames per second. |
| `--replay-model` | `str` | `None` | Original v39 model source for legacy NPZ replay; snapshot-based replay does not need it. |
| `--replay` | `str` | `None` | Replay an existing recording instead of simulating an episode. |
| `--audit-path` | `str` | `None` | Export circuit topology/assumptions as JSON in run mode. |
| `--record-path` | `str` | `None` | Run-mode recording filename; enable --record on. |
| `--debug-log` | `str` | `None` | Per-step diagnostic CSV; tuning may interpret this as a log directory. |
| `--grid-size` | `int` | `15` | Square environment side length in cells. |
| `--danger-speed` | `float` | `0.0` | Danger movement-update budget per environment step; 0 means static. |


**RingsParams — all declared fields**

| JSON key | Type | Default |
| --- | --- | --- |
| `ring_size` | `int` | `12` |
| `snn_step_ms` | `float` | `30.0` |
| `dt_ms` | `float` | `0.1` |
| `sensor_range` | `int` | `3` |
| `membrane_tau_ms` | `float` | `15.0` |
| `excitatory_tau_ms` | `float` | `4.0` |
| `inhibitory_tau_ms` | `float` | `6.0` |
| `ring_delay_ms` | `float` | `5.0` |
| `adaptation_tau_ms` | `float` | `60.0` |
| `adaptation_jump_mv` | `float` | `3.0` |
| `receptor_drive_mv` | `float` | `50.0` |
| `receptor_half` | `float` | `0.12` |
| `satiety_midpoint` | `float` | `0.95` |
| `mod_tau_ms` | `float` | `40.0` |
| `mod_refractory_ms` | `float` | `15.0` |
| `foraging_tau_ms` | `float` | `450.0` |
| `defense_tau_ms` | `float` | `240.0` |
| `fatigue_tau_ms` | `float` | `600.0` |
| `curiosity_tau_ms` | `float` | `600.0` |
| `foraging_release` | `float` | `0.12` |
| `defense_release` | `float` | `0.14` |
| `fatigue_release` | `float` | `0.05` |
| `curiosity_release` | `float` | `0.14` |
| `receptor_kd` | `float` | `0.25` |
| `curiosity_tonic_mv` | `float` | `18.0` |
| `foraging_drive_mv` | `float` | `30.0` |
| `curiosity_drive_mv` | `float` | `32.0` |
| `fatigue_inhibition_mv` | `float` | `12.0` |
| `fatigue_rest_drive_mv` | `float` | `20.0` |
| `defense_rest_inhibition_mv` | `float` | `18.0` |
| `defense_escape_drive_mv` | `float` | `70.0` |
| `chemical_defense_inhibition_mv` | `float` | `20.0` |
| `foraging_gain` | `float` | `0.8` |
| `defense_gain` | `float` | `0.8` |
| `turn_exploration_ratio` | `float` | `1.0` |
| `g_ring` | `float` | `2.0` |
| `g_support` | `float` | `0.4` |
| `escape_recurrence_ratio` | `float` | `0.7` |
| `g_food` | `float` | `1.3` |
| `g_danger` | `float` | `1.8` |
| `g_escape_motor` | `float` | `1.6` |
| `g_lateral_escape` | `float` | `0.8` |
| `g_afferent_mod` | `float` | `1.0` |
| `g_effort_mod` | `float` | `0.55` |
| `g_motor_mod` | `float` | `0.3` |
| `g_interneuron_in` | `float` | `1.8` |
| `g_competition_in` | `float` | `1.5` |
| `g_competition_out` | `float` | `0.35` |
| `g_safety_inhibition` | `float` | `3.0` |
| `g_satiety_inhibition` | `float` | `1.8` |
| `g_completion_inhibition` | `float` | `1.0` |
| `g_event_inhibition` | `float` | `0.5` |
| `noise_rate_hz` | `float` | `40.0` |
| `noise_g` | `float` | `0.12` |



