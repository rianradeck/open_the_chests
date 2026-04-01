# Open The Chests

Reproducible training/evaluation runs for Gymnasium environments with a consistent output layout.

Current focus:
- **Stable-Baselines3 (SB3)** pipelines (PPO/SAC)
- **TensorBoard** as the source of truth for training/eval curves

---

## 1) How to use

### Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- SB3 relies on **PyTorch**; depending on your platform, installing it may take time.
- The KUKA environment requires **PyBullet**.

### Train (writes TensorBoard training logs)

Example (SAC on KUKA):

```bash
python3 -m open_the_chests.cli.sb3_train \
  --env-id ColoredChestKuka-v0 \
  --algo sac \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --run-name test_run \
  --max-steps 200 \
  --observation-space extended
```

This command prints:
- the created `run_dir`
- the `results.json` path

Training logs are written under:
- `<run_dir>/tb/train/...`

### Eval (writes TensorBoard eval logs)

Evaluate a saved model and write eval logs into the same run directory:

```bash
python3 -m open_the_chests.cli.sb3_eval \
  --env-id ColoredChestKuka-v0 \
  --model-path <run_dir>/models/final_model.zip \
  --algo sac \
  --episodes 50 \
  --seed 42 \
  --device cpu \
  --run-dir <run_dir> \
  --total-timesteps 100000
```

Eval logs are written under:
- `<run_dir>/tb/eval/...`

### “Plot” (starts TensorBoard)

Instead of generating PNGs, `plot` launches TensorBoard and aggregates the runs you point it to.

Single run:

```bash
python3 -m open_the_chests.cli.plot --run-dir <run_dir>
```

All runs under `runs/`:

```bash
python3 -m open_the_chests.cli.plot --runs-root runs
```

It prints the URL (default `http://127.0.0.1:6006`).

---

## 2) Project structure

High-level layout:

```text
.
├── colored_chest_kuka_env.py        # KUKA env (kept intact); registration happens via import
├── open_the_chests/
│   ├── cli/                        # Entry points (sb3_train, sb3_eval, plot)
│   ├── envs/                       # Env registration + get_env factory
│   ├── frameworks/
│   │   └── sb3/                    # SB3 pipelines
│   ├── utils/                      # run directories, results.json, seeding
│   └── viz/                        # (optional) plotting utilities
├── legacy/                         # notebooks / legacy experiments
└── runs/                           # generated runs (artifacts)
```

Run artifacts (created by train/eval):

```text
runs/<run_name>/<timestamp>_seed<seed>/
├── config.json
├── models/
│   └── final_model.zip
├── results.json
└── tb/
    ├── train/
    └── eval/
```

`results.json` is a small index/summary for that run. TensorBoard contains the curves.

---

## 3) Running new runs (detailed workflow)

### A) Create multiple runs (same config, different seeds)

Example:

```bash
for seed in 0 1 2 3 4; do
  python3 -m open_the_chests.cli.sb3_train \
    --env-id ColoredChestKuka-v0 \
    --algo sac \
    --timesteps 200000 \
    --seed "$seed" \
    --device cpu \
    --run-name kuka_sac_baseline \
    --max-steps 200 \
    --observation-space extended
done
```

You will get one timestamped folder per run under `runs/kuka_sac_baseline/`.

### B) Evaluate each run (optional but recommended)

If you want evaluation logs for each run (not only the train curves), run `sb3_eval` per run.

Example pattern:

```bash
for d in runs/kuka_sac_baseline/*seed*; do
  python3 -m open_the_chests.cli.sb3_eval \
    --env-id ColoredChestKuka-v0 \
    --model-path "$d/models/final_model.zip" \
    --algo sac \
    --episodes 50 \
    --seed 42 \
    --device cpu \
    --run-dir "$d"
done
```

### C) Compare everything in TensorBoard

```bash
python3 -m open_the_chests.cli.plot --runs-root runs
```

TensorBoard will show multiple runs at once.

---

## 4) Contributing

### Adding a new training pipeline (e.g. AgileRL)

Goal: new frameworks should follow the same “contract” so runs are comparable and discoverable.

Recommended steps:

1) Create a new framework package:

```text
open_the_chests/frameworks/agilerl/
  train.py
  eval.py
```

2) Reuse the shared run utilities:
- Use `open_the_chests.utils.runs.create_run_dir(...)` to create run folders.
- Write `config.json` and `results.json` into the run directory.

3) TensorBoard logging conventions:
- Training logs should go to `<run_dir>/tb/train/...`
- Evaluation logs should go to `<run_dir>/tb/eval/...`

4) Add CLIs:

```text
open_the_chests/cli/agilerl_train.py
open_the_chests/cli/agilerl_eval.py
```

5) Use the environment factory:
- Always create environments via `open_the_chests.envs.factory.get_env(...)`

### PR checklist

- `python3 -m open_the_chests.cli.<your_cli> --help` works
- Produces the standard run folder with `models/`, `tb/`, `config.json`, `results.json`
- `python3 -m open_the_chests.cli.plot --runs-root runs` can discover the new runs

---

## Design notes (optional)

This repository also contains some design notes about reward shaping and algorithm choices.
If you want them documented as a separate markdown doc (instead of living in the README), open an issue/PR and we can move them into `docs/`.