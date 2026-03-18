# Exploiting Partial Observability in Slime Volleyball with Deep RL

**CSC 480 — Artificial Intelligence**
**Cal Poly San Luis Obispo — Instructor: Rodrigo Canaan**

**Team:** Javier Medina Bueno, Michael Man, Venkata G. Ande, Thomas Hagos, Antony Tartakovskiy


---

## Project Overview

This project investigates whether a reinforcement learning agent with full
visibility can exploit the blind spots of an opponent that operates under a
"fog of war."  We build on the
[SlimeVolleyGym](https://github.com/hardmaru/slimevolleygym) environment (by
David Ha) and train three agents using
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO:

| Agent | Description |
|---|---|
| **Agent A** (Baseline) | Built-in 120-parameter recurrent network provided by SlimeVolleyGym |
| **Agent B** (FrameStack PPO) | PPO trained for 10 M steps under partial observations (masked env) |
| **Agent C** (Exploiter PPO) | PPO trained for 10 M steps with full visibility against Agent B |

---

## Credits and External Resources

| Resource | Use |
|---|---|
| [SlimeVolleyGym](https://github.com/hardmaru/slimevolleygym) by David Ha | Base game environment and built-in baseline agent |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | PPO implementation and vectorized environment utilities |
| [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | RecurrentPPO (LSTM policy) |
| [Gymnasium](https://gymnasium.farama.org) | Environment API |

---

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <this-repo-url>
cd slimevolleygym
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -e .
pip install stable-baselines3[extra]
pip install sb3-contrib
pip install gymnasium pygame
```

> **Python version:** 3.9 or 3.10 recommended (tested on 3.10).

---

## Usage

```bash
python test_state.py
```

---

### Train the agents

#### Agent B — FrameStack PPO (10 M steps, ~6 hours on CPU)

```bash
python train_framestack.py
```

The trained model is saved to `logs_framestack/run_<timestamp>/ppo_framestack_slimevolley.zip`.

#### Agent C — Exploiter PPO (5 M steps by default)

```bash
# Default: 5 M steps, 4 parallel envs, auto device
python train_exploiter.py

# Custom settings
python train_exploiter.py \
    --steps 5000000 \
    --n-envs 4 \
    --device cpu \
    --agent-b logs_framestack/run_20260223_202559/ppo_framestack_slimevolley.zip

# Resume from a checkpoint
python train_exploiter.py --resume-from logs_exploiter/run_<id>/checkpoints/ppo_exploiter_3000000_steps.zip

# Use a preset opponent (2m / 5m / 10m steps of Agent B)
python train_exploiter.py --opponent 10m
```

#### LSTM Agent (5 M steps)

```bash
python train_lstm.py
```

---

### Watch a trained agent play

```bash
# Agent B (FrameStack) vs baseline
python test_framestack.py

# Agent C (Exploiter) vs Agent B — change episodes with --episodes
python test_exploiter.py \
    --agent-c logs_exploiter/run_20260303_204533/ppo_exploiter_slimevolley.zip \
    --opponent 10m \
    --episodes 5

# LSTM agent
python test_lstm.py
```

---

### Evaluate a model (headless, saves CSV)

```bash
# Agent B vs baseline, 200 episodes, save to CSV
python model_eval.py \
    --model framestack \
    --path logs_framestack/run_20260223_202559/ppo_framestack_slimevolley.zip \
    --episodes 200 \
    --save results.csv

# Agent C (exploiter) vs Agent B
python model_eval.py \
    --model exploiter \
    --path logs_exploiter/run_20260303_204533/ppo_exploiter_slimevolley.zip \
    --opponent 10m \
    --episodes 200 \
    --save results_exploiter.csv

# LSTM agent vs baseline
python model_eval.py \
    --model lstm \
    --path logs_lstm/ppo_lstm_slimevolley.zip \
    --episodes 200

# Baseline RNN against itself
python model_eval.py --model baseline --episodes 200
```

---

### Reproduce the round-robin evaluation

This reproduces the three-way match-up in Table 3 of the report
(200 games per match-up, fixed seed 721):

```bash
python eval_exploiter.py \
    --agent-c logs_exploiter/run_20260303_204533/ppo_exploiter_slimevolley.zip \
    --agent-b logs_framestack/run_20260223_202559/ppo_framestack_slimevolley.zip \
    --trials 200 \
    --seed 721 \
    --output results_roundrobin.csv
```

Results are written to `results_roundrobin.csv`.  Add `--render` to watch games.

---

### Analyze training curves

```bash
python analyze_training.py
```

Reads TensorBoard event files from `logs_framestack/`, `logs_lstm/`, and
`logs_exploiter/` and outputs `milestone_summary.csv`, `delta_analysis.csv`,
`learning_curves.png`, `bar_chart_milestones.png`, and `delta_analysis.png`.

---

## Results

### Agent B vs Agent A — 200 episodes (`results.csv`)

| Metric | Value |
|---|---|
| Wins / Losses / Draws | 0 / 0 / 200 |
| Avg reward | −0.98 ± 1.15 |
| Avg points scored | 0.34 |
| Avg points conceded | 1.32 |
| Score margin | −0.98 |
| Avg episode length | 3000 steps |

All 200 episodes end at the time limit (draws).  Agent B consistently
concedes more points than it scores; the score margin of −0.98 means Agent B
loses roughly one life more than Agent A per game on average.

---

### Round-robin evaluation — 200 games per match-up (`results_roundrobin.csv`)

Entries show the mean score ± std for the **right-side (first-listed) agent**.
A positive mean means the right-side agent wins more points on average.

| Match-up | Mean score | Std | Win% | Loss% | Draw% |
|---|---|---|---|---|---|
| C (right) vs B (left) | −0.84 | 1.34 | 12.0 | 57.0 | 31.0 |
| A (right) vs B (left) | +0.07 | 1.10 | 28.5 | 26.5 | 45.0 |
| C (right) vs A (left) | −1.95 | 1.58 |  6.5 | 82.5 | 11.0 |

Agent C failed to exploit Agent B's blind spots: it scores −0.84 against B,
while Agent A scores +0.07 — a gap of −0.91 in favor of A.
Agent C also lost heavily to Agent A (−1.95), indicating it overfitted to
Agent B's style during training.

---

### Training milestone rewards (`milestone_summary.csv`)

Mean training reward at 2 M, 5 M, and 10 M steps (measured against the
respective training opponent in TensorBoard logs):

| Agent | 2 M steps | 5 M steps | 10 M steps |
|---|---|---|---|
| FrameStack PPO (Agent B) | −3.70 | −1.88 | −1.72 |
| LSTM | −4.85 | −4.81 | — |
| Exploiter PPO (Agent C) | −4.80 | −0.80 | −0.21 |

---

## Project File Structure

```
slimevolleygym/
├── slimevolleygym/              # Core environment package (David Ha)
│   ├── slimevolley.py           # Main physics / game logic
│   ├── slimevolley_mask.py      # SlimeVolleyMasked-v0 (our fog-of-war env)
│   ├── mlp.py                   # Built-in neural network (Agent A)
│   └── rendering.py             # Pygame rendering helpers
│
├── train_framestack.py          # Train Agent B (FrameStack PPO, 10 M steps)
├── train_exploiter.py           # Train Agent C (Exploiter PPO, 5 M steps)
├── train_lstm.py                # Train LSTM agent (RecurrentPPO, 5 M steps)
│
├── eval_exploiter.py            # Three-way round-robin evaluation → CSV
├── model_eval.py                # Single-model evaluation with detailed metrics
├── analyze_training.py          # Plot and compare training curves
│
├── test_state.py                # Human vs baseline (interactive)
├── test_framestack.py           # Visual test — Agent B
├── test_exploiter.py            # Visual test — Agent C
├── test_lstm.py                 # Visual test — LSTM agent
│
├── logs_framestack/             # Agent B training logs and checkpoints
├── logs_exploiter/              # Agent C training logs and checkpoints
├── logs_lstm/                   # LSTM agent training logs and checkpoints
├── zoo/                         # Pre-trained models from original SlimeVolleyGym
│
├── results.csv                  # Agent B vs Agent A evaluation (200 episodes)
├── results_roundrobin.csv       # Three-way round-robin results
├── milestone_summary.csv        # Training rewards at 2M / 5M / 10M steps
├── delta_analysis.csv           # Step-to-step improvement analysis
│
├── learning_curves.png          # Training curve plot
├── bar_chart_milestones.png     # Milestone comparison bar chart
├── delta_analysis.png           # Per-interval improvement chart
│
├── report.tex                   # Academic report (LaTeX)
├── bibliography.bib             # BibTeX references
└── training_scripts/            # Legacy SB2 scripts (not used in this project)
```

---

## Compiling the LaTeX Report

The references require a full BibTeX build sequence:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Running only `pdflatex report.tex` once will leave all citations as `[?]`.

---

## TensorBoard

```bash
tensorboard --logdir logs_framestack    # Agent B training curves
tensorboard --logdir logs_exploiter     # Agent C training curves
tensorboard --logdir logs_lstm          # LSTM training curves
```
