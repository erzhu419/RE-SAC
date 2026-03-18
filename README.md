# RE-SAC: Regularized Ensemble Soft Actor-Critic

This repository contains the code and experiment results for the RE-SAC algorithm, a regularized ensemble extension of the Soft Actor-Critic (SAC) algorithm for bus fleet control.

## Environment Setup

```bash
# Create conda environment
conda create -n RE-SAC python=3.10 -y
conda activate RE-SAC

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Project Structure

```
RE-SAC/
├── sac_ensemble_original_logging.py   # Core RE-SAC training script
├── normalization.py                   # State/reward normalization utilities
├── env_original/                      # Bus simulation environment
│   ├── sim.py                         # Main environment class
│   ├── bus.py, route.py, station.py   # Environment components
│   └── data/                          # Bus route data (xlsx)
├── scripts/                           # Training launch scripts
│   ├── run_ensemble.sh                # Train single ensemble (default: 10)
│   ├── run_all_experiments.sh         # Run all experiments in parallel
│   ├── run_aleatoric_only.sh          # Ablation: aleatoric uncertainty only
│   └── run_epistemic_only.sh          # Ablation: epistemic uncertainty only
├── plots/                             # Visualization scripts
│   ├── plot_all_experiments.py        # Reward comparison across methods
│   └── plot_all_q_values.py           # Q-value estimation comparison
└── results/                           # Pre-computed experiment logs (.npy)
    ├── baseline_sac/                  # Vanilla SAC baseline
    ├── ensemble_{2,5,10,20,40}/       # Different ensemble sizes
    ├── aleatoric_only/                # Ablation study
    ├── epistemic_only/                # Ablation study
    ├── dsac_v1/                       # DSAC-v1 baseline
    └── bac/                           # BAC baseline
```

## Training

Train a single ensemble (default size 10):
```bash
bash scripts/run_ensemble.sh          # Ensemble size = 10
bash scripts/run_ensemble.sh 20       # Ensemble size = 20
```

Run all experiments in parallel:
```bash
bash scripts/run_all_experiments.sh
```

## Plotting Results

Generate comparison plots from pre-computed results:
```bash
python plots/plot_all_experiments.py   # Reward curves
python plots/plot_all_q_values.py      # Q-value analysis
```

Output plots are saved to `results/`.

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--ensemble_size` | 10 | Number of critics in the ensemble |
| `--weight_reg` | 0.01 | Weight regularization coefficient |
| `--beta` | -2 | LCB coefficient (negative = pessimistic) |
| `--beta_ood` | 0.01 | OOD consensus penalty |
| `--beta_bc` | 0.001 | Behavior cloning loss weight |
| `--maximum_alpha` | 0.6 | Max entropy coefficient |
| `--critic_actor_ratio` | 2 | Critic/actor update ratio |
| `--batch_size` | 2048 | Training batch size |
| `--max_episodes` | 500 | Maximum training episodes |
