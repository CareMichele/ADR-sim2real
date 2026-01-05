# Starting code for final course project extension of Robot Learning - 01HFNOV

# Hopper Project – Final Course Project Extension (Robot Learning – 01HFNOV)

This repository contains the final course project extension for **Robot Learning**, focusing on **transfer and hybrid learning methods** on the MuJoCo **Hopper** environment under dynamics mismatch (e.g. mass shift).

---

## 📁 Project Structure

```text
hopper_project/
├── README.md                 # Setup instructions, run commands, results summary
├── requirements.txt          # torch, gym[MuJoCo], sb3, mujoco, casadi (PILQR)
├── configs/                  # Hyperparameter configuration files
│   ├── ppo.yaml
│   ├── mbpo.yaml
│   ├── pilqr_hybrid.yaml
│   └── dr_doraemon.yaml
├── src/                      # Core source code
│   ├── envs/
│   │   ├── hopper_wrapper.py  # Custom Hopper source/target env (mass shift)
│   │   └── dr_wrapper.py      # DORAEMON / ADR domain randomization wrapper
│   ├── agents/
│   │   ├── ppo_agent.py       # PPO agent (Stable-Baselines3)
│   │   ├── mbpo_agent.py      # MBPO implementation / integration
│   │   └── pilqr_hybrid.py    # PILQR hybrid controller (adapted from HybridLearning)
│   ├── utils/
│   │   ├── logger.py          # TensorBoard / Weights & Biases logging
│   │   └── eval.py            # Source & target evaluation metrics
│   └── train.py               # Main training + evaluation loop
├── hybrid_learning/          # MurpheyLab Hybrid Learning repo (submodule/clone)
│   ├── enjoy_hlt.py           # Hybrid controller execution script
│   └── ...                    # Remaining repository files
├── data/                     # MuJoCo assets and saved checkpoints
│   ├── hopper_source.xml
│   ├── hopper_target.xml
│   └── checkpoints/
├── logs/                     # Training logs (TensorBoard)
│   ├── ppo/
│   ├── hybrid/
│   └── mbpo/
├── results/                  # Final results and plots
│   ├── curves_reward.png
│   └── tables_transfer.csv   # Source → target performance gap
└── analysis_hopper.ipynb     # Jupyter notebooks for analysis



Official assignment at [Google Doc](https://docs.google.com/document/d/1XWE2NB-keFvF-EDT_5muoY8gdtZYwJIeo48IMsnr9l0/edit?usp=sharing).


