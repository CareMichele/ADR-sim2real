# Automatic Domain Randomization – Robot Learning Project

## 🎯 Project Objectives

Implementation and evaluation of **Automatic Domain Randomization (ADR)** as a smarter alternative to Uniform Domain Randomization (UDR) for sim-to-sim transfer learning. We test ADR on MuJoCo Hopper and Ant environments across three difficulty levels (Easy, Medium, Hard) to understand how adaptive randomization improves policy robustness compared to manual tuning.

---

## 📁 Project Structure

```text
project-extension/
├── README.md
├── requirements.txt
├── report_project.ipynb      # Final project report
├── configs/
│   └── hopper.xml            # MuJoCo Hopper configuration
├── src/
│   ├── train.py              # Main ADR training script
│   ├── evaluate.py           # Model evaluation script
│   ├── evaluate_all_models.py # Cross-evaluation script
│   ├── adr_manager.py        # ADR range adaptation logic
│   ├── adr_wrapper.py        # Environment wrapper for ADR
│   ├── envs/
│   │   ├── custom_hopper.py  # Custom Hopper with randomization
│   │   └── custom_ant.py     # Custom Ant with randomization
│   └── utils/
│       └── plotting.py       # Visualization utilities
└── data/
    ├── checkpoints/          # Trained models
    ├── logs/                 # Training histories (JSON)
    ├── imgs/                 # Training plots
    ├── videos/               # Demo videos for report
    └── evaluation_results/   # Cross-evaluation data
```


