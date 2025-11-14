# 🕹️ Deep Q-Network (DQN) for Atari Pong  
### CSCN 8020 — Reinforcement Learning Programming • Assignment 3

This repository contains the implementation, experiments, and analysis for a Deep Q-Network (DQN) agent trained on the **PongDeterministic-v4** Atari environment. The project evaluates the effect of changing **mini-batch size** and **target network update rate**, and reports all required metrics according to the assignment specification.

---

## 📁 Repository Structure

```
📦 DQN-Pong-Assignment3
│
├── assignment3.ipynb
├── assignment3_utils.py
│
├── Results/
│   ├── baseline_b8_t10.csv
│   ├── baseline_b8_t10_score.png
│   ├── baseline_b8_t10_avg5.png
│   ├── batch16_t10.csv
│   ├── batch8_t3.csv
│   └── summary.csv
│
├── New/
│   ├── mini_batch_score_comparison.png
│   ├── mini_batch_avg5_comparison.png
│   ├── target_update_score_comparison.png
│   └── target_update_avg5_comparison.png
│
└── README.md
```

---

## 🧠 Project Overview

The goal of this assignment is to:

- Implement a **Deep Q-Network (DQN)** for Pong  
- Modify the CNN input to use **4 stacked frames (84×80)**  
- Evaluate:
  - Score per episode  
  - Average reward of last 5 episodes  
- Change two parameters independently:
  - Mini-batch size → *8 vs 16*  
  - Target network update rate → *every 3 vs every 10 episodes*  
- Report results and select the **best configuration**

---

## 🛠️ Installation & Dependencies

### Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### Install dependencies
```bash
pip install gymnasium[atari,accept-rom-license]
pip install autorom[accept-rom-license]
pip install torch torchvision numpy matplotlib pandas
```

---

## 🎮 Environment & Preprocessing

### Environment
- `PongDeterministic-v4`
- 6 discrete actions
- Deterministic frame skipping

### Preprocessing (assignment3_utils.py)
- Crop unimportant regions  
- Resize to **84×80**  
- Grayscale conversion  
- Normalize to [-1, +1]  
- Stack **4 frames** → (4, 84, 80)

---

## 🧩 Neural Network Architecture

| Layer | Parameters | Output |
|-------|------------|--------|
| Input | 4 × 84 × 80 | (4, 84, 80) |
| Conv1 | 32 filters, 8×8, stride=4 | (32, 20, 20) |
| Conv2 | 64 filters, 4×4, stride=2 | (64, 9, 9) |
| Conv3 | 64 filters, 3×3, stride=1 | (64, 7, 7) |
| Flatten | — | 3136 |
| FC1 | Dense 512 | 512 |
| FC2 | Dense 6 | Q-values |

Training:
- Loss: Smooth L1  
- Optimizer: Adam (1e-4)  
- Discount: γ = 0.95  
- Replay buffer: 100k  

---

## 🚀 Running the Notebook

```bash
jupyter notebook assignment3.ipynb
```

---

## 📊 Metrics Collected

- **Score per Episode**  
- **Average Reward (Last 5 Episodes)**  

Saved under **Results/**.

---

## 📈 Experiment Results

### Baseline (batch=8, target=10)
- `baseline_b8_t10_score.png`
- `baseline_b8_t10_avg5.png`

### Mini-batch comparison (8 vs 16)
- `mini_batch_score_comparison.png`
- `mini_batch_avg5_comparison.png`

### Target-update comparison (3 vs 10)
- `target_update_score_comparison.png`
- `target_update_avg5_comparison.png`

---

## 🏆 Best Hyperparameter Combination

**Batch Size = 8**  
**Target Network Update Rate = 10 episodes**

Provides the best balance of:
- Fast early learning  
- Stable gradients  
- Smooth performance curves  

---

## 📜 Conclusion

This project demonstrates the successful implementation of DQN with stacked visual input for Pong. Experimental results show that smaller mini-batches and less frequent target updates deliver more stable and efficient learning within a short training budget.
