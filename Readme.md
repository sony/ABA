# 🚀 Adaptive Budget Optimization for Multichannel Advertising Using Combinatorial Bandits (AAMAS 2025 - Extended Abstract)

Welcome to the **Adaptive Budget Optimization** repository! This project focuses on simulating long-running ad campaigns and implementing the **TUCBMAE algorithm** to optimize budget allocation efficiently. 📊📈
It is the official implementation of the paper by Briti Gangopadhyay, Zhao Wang, Alberto Silvia Chiappa and Takamatsu Shingo.

---
## 📌 System Requirements

The code has been successfully tested on:

- **🖥 Ubuntu 20.04.2 LTS**

---
## 📂 Datasets

This repository includes **two real-world advertising datasets**, each with three sub-campaigns run on the **Google Ads platform**:

1. 📡 **Attendance System**
2. 🌐 **Internet Service Provider**

🔹 Special thanks to **Sony Biz Networks Corporation** for providing this dataset from their services **NURO Biz** and **AKASHI**.

🔐 **Privacy Note:** Click and cost values have been **randomly projected** to maintain predictive power while ensuring that original values cannot be reconstructed. **The Criteo dataset results are fully reproducible.**

---
## ⚙️ Installation & Dependencies

Follow these steps to set up the required environment:

```sh
conda create --name tucbmae python=3.8
conda activate tucbmae
python -m pip install -r requirements.txt
pip install -e .
```
---
## 🔧 Hyperparameter Configuration

Configure hyperparameters in `configpolicy.py`. Key parameters include:

🔹 **Simulation Parameters:**
- `num_month` → Number of months for simulation (e.g., `16` for AI Prediction, `1` for Criteo).

🔹 **Data Paths:**
- `data_google` → Path to Google campaign data (e.g., `data/attendance_campaign_data.csv`).
- `data_smn` → Path to another platform’s data.
- `data_criterio` → Path to Criteo dataset (e.g., `data/criterio_data_filtered.csv`).

🔹 **Experiment Settings:**
- `use_wandb` → Enable logging results to **Weights & Biases**.
- `baseline_name` → Use `GPUCBBaseline_Policy` (GP-based) or `Human` (manual allocation).
- `exploration_strategy` → Choose `ucb` (UCB exploration) or `ts` (Thompson Sampling).
- `adaptation_strategy` → Select from `discounted_reward`, `sliding_window`, `no_change_detection`, `mae_test`.
- `use_psudo_conversion` → Use **pseudo conversions** as reward if `True`, else use **clicks**.

---
## 🎮 Simulation Environment

🔹 **Key Features:**
- 📅 **Episodes** → Each episode represents **one month** of ad campaign simulation.
- 💰 **Actions** → Budget allocation for campaigns.
- 🎯 **Rewards** → Clicks or pseudo conversions.
- 📊 **Observations** → Performance data (cost, conversion rates, history, etc.).
- 🗃 **Data Format:**
  - `date`, `campaign_name`, `cost`, `click`, `ctv`, `vtv` (view-through conversions, if available).
- ⚙ **Environment Configurations:**
  - `Tp = 20` (future window size & stationary days).
  - `change_threshold = 1`, `budget_granularity = 500`.

🔹 **Cost Control & Reward Modeling:**
- 🏦 **Monthly Budget Setting** → Based on actual total costs consumed.
- 📈 **Reward Function** → Modeled as a function of cost vs. clicks or pseudo conversions.
- ⚠ **Cost Control** → Uses Google Ads-style budgeting rules.

---
## 🔄 Running Experiments

### 🚀 Running the TUCBMAE Algorithm

```sh
python main.py
```

### 🏁 Running Baseline Comparisons

```sh
python eval_baseline.py
```

---

### License

This project is licensed under the Attribution-NonCommercial 4.0 International License.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes

### Cite

If you use or reference ABA or the provided dataset, please cite us with the following BibTeX entry:

```bibtex
@article{gangopadhyay2025adaptive,
  title={Adaptive Budget Optimization for Multichannel Advertising Using Combinatorial Bandits},
  author={Gangopadhyay, Briti and Wang, Zhao and Chiappa, Alberto Silvio and Takamatsu, Shingo},
  journal={arXiv preprint arXiv:2502.02920},
  year={2025}
}


📜 **For more details, refer to our research paper or reach out to briti.gangopadhyay@sony.com for assistance!** ✉️
