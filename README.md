# 🧠 Batch Size & Deep Neural Network Generalization — A Controlled Causal Study
> **Author:** Perumal Marimuthu

---

## 📌 Overview

This project investigates the **causal effect of batch size on neural network generalization performance**, using a rigorous Bayesian causal inference framework. Rather than simply observing a correlation, we design a controlled experiment and use a **Directed Acyclic Graph (DAG)** to reason about confounding, then estimate the **Average Treatment Effect (ATE)** via Bayesian linear regression.

**Research Question:**
> *What is the causal effect of batch size on a neural network's generalization performance, controlling for the size of the training dataset?*

**Key Finding:**
> Large-batch training (512) causes approximately **0.59% lower test accuracy** compared to small-batch training (32), with a posterior probability P(β_T < 0 | data) = **0.998** — strongly consistent with Keskar et al. (2017)'s sharp minima theory.

---

## 🔬 Causal Framework

### Target Estimand: Average Treatment Effect (ATE)

$$\text{ATE} = \mathbb{E}[\text{Test Accuracy}_{\text{large batch}} - \text{Test Accuracy}_{\text{small batch}} \mid \text{Dataset Size}]$$

### Causal DAG

```
        C (Dataset Size)
       /        \
      ↓          ↓
  T (Batch Size) → Y (Test Accuracy)
```

| Variable | Role | Values |
|---|---|---|
| **T** (Treatment) | Batch size | 0 = Small (32), 1 = Large (512) |
| **Y** (Outcome) | Test accuracy | Continuous [0, 1] |
| **C** (Confounder) | Dataset size | 2,000 / 10,000 / 60,000 samples |

By conditioning on **C**, we block the backdoor path and isolate the true causal effect of batch size on accuracy.

---

## 🧪 Experimental Design

| Factor | Levels |
|---|---|
| Batch sizes | Small (32), Large (512) |
| Dataset sizes | 2,000 / 10,000 / 60,000 samples |
| Replicates | 10 runs per configuration |
| **Total models trained** | **60** |

Each model is a 2-layer dense neural network (784 → 128 → 128 → 10) trained on MNIST with Adam optimizer, dropout, and early stopping.

---

## 📊 Key Results

| Parameter | Posterior Mean | 89% HDI |
|---|---|---|
| α (Baseline Accuracy) | 0.9524 | [0.9499, 0.9548] |
| **β_T (Batch Size Effect)** | **−0.0059** | **[−0.0093, −0.0022]** |
| β_C (Dataset Size Effect) | +0.0267 | [+0.0249, +0.0284] |
| σ (Observation Noise) | 0.0072 | [0.0060, 0.0085] |

**Interpretation:** Large-batch training reduces test accuracy by ~0.59 percentage points on average, with strong posterior evidence (P = 0.998) that the effect is negative — consistent with the sharp minima hypothesis.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| **TensorFlow / Keras** | Neural network training (60 models) |
| **PyMC 5** | Bayesian posterior inference (NUTS sampler) |
| **ArviZ** | MCMC diagnostics, trace plots, HDI |
| **NumPy / Pandas** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **scikit-learn** | Data preprocessing (StandardScaler) |
| **SciPy** | Statistical checks (Q-Q plot) |
| **Graphviz** | DAG visualization |

---

## 📁 Project Structure

```
batch-size-generalization-causal-study/
├── Data Science Project.ipynb          # Full analysis notebook (Jupyter/Colab)
├── Data Science Project.pdf            # Written project report
├── Data Science Project.ipynb-Colab.pdf # Notebook PDF export
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Perumal100/batch-size-generalization-causal-study.git
cd batch-size-generalization-causal-study
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook

**Option A — Google Colab (recommended, no setup needed):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `Data Science Project.ipynb` to Colab and run all cells.

**Option B — Local Jupyter:**
```bash
jupyter notebook "Data Science Project.ipynb"
```

> ⚠️ Note: The experiment trains 60 neural networks on MNIST. On CPU it may take 15–30 minutes. GPU-accelerated runtime (e.g., Colab T4) is recommended.

---

## 🔑 Notebook Sections

1. **Introduction** — Research question, estimand, dataset description
2. **Causal Model** — DAG definition, variable roles, backdoor blocking
3. **Statistical Model** — Bayesian linear regression, prior specification
4. **Environment Setup** — Package installation and verification
5. **Model Validation on Simulated Data** — Prior predictive + parameter recovery
6. **Data Collection** — Training 60 neural networks (factorial design)
7. **Posterior Analysis** — Bayesian inference on real experimental data
8. **Model Diagnostics** — R-hat, ESS, trace plots
9. **Posterior Predictive Checks** — Residuals, Q-Q plots
10. **Discussion & Conclusions** — ATE interpretation, theory alignment
11. **Future Work** — Extensions to CNNs, other datasets, interaction terms

---

## 📖 References

1. Keskar, N. S. et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR. https://arxiv.org/abs/1609.04836
2. Gelman, A. et al. (2013). *Bayesian Data Analysis (3rd ed.).* Chapman and Hall/CRC.
3. Kumar, S. et al. (2021). *Revisiting Small Batch Training for Deep Neural Networks.* https://arxiv.org/abs/2009.02358
4. PyMC Development Team (2023). *PyMC: Bayesian Statistical Modeling.* https://www.pymc.io
5. Kumar, R. et al. (2016). *ArviZ: Exploratory Analysis of Bayesian Models.* https://arviz-devs.github.io/arviz/

---

## 📄 License

MIT License
