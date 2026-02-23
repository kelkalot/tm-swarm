# UNSW-NB15 Intrusion Detection: Collective TM vs Oracle

This directory contains the code to reproduce the intrusion detection experiments detailed in the `tm-swarm` paper, using the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset.

## How to Reproduce the Numbers

We enforce strict NumPy random seeds during data preparation and evaluation to ensure the results are completely deterministic. 

### 1. Download and Prepare the Data
First, download the raw data and run the deterministic binarization script. This script utilizes **Seed 42** during the permutation and sampling process to guarantee identical dataset splits across runs.

```bash
python download_data.py
python prepare_data.py
```

### 2. Establish the Centralized Oracle Baseline
Train a monolithic Tsetlin Machine on the entirety of the prepared training data. To generate the `98.04% ± 0.21%` baseline, we run the script across 5 different initialization/shuffling seeds.

```bash
# Uses seeds: [42, 123, 456, 789, 1024]
python run_oracle.py
```

### 3. Run the Collective Baseline
Train 3 isolated agents on distinct attack subsets, share synthetic knowledge packets, and evaluate them on unseen attacks.

```bash
python run_experiment.py
```
This proves that the agents successfully achieve statistical parity (`~0.989`) with the Oracle simply by exchanging symbolic logic.

### 4. Run the Zero-Shot Onboarding Experiment
Demonstrate how a brand new sensor ("Agent D") with zero training data can achieve immediate operational readiness by absorbing the collective's knowledge.

```bash
python zero_shot_onboarding.py
```
*Expected Result: Agent D reaches ~98.8% accuracy.*

### 5. Run the LLM-Driven Experiment
Process unstructured text summaries using `qwen3:8b` via Ollama to extract noisy boolean features.

```bash
python run_llm_experiment.py
```

### 6. Visualize the Results
Regenerate all charts and graphics found in the root README:

```bash
python visualize_results.py
python visualize_all.py
```
