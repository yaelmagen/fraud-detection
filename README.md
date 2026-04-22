# 🛡️ Fraud Detection — Real-Time Decision System Simulation

A **Streamlit-based fraud decision engine** that simulates a real-time transaction stream, scoring each event with an unsupervised ML ensemble and deterministic rules, providing SHAP-based explainability and human-in-the-loop feedback.

---

## Project Structure

```
fraud-detection/
├── app/
│   └── streamlit_app.py        # UI & simulation orchestration
├── src/
│   ├── feature_store.py        # Lightweight in-memory feature store
│   ├── feature_engine.py       # Point-in-time feature computation
│   ├── models.py               # IF + LOF ensemble (train + save/load)
│   ├── model_registry.py       # Versioned model management
│   ├── scoring.py              # Normalization, weighting, hard rules
│   ├── explainability.py       # SHAP waterfall + human-readable reasons
│   └── drift.py                # Lightweight drift detection
├── models/                     # Versioned model artifacts (pickle + JSON)
│   ├── isolation_forest_v1.pkl
│   ├── lof_v1.pkl
│   ├── preprocessor_v1.pkl
│   ├── scaler_v1.pkl
│   └── metadata_v1.json
├── scripts/
│   └── train_models.py         # Offline training script
├── data/
│   ├── DS_Test_Dataset.xlsx    # Raw dataset
│   ├── retrain_trigger.json    # Drift / retrain signal
│   ├── history.csv             # Auto-generated transaction history
│   └── feedback.csv            # Analyst feedback log
├── Nuvei_fraud.ipynb           # Original EDA & model selection notebook
├── requirements.txt
└── readme.md
```

---

## Ensemble Logic

The system combines **three complementary perspectives** into a unified risk score [0–1]:

| Layer | Weight | Purpose |
|-------|--------|---------|
| **Isolation Forest** | 70% | Detects *global* behavioral outliers — extreme transaction amounts, velocity bursts, seniority-based anomalies. |
| **Local Outlier Factor (LOF)** | 30% | Catches *local density* anomalies — "silent" deviations invisible to the global model (e.g., high-value transactions in rare geo-merchant combinations). |
| **Deterministic Rules** | Override | Hard flags for Impossible Travel, Suspicious Velocity, Same-Amount Bursts, and Broken Records. When triggered, risk floor is set to 0.85 (guaranteed **Block**). |

### Decision Thresholds

| Score Range | Decision | Action |
|-------------|----------|--------|
| < 0.5 | **Approve** | Transaction passes |
| 0.5 – 0.8 | **Review** | Queued for analyst review |
| ≥ 0.8 | **Block** | Immediately flagged/blocked |

---

## Cold Start Problem

New users with no transaction history receive **safe neutral defaults**:

1. **Mathematical Imputation** — `amount_to_avg_ratio = 1.0` (neutral), `time_since_last_payment = 999999` (effectively infinite) to prevent false positives.
2. **Contextual Encoding** — `is_new_user = 1` and `seniority = 0` flags let the model distinguish "no history" from "suspicious deviation."
3. **Non-Historical Rules** — Deterministic indicators (Impossible Travel, Broken Records) provide protection from day one, regardless of behavioral history.

---

## Why Hybrid (Rules + ML)?

| Challenge | Pure ML | Hybrid Approach |
|-----------|---------|-----------------|
| Geographic impossibility | May learn patterns but not 100% reliable | Hard rule → guaranteed 100% catch rate |
| Bot-like same-amount bursts | IF captures extreme cases; LOF can miss if density is uniform | Deterministic count + ML boost |
| Explainability for analysts | SHAP values alone can be opaque | Human-readable text reasons + SHAP |
| New attack vectors | Requires retraining | Rule layer can be updated instantly |
| Nuanced silent anomalies | IF can miss local deviations | LOF provides a complementary "local lens" |

The combination ensures **no single blind spot**: rules handle clear logical violations with zero latency, while as ML ensemble captures statistical anomalies that no finite rule set could enumerate.

---

## Quick Start

```bash
# 1. Create virtual environment (Python 3.10+ required)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (pre-trained models included)
streamlit run app/streamlit_app.py
```

**Note**: Pre-trained model files are included in the `models/` directory, so no training is required for initial setup. The SHAP explainer is built dynamically from the test set at app startup, matching the notebook's approach.

---

## Model Lifecycle

```
┌─────────────┐       ┌──────────────┐       ┌─────────────────┐
│  Notebook /  │ train │   models/    │ load  │   Streamlit     │
│  train_      │──────▶│  *_v1.pkl    │──────▶│   App           │
│  models.py   │       │  metadata    │       │  (inference)    │
└─────────────┘       └──────────────┘       └─────────────────┘
```

1. **Offline training** — Models are trained in the notebook or via `python scripts/train_models.py`. The last 100 rows are held out, then the remaining data is split 65/35 (train/test) chronologically.
2. **Versioned persistence** — Each training run saves a new version (`v1`, `v2`, …) under `models/`. Existing versions are never overwritten. Each version includes: Isolation Forest, LOF, preprocessor, scaler, and a `metadata_vN.json` with the feature list and calibrator bounds.
3. **Fast inference** — On app startup, the latest pretrained version is loaded from disk via `joblib`. No training occurs at launch.
4. **Manual retraining** — The sidebar **🔄 Retrain Models** button trains a new version on warm-up data, saves it, and hot-reloads it into the session.
5. **Drift detection** — A lightweight heuristic (`src/drift.py`) compares the live stream's average amount and top-5 country distribution against the training baseline every 10 transactions. When drift is detected, `data/retrain_trigger.json` is updated and a warning appears in the sidebar.

---

## Real-Time Simulation

The app simulates real-time scoring with a **holdout-based split** that mirrors the notebook:

1. **Last 100 rows** — Held out as the "live stream" for simulation.
2. **Remaining data** — Split chronologically into **65% train / 35% test**.
   - **Train set** — Used to fit the ensemble (Isolation Forest + LOF) and bootstrap the feature store.
   - **Test set** — Used as background data for the SHAP explainer, matching the notebook's approach.
3. **Feature store** — Bootstrapped with all pre-holdout data (train + test) so the simulation starts with full user history context.

Each click of **Process Next** takes the next chronological holdout row, computes point-in-time features from the feature store (no leakage), scores it through the pretrained ensemble, and appends it to history.

---

## Feature Store

The lightweight feature store (`src/feature_store.py`) maintains:

- **Raw layer** — full transaction history (CSV-backed `DataFrame`)
- **Aggregate layer** — per-user in-memory cache (`dict`) for O(1) lookups

### Update Pattern

```
1. Retrieve user state from cache
2. Compute point-in-time features (no leakage)
3. Score transaction via ensemble
4. Update cache incrementally (no full recompute)
5. Append to history
```

Cached user-level aggregates: `txn_count`, `amount_avg`, `amount_sum`, `last_timestamp`, `last_country/state/region`, and a 24h rolling-window amount list.

---

## Design Choices

| Decision | Rationale |
|----------|-----------|
| **Pretrained models (no auto-train on start)** | Mirrors production: training is expensive and should be deliberate. App startup is fast (~2 s for model load vs ~30 s for training). |
| **Manual retraining** | In a demo context, retraining should be an explicit analyst action so the user can observe version increments and drift signals. |
| **Versioned artifacts** | Enables rollback, A/B comparison, and audit trails — standard MLOps practice. |
| **Holdout split (last 100 / 65-35 train-test)** | Matches notebook exactly. Test set provides SHAP background, holdout simulates unseen live traffic. |
| **Drift detection as a heuristic** | Full statistical tests (KS, PSI) are overkill for a simulation. A simple mean-shift + distribution-change check demonstrates the concept cleanly. |
