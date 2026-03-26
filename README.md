# 🚀 Dynamic Delivery Promise Optimization — Flink Quick Commerce

> **Predicting per-order delivery times to replace fixed ETA buckets with intelligent, dynamic promises.**

## The Problem

Quick-commerce platforms like Flink currently use fixed delivery time buckets (e.g., "10 min", "15 min", "25 min") as customer-facing promises. This one-size-fits-all approach leads to:

- **Broken promises** during peak hours → customer churn
- **Over-conservative estimates** during quiet periods → lost conversions
- **No operational feedback loop** between warehouse load and customer expectations

## The Solution

A machine learning pipeline that predicts delivery time for each order based on real-time operational, temporal, and environmental signals — enabling **dynamic, per-order delivery promises**.

## Key Results

| Metric | Value |
|--------|-------|
| **Model** | Gradient Boosting / HistGradientBoosting |
| **MAE** | ~2.5 min |
| **R²** | ~0.88 |
| **Promise fulfillment improvement** | +8-12 pp over fixed buckets |

## Features Used (30 engineered features)

**Temporal**: Cyclical encoding (hour, day-of-week, month), peak-hour flags, weekend indicator

**Operational**: Warehouse utilization, active orders, available riders, rider-to-order ratio, delivery distance

**Weather**: Temperature, rain (binary + intensity), wind speed

**Order**: Item count, basket value, heavy items flag, product category

**Interactions**: Load per rider, distance × rain, items × utilization, composite pressure score

## Project Structure

```
flink-delivery-optimization/
├── data/
│   └── flink_deliveries.csv          # 15K synthetic orders
├── src/
│   ├── generate_data.py              # Synthetic data generator
│   └── pipeline.py                   # Full ML pipeline
├── visuals/
│   ├── 01_eda_overview.png           # Exploratory analysis
│   ├── 02_model_comparison.png       # 5-fold CV results
│   └── 03_evaluation.png             # Test set evaluation & business impact
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
cd src && python generate_data.py

# Run the full pipeline
python pipeline.py
```

## Approach

### 1. Synthetic Data Generation
Realistic delivery data modeled after German quick-commerce operations across 6 warehouses in Berlin, Hamburg, Munich, Cologne, and Düsseldorf. Includes:
- Realistic hourly/weekly demand patterns (lunch & dinner peaks)
- German seasonal weather patterns
- Correlated operational features (warehouse load ↔ rider availability ↔ wait times)

### 2. Feature Engineering
- **Cyclical encoding** for temporal features (avoids discontinuities at midnight/Monday)
- **Interaction features** capturing compound effects (e.g., distance in rain, load per rider)
- **Pressure score**: Composite metric combining utilization, rider load, weather, and peak-hour stress

### 3. Model Selection
5-fold cross-validation across 5 model families:
- Ridge / Lasso (linear baselines)
- Random Forest
- Gradient Boosting
- HistGradientBoosting

### 4. Business Impact Simulation
Compared dynamic ML-based promises (prediction + 2 min buffer) against fixed bucket system on test data, measuring both **promise fulfillment rate** and **average promise competitiveness**.

## Next Steps (Production Roadmap)

1. **Real data integration** — Replace synthetic data with actual Flink order & delivery logs
2. **Real-time inference** — Deploy as a low-latency API behind the checkout flow
3. **Online learning** — Continuously retrain on incoming orders
4. **A/B testing** — Compare dynamic vs. fixed promises on conversion & retention KPIs
5. **Confidence intervals** — Provide prediction intervals, not just point estimates

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn · NumPy

## Author

**Hrushikesh Mantri**
Operations Manager (Shift Lead) @ Flink | M.Sc. Control Theory & Microsystems, Universität Bremen
Former Data Scientist @ DLR (German Aerospace Center)

---

*This project uses synthetic data only. No proprietary Flink data was used.*
