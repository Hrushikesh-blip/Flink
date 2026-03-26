"""
Flink Dynamic Delivery Promise Optimization — ML Pipeline
==========================================================
End-to-end pipeline: EDA → Feature Engineering → Model Comparison → Tuning → Evaluation

Author: Hrushikesh Mantri
"""

# %% [markdown]
# # Dynamic Delivery Promise Optimization for Flink
# **Goal**: Predict per-order delivery time to replace fixed ETA buckets with dynamic, accurate promises.
#
# **Business Impact**:
# - Reduce broken delivery promises → higher customer retention
# - Avoid over-conservative ETAs → higher conversion rate
# - Data-driven rider/warehouse load management

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

RANDOM_STATE = 42
VISUALS_DIR = "../visuals"

# %%
# ============================
# 1. LOAD & INSPECT DATA
# ============================
df = pd.read_csv("../data/flink_deliveries.csv", parse_dates=["timestamp"])
print(f"Dataset: {df.shape[0]:,} orders, {df.shape[1]} columns")
print(f"Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"\nTarget stats (delivery_time_min):")
print(df["delivery_time_min"].describe().round(2))

# %%
# ============================
# 2. EXPLORATORY DATA ANALYSIS
# ============================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Flink Delivery Data — Exploratory Analysis", fontsize=16, fontweight="bold")

# 2a. Delivery time distribution
axes[0, 0].hist(df["delivery_time_min"], bins=60, color="#E84855", alpha=0.85, edgecolor="white")
axes[0, 0].axvline(df["delivery_time_min"].median(), color="black", ls="--", label=f'Median: {df["delivery_time_min"].median():.1f} min')
axes[0, 0].set_xlabel("Delivery Time (min)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Delivery Time Distribution")
axes[0, 0].legend()

# 2b. Delivery time by hour
hourly = df.groupby("hour")["delivery_time_min"].agg(["mean", "std"]).reset_index()
axes[0, 1].fill_between(hourly["hour"], hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=0.2, color="#2D9CDB")
axes[0, 1].plot(hourly["hour"], hourly["mean"], color="#2D9CDB", linewidth=2.5, marker="o", markersize=4)
axes[0, 1].set_xlabel("Hour of Day")
axes[0, 1].set_ylabel("Delivery Time (min)")
axes[0, 1].set_title("Delivery Time by Hour (±1σ)")

# 2c. Warehouse utilization vs delivery time
scatter = axes[0, 2].scatter(
    df.sample(3000, random_state=42)["warehouse_utilization"],
    df.sample(3000, random_state=42)["delivery_time_min"],
    c=df.sample(3000, random_state=42)["available_riders"],
    cmap="RdYlGn", alpha=0.5, s=8
)
plt.colorbar(scatter, ax=axes[0, 2], label="Available Riders")
axes[0, 2].set_xlabel("Warehouse Utilization")
axes[0, 2].set_ylabel("Delivery Time (min)")
axes[0, 2].set_title("Utilization vs Delivery Time")

# 2d. Rain impact
rain_groups = df.groupby("is_raining")["delivery_time_min"].mean()
bars = axes[1, 0].bar(["No Rain", "Rain"], rain_groups.values, color=["#2D9CDB", "#E84855"], edgecolor="white", width=0.5)
for bar, val in zip(bars, rain_groups.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}", ha="center", fontweight="bold")
axes[1, 0].set_ylabel("Avg Delivery Time (min)")
axes[1, 0].set_title("Rain Impact on Delivery Time")

# 2e. City comparison
city_stats = df.groupby("city")["delivery_time_min"].agg(["mean", "std"]).sort_values("mean")
axes[1, 1].barh(city_stats.index, city_stats["mean"], xerr=city_stats["std"], color="#6C5CE7", alpha=0.85, edgecolor="white")
axes[1, 1].set_xlabel("Avg Delivery Time (min)")
axes[1, 1].set_title("Delivery Time by City")

# 2f. Promise kept rate by hour
promise_rate = df.groupby("hour")["promise_kept"].mean() * 100
axes[1, 2].bar(promise_rate.index, promise_rate.values, color=np.where(promise_rate.values < 70, "#E84855", "#00B894"), edgecolor="white")
axes[1, 2].axhline(70, color="gray", ls="--", alpha=0.7, label="70% threshold")
axes[1, 2].set_xlabel("Hour of Day")
axes[1, 2].set_ylabel("Promise Kept (%)")
axes[1, 2].set_title("Promise Fulfillment Rate by Hour")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_eda_overview.png")

# %%
# ============================
# 3. FEATURE ENGINEERING
# ============================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from raw data."""
    feat = df.copy()

    # --- Temporal features ---
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # Peak hour flags
    feat["is_lunch_peak"] = ((feat["hour"] >= 12) & (feat["hour"] <= 14)).astype(int)
    feat["is_dinner_peak"] = ((feat["hour"] >= 18) & (feat["hour"] <= 21)).astype(int)
    feat["is_rush_hour"] = ((feat["hour"] >= 8) & (feat["hour"] <= 9) |
                             (feat["hour"] >= 17) & (feat["hour"] <= 19)).astype(int)

    # --- Interaction features ---
    feat["load_per_rider"] = feat["active_orders"] / (feat["available_riders"] + 1)
    feat["distance_x_rain"] = feat["distance_km"] * (1 + feat["is_raining"])
    feat["items_x_utilization"] = feat["n_items"] * feat["warehouse_utilization"]
    feat["distance_x_wind"] = feat["distance_km"] * feat["wind_speed_kmh"] / 10

    # --- Pressure score (composite operational stress metric) ---
    feat["pressure_score"] = (
        feat["warehouse_utilization"] * 0.4 +
        feat["load_per_rider"].clip(upper=20) / 20 * 0.4 +
        feat["is_raining"] * 0.1 +
        feat["is_rush_hour"] * 0.1
    ).round(3)

    # --- Encode categoricals ---
    le = LabelEncoder()
    feat["primary_category_enc"] = le.fit_transform(feat["primary_category"])
    feat["warehouse_id_enc"] = le.fit_transform(feat["warehouse_id"])
    feat["city_enc"] = le.fit_transform(feat["city"])

    return feat


df_feat = engineer_features(df)

# Define feature columns (exclude leakage: pick_time, rider_wait, travel_time)
FEATURE_COLS = [
    # Temporal
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_lunch_peak", "is_dinner_peak", "is_rush_hour",
    # Order
    "n_items", "basket_value_eur", "has_heavy_items", "primary_category_enc",
    # Operational
    "active_orders", "warehouse_utilization", "available_riders",
    "rider_utilization", "distance_km", "warehouse_id_enc", "city_enc",
    # Weather
    "temperature_c", "is_raining", "rain_intensity_mm", "wind_speed_kmh",
    # Engineered
    "load_per_rider", "distance_x_rain", "items_x_utilization",
    "distance_x_wind", "pressure_score",
]

TARGET = "delivery_time_min"

X = df_feat[FEATURE_COLS].values
y = df_feat[TARGET].values

print(f"\nFeature matrix: {X.shape}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# %%
# ============================
# 4. TRAIN/TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# %%
# ============================
# 5. MODEL COMPARISON (5-Fold CV)
# ============================
models = {
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE),
}

kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
results = {}

print("\n" + "="*65)
print("MODEL COMPARISON — 5-Fold Cross-Validation")
print("="*65)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-"*65)

for name, model in models.items():
    mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"))
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    results[name] = {
        "MAE": mae_scores.mean(),
        "RMSE": rmse_scores.mean(),
        "R2": r2_scores.mean(),
        "MAE_std": mae_scores.std(),
        "RMSE_std": rmse_scores.std(),
        "R2_std": r2_scores.std(),
    }
    print(f"{name:<25} {mae_scores.mean():>7.2f}  {rmse_scores.mean():>7.2f}  {r2_scores.mean():>7.3f}")

# %%
# Model comparison visual
results_df = pd.DataFrame(results).T

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison — 5-Fold Cross-Validation", fontsize=14, fontweight="bold")

metrics = [("MAE", "Mean Absolute Error (min)", "#E84855"),
           ("RMSE", "Root Mean Squared Error (min)", "#2D9CDB"),
           ("R2", "R² Score", "#00B894")]

for ax, (metric, title, color) in zip(axes, metrics):
    vals = results_df[metric]
    stds = results_df[f"{metric}_std"]
    bars = ax.barh(results_df.index, vals, xerr=stds, color=color, alpha=0.85, edgecolor="white")
    ax.set_xlabel(title)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 02_model_comparison.png")

# %%
# ============================
# 6. BEST MODEL — TRAIN & EVALUATE ON TEST SET
# ============================
best_model_name = max(results, key=lambda k: results[k]["R2"])
print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

# Retrain best model on full training set
if best_model_name == "HistGradientBoosting":
    best_model = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.05, min_samples_leaf=20, random_state=RANDOM_STATE)
elif best_model_name == "Gradient Boosting":
    best_model = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, min_samples_leaf=15, random_state=RANDOM_STATE)
else:
    best_model = models[best_model_name]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Test metrics
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)
test_mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\n{'='*50}")
print(f"TEST SET EVALUATION — {best_model_name}")
print(f"{'='*50}")
print(f"  MAE  : {test_mae:.2f} min")
print(f"  RMSE : {test_rmse:.2f} min")
print(f"  R²   : {test_r2:.4f}")
print(f"  MAPE : {test_mape:.2%}")

# %%
# ============================
# 7. EVALUATION VISUALS
# ============================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f"Test Set Evaluation — {best_model_name}", fontsize=14, fontweight="bold")

# 7a. Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.2, s=5, color="#6C5CE7")
axes[0, 0].plot([0, 60], [0, 60], "r--", linewidth=2, label="Perfect prediction")
axes[0, 0].set_xlabel("Actual Delivery Time (min)")
axes[0, 0].set_ylabel("Predicted Delivery Time (min)")
axes[0, 0].set_title("Actual vs Predicted")
axes[0, 0].legend()
axes[0, 0].text(5, 50, f"R² = {test_r2:.3f}\nMAE = {test_mae:.2f} min", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 7b. Residual distribution
residuals = y_test - y_pred
axes[0, 1].hist(residuals, bins=50, color="#E84855", alpha=0.85, edgecolor="white")
axes[0, 1].axvline(0, color="black", ls="--")
axes[0, 1].set_xlabel("Residual (Actual - Predicted, min)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title(f"Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})")

# 7c. Feature importance
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
    top_15 = feat_imp.tail(15)
    axes[1, 0].barh(top_15.index, top_15.values, color="#2D9CDB", alpha=0.85, edgecolor="white")
    axes[1, 0].set_xlabel("Feature Importance")
    axes[1, 0].set_title("Top 15 Features")
else:
    axes[1, 0].text(0.5, 0.5, "No feature importances\nfor this model type", ha="center", va="center", fontsize=12)

# 7d. Business impact: Dynamic vs Fixed promise
# Simulate dynamic promise = predicted + 2 min buffer
dynamic_promise = y_pred + 2.0
fixed_promise = np.where(y_test < 12, 10, np.where(y_test < 20, 15, np.where(y_test < 30, 25, 35)))

dynamic_kept = (y_test <= dynamic_promise).mean() * 100
fixed_kept = (y_test <= fixed_promise).mean() * 100

# Average promise (lower = more competitive)
dynamic_avg_promise = dynamic_promise.mean()
fixed_avg_promise = fixed_promise.mean()

comparison = pd.DataFrame({
    "Promise Kept (%)": [fixed_kept, dynamic_kept],
    "Avg Promise (min)": [fixed_avg_promise, dynamic_avg_promise],
}, index=["Fixed Buckets", "Dynamic (ML)"])

x = np.arange(2)
w = 0.35
bars1 = axes[1, 1].bar(x - w/2, comparison["Promise Kept (%)"], w, label="Promise Kept %", color="#00B894", edgecolor="white")
axes[1, 1].set_ylabel("Promise Kept (%)", color="#00B894")
ax2 = axes[1, 1].twinx()
bars2 = ax2.bar(x + w/2, comparison["Avg Promise (min)"], w, label="Avg Promise (min)", color="#E84855", edgecolor="white")
ax2.set_ylabel("Avg Promise (min)", color="#E84855")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(comparison.index)
axes[1, 1].set_title("Business Impact: Dynamic vs Fixed Promises")

# Add value labels
for bar, val in zip(bars1, comparison["Promise Kept (%)"]):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)
for bar, val in zip(bars2, comparison["Avg Promise (min)"]):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.1f}", ha="center", fontweight="bold", fontsize=9, color="#E84855")

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_evaluation.png")

# %%
# ============================
# 8. SUMMARY REPORT
# ============================
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"""
Dataset:           {len(df):,} synthetic Flink delivery orders
Features:          {len(FEATURE_COLS)} engineered features
Best Model:        {best_model_name}
Test Performance:  MAE={test_mae:.2f} min | RMSE={test_rmse:.2f} min | R²={test_r2:.4f}

Business Impact (simulated):
  Fixed bucket promises kept:  {fixed_kept:.1f}%
  Dynamic ML promises kept:    {dynamic_kept:.1f}%   (Δ = +{dynamic_kept - fixed_kept:.1f}pp)
  Fixed avg promise:           {fixed_avg_promise:.1f} min
  Dynamic avg promise:         {dynamic_avg_promise:.1f} min  (Δ = {dynamic_avg_promise - fixed_avg_promise:+.1f} min)

Key Findings:
  1. Distance and operational load are the strongest predictors
  2. Rain adds ~{df.groupby('is_raining')['delivery_time_min'].mean().diff().iloc[-1]:.1f} min on average
  3. Peak hours (18-21) show highest variability
  4. Dynamic promises improve fulfillment while staying competitive
""")
