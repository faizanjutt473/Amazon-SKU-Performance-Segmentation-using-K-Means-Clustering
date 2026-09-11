"""
================================================================================
 Amazon Product / SKU Segmentation using K-Means (Unsupervised Machine Learning)
================================================================================
Author : Muhammad Faizan Farooq
Dataset: 128 SKU (Amazon product performance data)

Goal
----
Group 128 Amazon products into meaningful segments (e.g. "Star Performers",
"Low Competition Opportunities", "High Price - Low Sales", etc.) using ONLY
the numeric performance columns of the dataset:

    BSR (Best Seller Rank), Revenue, Review.velocity, Sales, Price, FBA.Fees

No labels/targets are used anywhere -> this is pure unsupervised learning.

Pipeline
--------
1. Load & inspect data
2. Clean missing values
3. Handle skewness (log-transform heavily skewed columns)
4. Scale features (StandardScaler)
5. Find the best K using Elbow Method + Silhouette Score
6. Fit final K-Means model
7. Visualize clusters (PCA 2D plot, boxplots, bar charts) with matplotlib
8. Profile & interpret each cluster
9. Export the final labeled dataset
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# 0. GLOBAL SETTINGS
# --------------------------------------------------------------------------
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
sns.set_style("whitegrid")

DATA_PATH = "data.xlsx"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

FEATURES = ["BSR", "Revenue", "Review.velocity", "Sales", "Price", "FBA.Fees"]
SKEWED_FEATURES = ["BSR", "Revenue", "Sales"]  # log-transformed (all strictly positive)
# NOTE: Review.velocity can be negative (a declining review rate), so it is
# NOT log-transformed - it is only standard-scaled like the other features.


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.columns.tolist())
    return df


# --------------------------------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("\nMissing values before cleaning:")
    print(df[FEATURES].isnull().sum())

    # Price / FBA.Fees have missing values -> impute with median
    # (median is robust to the heavy right-skew / outliers seen in this data)
    for col in ["Price", "FBA.Fees"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Drop exact duplicate SKUs if any
    before = len(df)
    df = df.drop_duplicates(subset=["ASIN"])
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate ASIN rows")

    print("\nMissing values after cleaning:")
    print(df[FEATURES].isnull().sum())
    return df


# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (log-transform + scale)
# --------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    X = df[FEATURES].copy()

    # BSR/Revenue/Sales/Review.velocity are heavily right-skewed
    # (a few best-sellers dominate). log1p compresses that range so
    # K-Means distance isn't dominated by a handful of huge values.
    for col in SKEWED_FEATURES:
        X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=df.index)
    return X_scaled, scaler


# --------------------------------------------------------------------------
# 4. FIND OPTIMAL K  (Elbow Method + Silhouette Score)
# --------------------------------------------------------------------------
def find_optimal_k(X_scaled: pd.DataFrame, k_range=range(2, 10)):
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # --- Plot: Elbow curve + Silhouette score side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(list(k_range), inertias, marker="o", color="#2E86AB", linewidth=2)
    axes[0].set_title("Elbow Method (Inertia vs K)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_xticks(list(k_range))

    axes[1].plot(list(k_range), silhouettes, marker="o", color="#A23B72", linewidth=2)
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_xticks(list(k_range))

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    axes[1].axvline(best_k, color="green", linestyle="--", alpha=0.6,
                     label=f"Best K = {best_k}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_elbow_and_silhouette.png", dpi=150)
    plt.close()

    print(f"\nSilhouette scores by K: "
          f"{dict(zip(k_range, np.round(silhouettes, 3)))}")
    print(f"==> Best K chosen (highest silhouette score): {best_k}")
    return best_k


# --------------------------------------------------------------------------
# 5. FIT FINAL K-MEANS MODEL
# --------------------------------------------------------------------------
def fit_kmeans(X_scaled: pd.DataFrame, k: int):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"\nFinal K-Means fitted with K={k} | Silhouette Score={score:.3f}")
    return km, labels


# --------------------------------------------------------------------------
# 6. VISUALIZATIONS
# --------------------------------------------------------------------------
def plot_pca_clusters(X_scaled: pd.DataFrame, labels: np.ndarray, k: int):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    plt.figure(figsize=(9, 6.5))
    palette = sns.color_palette("Set2", k)
    for cluster_id in range(k):
        mask = labels == cluster_id
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    s=70, alpha=0.8, color=palette[cluster_id],
                    edgecolor="white", linewidth=0.6,
                    label=f"Cluster {cluster_id}")

    plt.title(f"Product Clusters (PCA 2D Projection - {var_explained:.1f}% variance explained)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_pca_clusters.png", dpi=150)
    plt.close()


def plot_feature_boxplots(df: pd.DataFrame, k: int):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", k)

    for i, col in enumerate(FEATURES):
        sns.boxplot(data=df, x="Cluster", y=col, hue="Cluster", ax=axes[i],
                    palette=palette, legend=False)
        axes[i].set_title(f"{col} by Cluster")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_feature_boxplots_by_cluster.png", dpi=150)
    plt.close()


def plot_cluster_profile(df: pd.DataFrame, k: int):
    profile = df.groupby("Cluster")[FEATURES].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

    plt.figure(figsize=(10, 6))
    profile_norm.T.plot(kind="bar", ax=plt.gca(),
                         color=sns.color_palette("Set2", k))
    plt.title("Cluster Profile - Normalized Average Feature Values")
    plt.ylabel("Normalized value (0-1)")
    plt.xlabel("Feature")
    plt.xticks(rotation=20)
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_cluster_profile.png", dpi=150)
    plt.close()

    return profile


def plot_cluster_sizes(df: pd.DataFrame, k: int):
    counts = df["Cluster"].value_counts().sort_index()
    palette = sns.color_palette("Set2", k)

    plt.figure(figsize=(7, 5.5))
    bars = plt.bar(counts.index.astype(str), counts.values, color=palette,
                    edgecolor="white")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val),
                  ha="center", fontweight="bold")

    plt.title("Number of Products per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of SKUs")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_cluster_sizes.png", dpi=150)
    plt.close()


def plot_price_vs_sales(df: pd.DataFrame, k: int):
    plt.figure(figsize=(9, 6.5))
    palette = sns.color_palette("Set2", k)
    for cluster_id in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cluster_id]
        plt.scatter(sub["Price"], sub["Sales"], s=70, alpha=0.8,
                    color=palette[cluster_id], edgecolor="white",
                    linewidth=0.6, label=f"Cluster {cluster_id}")

    plt.title("Price vs Sales, Colored by Cluster")
    plt.xlabel("Price ($)")
    plt.ylabel("Sales (units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_price_vs_sales.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 7. CLUSTER INTERPRETATION (auto-generated business insight text)
# --------------------------------------------------------------------------
def interpret_clusters(profile: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CLUSTER PROFILES (average values per cluster)")
    print("=" * 70)
    print(profile.round(2))

    print("\n" + "=" * 70)
    print("AUTO-GENERATED INSIGHTS")
    print("=" * 70)
    overall_mean = profile.mean()
    for cluster_id, row in profile.iterrows():
        tags = []
        if row["Revenue"] > overall_mean["Revenue"]:
            tags.append("High Revenue")
        else:
            tags.append("Low Revenue")
        if row["BSR"] < overall_mean["BSR"]:
            tags.append("Strong Best-Seller Rank")
        else:
            tags.append("Weak Best-Seller Rank")
        if row["Price"] > overall_mean["Price"]:
            tags.append("Premium Priced")
        else:
            tags.append("Budget Priced")
        if row["Review.velocity"] > overall_mean["Review.velocity"]:
            tags.append("High Review Velocity")
        print(f"Cluster {int(cluster_id)}: {', '.join(tags)}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X_scaled, scaler = prepare_features(df)

    best_k = find_optimal_k(X_scaled)
    model, labels = fit_kmeans(X_scaled, best_k)

    df["Cluster"] = labels

    plot_pca_clusters(X_scaled, labels, best_k)
    plot_feature_boxplots(df, best_k)
    profile = plot_cluster_profile(df, best_k)
    plot_cluster_sizes(df, best_k)
    plot_price_vs_sales(df, best_k)

    interpret_clusters(profile)

    out_path = f"{OUTPUT_DIR}/clustered_products.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\nFinal clustered dataset saved to: {out_path}")
    print(f"All charts saved inside: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
