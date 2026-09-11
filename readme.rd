Amazon SKU Performance Segmentation using K-Means Clustering

Unsupervised machine learning project that segments 128 Amazon product SKUs into performance-based clusters — no labeled target variable required. The goal is to help sellers and analysts quickly spot best-sellers, underperformers, and pricing tiers from raw product data.

Overview

Real-world e-commerce data rarely comes pre-labeled. This project shows how K-Means clustering can uncover hidden structure in Amazon product performance metrics (Best Seller Rank, Revenue, Sales, Price, Review Velocity, FBA Fees) and turn raw numbers into actionable product segments.

Dataset

data.xlsx — 128 Amazon SKUs with the following columns:

Column	Description
Category	Product category
BSR	Amazon Best Seller Rank (lower = better)
Revenue	Estimated monthly revenue ($)
Review.velocity	Rate of new reviews
Sales	Units sold
Brand	Product brand
ASIN	Amazon product ID
Price	Listing price ($)
FBA.Fees	Fulfillment by Amazon fee ($)
Product.Details	Product title/description
Approach
Data Cleaning — imputed missing Price / FBA.Fees values with the median; removed duplicate ASINs
Feature Engineering — log-transformed heavily right-skewed columns (BSR, Revenue, Sales) so a handful of extreme best-sellers don't dominate distance calculations, then standardized all features with StandardScaler
Optimal K Selection — compared the Elbow Method and Silhouette Score across K = 2–9, selecting the K with the highest silhouette score
Clustering — fit a final KMeans model on the scaled features
Visualization — PCA 2D projection, per-feature boxplots by cluster, normalized cluster profile chart, cluster size distribution, and a Price vs Sales scatter plot
Interpretation — auto-generated a plain-language profile for each cluster (e.g. "High Revenue, Strong Best-Seller Rank, Premium Priced")
Tech Stack

Python · Pandas · NumPy · Scikit-learn (KMeans, PCA, StandardScaler, Silhouette Score) · Matplotlib · Seaborn

Project Structure
product_segmentation/
├── data.xlsx                      # raw input dataset
├── product_segmentation.py        # full pipeline script
├── outputs/
│   ├── 1_elbow_and_silhouette.png
│   ├── 2_pca_clusters.png
│   ├── 3_feature_boxplots_by_cluster.png
│   ├── 4_cluster_profile.png
│   ├── 5_cluster_sizes.png
│   ├── 6_price_vs_sales.png
│   └── clustered_products.xlsx    # final labeled dataset
└── README.md
How to Run
bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
python product_segmentation.py

All charts and the labeled dataset are written to the outputs/ folder.

Results

Silhouette analysis selected K = 5 as the optimal number of clusters (silhouette score ≈ 0.39). The model identified five distinct product segments:

Cluster	Profile
0	Low revenue, weak BSR, premium priced — underperforming products
1	Low revenue, strong BSR, budget priced — steady low-cost sellers
2	High revenue, strong BSR, premium priced — star performers
3	Low revenue, strong BSR, premium priced — niche/high-margin items
4	Budget priced, very high review velocity — trending/viral products

These segments can guide pricing strategy, inventory prioritization, and marketing focus.

Author

Muhammad Faizan Farooq .
