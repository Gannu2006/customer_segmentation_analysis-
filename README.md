# Customer Personality Analysis & Segmentation 🎯

A machine learning project that performs unsupervised customer segmentation and clustering on customer transaction and demographic data, featuring an interactive browser-based analytics dashboard.

---

## 📌 Project Overview
Customer personality analysis helps businesses understand their customers' behavioral patterns, spending habits, channel engagement, and marketing campaign responsiveness. 

By grouping customers with similar traits, marketing teams can design tailored campaigns, reduce churn, optimize product offerings, and improve ROI.

---

## 🔬 Methodology & Workflow

1. **Data Cleaning & Preprocessing:**
   - Handled missing values (e.g., median imputation for `Income`).
   - Parsed dates and calculated customer tenure in days.
   - Outlier clipping between the 1st and 99th percentiles.
   - Column transformation: `StandardScaler` for continuous numerical features and `OneHotEncoder` for categorical features (`Education`, `Marital_Status`).

2. **Feature Engineering:**
   - **`Age`**: $2014 - \text{Year\_Birth}$
   - **`Total_Spending`**: Aggregate spending across Wines, Fruits, Meat, Fish, Sweets, and Gold.
   - **`Total_Purchases`**: Sum of Web, Catalog, and Store transactions.
   - **`Total_Household_Children`**: Sum of `Kidhome` and `Teenhome`.
   - **`Campaigns_Accepted`**: Total marketing campaigns accepted (Campaigns 1–5).

3. **Clustering & Optimal $k$ Selection:**
   - Evaluated cluster cohesion and separation using **Silhouette Scores** across $k = 2$ through $k = 8$.
   - **$k = 2$** yielded the highest silhouette score (**0.3120**).

4. **Dimensionality Reduction & Visualization:**
   - **Principal Component Analysis (PCA)** reduced the multidimensional feature space into 2 primary components (explaining **52.2%** of variance) for clear 2D cluster visualization.

---

## 📊 Key Findings & Segment Profiles

| Metric | High-Value Customers (Cluster 0) | Regular Customers (Cluster 1) |
| :--- | :--- | :--- |
| **Customer Base** | 903 (**40.3%**) | 1,337 (**59.7%**) |
| **Average Age** | 46.95 years | 44.01 years |
| **Average Income** | **\$71,339.04** | **\$39,337.26** |
| **Average Spending** | **\$1,233.63** (6.8x higher) | **\$181.76** |
| **Top Categories** | Wines (\$607.63), Meat (\$358.63) | Wines (\$98.82), Meat (\$37.49) |
| **Preferred Channels** | Store (8.60) & Catalog (5.31) | Store (3.89) & Web (2.93) |
| **Monthly Web Visits** | 3.75 visits/month | **6.38 visits/month** |
| **Campaign Response** | **22.6%** conversion rate | **9.7%** conversion rate |

### Strategic Recommendations:
- **High-Value Segment:** Introduce VIP concierge services, luxury wine reserve allocations, and personalized direct-mail catalogs.
- **Regular Segment:** Implement cart abandonment incentives, bundle promotions, and entry-level discounts to convert high web browsing frequency into purchases.

---

## 📁 Repository Structure

```text
├── customer_segmentation.py            # Main pipeline script (preprocessing, clustering, PCA, exports)
├── generate_dashboard.py               # Generates the self-contained HTML/JS web dashboard
├── index.html                          # Interactive browser dashboard
├── customer_segmentation.csv           # Original dataset
├── customer_segments_output.csv        # Segmented dataset with Cluster & Segment labels
├── segment_profile.csv                 # Aggregate segment profiles and metrics
├── channel_preferences_by_segment.csv  # Channel purchase patterns per segment
├── campaign_response_by_segment.csv    # Marketing response rates per segment
├── spend_by_segment.csv                # Product category spending per segment
├── kmeans_cluster_selection.png        # Silhouette score curve
├── customer_segmentation_pca.png       # 2D PCA cluster visualization plot
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the segmentation pipeline
```bash
python customer_segmentation.py
```

### 4. View the dashboard in your browser
Open `index.html` directly in any web browser, or serve it via Python:
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080/index.html`.

