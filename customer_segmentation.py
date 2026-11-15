import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

try:
    display
except NameError:
    def display(x):
        print(x)

# Load the Kaggle dataset (download the CSV into this project folder)
df = pd.read_csv("customer_segmentation.csv")
print("Dataset shape:", df.shape)
print(df.head())
print(df.isnull().sum()[df.isnull().sum() > 0])

# Data preparation and feature engineering
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
df["Age"] = 2014 - df["Year_Birth"]
spend_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
df["Total_Spending"] = df[spend_cols].sum(axis=1)
df["Total_Purchases"] = df[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]].sum(axis=1)
df["Total_Household_Children"] = df["Kidhome"] + df["Teenhome"]
df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
df["Campaigns_Accepted"] = df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)

# Do not use ID, campaign acceptance, or Response as clustering inputs.
# Campaign response will be analysed after clustering.
num_features = ["Age","Income","Total_Household_Children","Recency",
    "MntWines","MntFruits","MntMeatProducts","MntFishProducts",
    "MntSweetProducts","MntGoldProds","NumDealsPurchases",
    "NumWebPurchases","NumCatalogPurchases","NumStorePurchases",
    "NumWebVisitsMonth","Total_Spending","Total_Purchases","Customer_Tenure_Days"]
cat_features = ["Education","Marital_Status"]

Xdf = df[num_features + cat_features].copy()
for col in num_features:
    low, high = Xdf[col].quantile([0.01, 0.99])
    Xdf[col] = Xdf[col].clip(low, high)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_features),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
])
X = preprocessor.fit_transform(Xdf)
if hasattr(X, "toarray"):
    X = X.toarray()

# Find a suitable number of clusters using silhouette score
scores = {}
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    scores[k] = silhouette_score(X, labels)

plt.figure(figsize=(8,5))
plt.plot(list(scores.keys()), list(scores.values()), marker="o", color="#2563eb", linewidth=2.5, markersize=8)
plt.xlabel("Number of clusters (k)", fontsize=12)
plt.ylabel("Silhouette score", fontsize=12)
plt.title("K-Means Cluster Selection (Silhouette Analysis)", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("kmeans_cluster_selection.png", dpi=300, bbox_inches="tight")
plt.close()
best_k = max(scores, key=scores.get)
print("Selected k:", best_k, "Silhouette:", round(scores[best_k], 4))

# Final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(X)

# Segment profiling
df["Total_Spending"] = df[spend_cols].sum(axis=1)
df["Total_Purchases"] = df[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]].sum(axis=1)

# Assign readable Segment labels based on spending
mean_spending = df.groupby("Cluster")["Total_Spending"].mean()
high_cluster = mean_spending.idxmax()
df["Segment"] = df["Cluster"].apply(lambda c: "High-Value Customers" if c == high_cluster else "Regular Customers")

profile = df.groupby(["Cluster", "Segment"]).agg(
    Customers=("ID","count"), AvgAge=("Age","mean"), AvgIncome=("Income","mean"),
    AvgRecency=("Recency","mean"), AvgSpending=("Total_Spending","mean"),
    AvgPurchases=("Total_Purchases","mean"), ResponseRate=("Response","mean")
).round(2).reset_index()
display(profile)

# Visualize clusters in 2D using PCA
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
df["PCA1"] = coords[:,0]
df["PCA2"] = coords[:,1]

plt.figure(figsize=(9,6))
scatter = plt.scatter(coords[:,0], coords[:,1], c=df["Cluster"], cmap="viridis", alpha=0.6, s=25)
plt.xlabel("Principal Component 1 (Variance: {:.1f}%)".format(pca.explained_variance_ratio_[0]*100), fontsize=12)
plt.ylabel("Principal Component 2 (Variance: {:.1f}%)".format(pca.explained_variance_ratio_[1]*100), fontsize=12)
plt.title("Customer Segmentation – 2D PCA Cluster View", fontsize=14, fontweight="bold")
plt.colorbar(scatter, label="Cluster")
plt.grid(True, linestyle="--", alpha=0.3)
plt.savefig("customer_segmentation_pca.png", dpi=300, bbox_inches="tight")
plt.close()

# Customer behaviour and product preferences
spend_summary = df.groupby("Segment")[spend_cols].mean().round(2)
channel_summary = df.groupby("Segment")[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth"]].mean().round(2)
campaign_summary = df.groupby("Segment").agg(
    Customers=("ID", "count"),
    ResponseRate=("Response", "mean"),
    AvgCampaignsAccepted=("Campaigns_Accepted", "mean"),
    ComplaintRate=("Complain", "mean")
).round(3)

print("\n--- Spending by Product Category ---")
display(spend_summary)
print("\n--- Channel Preferences ---")
display(channel_summary)
print("\n--- Campaign Response by Segment ---")
display(campaign_summary)

# Export results
df.to_csv("customer_segments_output.csv", index=False)
profile.to_csv("segment_profile.csv", index=False)
spend_summary.to_csv("spend_by_segment.csv")
channel_summary.to_csv("channel_preferences_by_segment.csv")
campaign_summary.to_csv("campaign_response_by_segment.csv")

# Export complete data package for Browser Dashboard
dashboard_data = {
    "silhouette_scores": {int(k): round(float(v), 4) for k, v in scores.items()},
    "best_k": int(best_k),
    "explained_variance": [round(float(v)*100, 2) for v in pca.explained_variance_ratio_],
    "segment_profile": profile.to_dict(orient="records"),
    "spend_summary": spend_summary.reset_index().to_dict(orient="records"),
    "channel_summary": channel_summary.reset_index().to_dict(orient="records"),
    "campaign_summary": campaign_summary.reset_index().to_dict(orient="records"),
    "sample_points": df[["ID", "PCA1", "PCA2", "Cluster", "Segment", "Income", "Total_Spending", "Age", "Response"]].to_dict(orient="records")
}

with open("dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print("\nSaved all outputs:")
print(" - customer_segments_output.csv")
print(" - segment_profile.csv")
print(" - channel_preferences_by_segment.csv")
print(" - campaign_response_by_segment.csv")
print(" - spend_by_segment.csv")
print(" - dashboard_data.json")
print(" - kmeans_cluster_selection.png")
print(" - customer_segmentation_pca.png")
print("\nFinal cluster counts:\n", df["Segment"].value_counts())
