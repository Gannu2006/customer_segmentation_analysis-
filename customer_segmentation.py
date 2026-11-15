import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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

plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.xlabel("Number of clusters"); plt.ylabel("Silhouette score")
plt.title("K-Means cluster selection"); plt.grid(alpha=.3); plt.show()
best_k = max(scores, key=scores.get)
print("Selected k:", best_k, "Silhouette:", round(scores[best_k], 4))

# Final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(X)

# Segment profiling
df["Total_Spending"] = df[spend_cols].sum(axis=1)
df["Total_Purchases"] = df[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]].sum(axis=1)
profile = df.groupby("Cluster").agg(
    Customers=("ID","count"), AvgAge=("Age","mean"), AvgIncome=("Income","mean"),
    AvgRecency=("Recency","mean"), AvgSpending=("Total_Spending","mean"),
    AvgPurchases=("Total_Purchases","mean"), ResponseRate=("Response","mean")
).round(2)
display(profile)

# Visualize clusters in 2D using PCA
coords = PCA(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(coords[:,0], coords[:,1], c=df["Cluster"], cmap="viridis", alpha=.55, s=18)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.title("Customer segmentation – PCA view"); plt.colorbar(label="Cluster"); plt.show()

# Customer behaviour and product preferences
display(df.groupby("Cluster")[spend_cols].mean().round(2))
display(df.groupby("Cluster")[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]].mean().round(2))
display(df.groupby("Cluster")["Response"].mean().rename("Campaign response rate"))

# Export results
df.to_csv("customer_segments_output.csv", index=False)
profile.to_csv("segment_profile_output.csv")
print("Saved customer_segments_output.csv and segment_profile_output.csv")

print("Final cluster counts:\n", df["Cluster"].value_counts().sort_index())
