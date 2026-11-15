# Customer Segmentation Using Machine Learning (Kaggle Dataset)

## Dataset source
Kaggle: https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering

The project uses the CSV supplied for this assignment (`customer_segmentation.csv`). It contains 2,240 rows and 29 columns. The uploaded file has 24 missing Income values, which are imputed with the median during preprocessing.

## Project objective
Segment customers using demographic information and purchasing behaviour, understand product and channel preferences, and derive targeted marketing insights.

## Dataset features used
- Demographics: Year_Birth (derived Age), Education, Marital_Status, Income, Kidhome, Teenhome.
- Purchase recency: Recency.
- Product spend: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds.
- Shopping channels: NumWebPurchases, NumCatalogPurchases, NumStorePurchases.
- Engagement and deals: NumWebVisitsMonth, NumDealsPurchases.
- Campaign response fields: AcceptedCmp1–5 and Response are retained for post-clustering analysis, not used as clustering inputs.

## Methodology
1. Load and inspect the dataset.
2. Impute missing Income with the median.
3. Parse Dt_Customer and derive age, total spending, total purchases, household children, customer tenure and number of accepted campaigns.
4. Exclude ID and constant Z_CostContact/Z_Revenue fields. Exclude campaign acceptance and Response from model inputs to avoid target leakage.
5. Clip numerical model inputs at the 1st and 99th percentiles to limit extreme-value influence.
6. Standardize numerical features and one-hot encode Education and Marital_Status.
7. Compare K-Means models for k=2 through k=8 using silhouette score.
8. Fit the best-scoring candidate and profile each cluster.
9. Use PCA to visualize the high-dimensional customer groups in two dimensions.
10. Analyse product preferences, shopping channels and campaign response by cluster.

## Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook.

## Run
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the notebook:
```bash
jupyter notebook Customer_Segmentation_Kaggle.ipynb
```
Or run:
```bash
python customer_segmentation.py
```
Keep `customer_segmentation.csv` in the same folder as the notebook/script.

## Results from the uploaded CSV
- Records: 2,240
- Original columns: 29
- Missing Income values: 24
- Selected number of clusters: 2
- Silhouette score: 0.3120

The cluster profiles and exact group sizes are in `outputs/segment_profile.csv`. Segment names are descriptive interpretations of observed cluster profiles, not ground-truth labels.

## Business insights and recommended actions
Interpret the generated segment profiles:
- High spending and frequent purchases: loyalty rewards, VIP benefits and early access.
- Long recency / fewer purchases: carefully targeted win-back campaigns.
- Strong category preferences: personalized product recommendations.
- High web visits but low web purchases: improve online conversion and product discovery.
- Campaign response differences: tailor future campaigns by observed segment response.

## Limitations
- K-Means is sensitive to scaling, outliers and the chosen feature set.
- PCA is used only for visualization; it does not define the clusters.
- Cluster IDs have no inherent ranking.
- Campaign acceptance is excluded from clustering and examined after clustering to reduce leakage.
- The dataset reflects a particular marketing dataset and time period; findings should not be generalized without validation.

## Conclusion
This project applies unsupervised machine learning to group customers with similar demographic and purchasing characteristics. Cluster profiles and visualizations provide a basis for customer analytics, purchase-pattern analysis and targeted marketing recommendations.
