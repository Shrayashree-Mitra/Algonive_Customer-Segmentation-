import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\algonive\task 1 dataset.csv")

print("=" * 50)
print("     CUSTOMER SEGMENTATION SYSTEM")
print("=" * 50)
print(f"\n Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

print("\n STEP 2: DATA CLEANING & PREPROCESSING")
print("-" * 45)

print(f"Missing values:\n{df.isnull().sum()}")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")

print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe().round(2)}")

# Encode Gender: Female=0, Male=1
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print("\n Gender encoded: Female=0, Male=1")


print("\n EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 45)

print("Gender Distribution:")
print(df['Gender'].value_counts())
print(f"\nAge Range        : {df['Age'].min()} – {df['Age'].max()} yrs")
print(f"Income Range     : {df['Annual Income (k$)'].min()} – {df['Annual Income (k$)'].max()} k$")
print(f"Spending Range   : {df['Spending Score (1-100)'].min()} – {df['Spending Score (1-100)'].max()}")

corr = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']].corr()
print(f"\nCorrelation Matrix:\n{corr.round(3)}")


print("\n FEATURE ENGINEERING")
print("-" * 45)

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Features selected : {features}")
print(" Features scaled using StandardScaler")


print("\n  FINDING OPTIMAL NUMBER OF CLUSTERS")
print("-" * 45)

inertia_vals = []
sil_vals = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_vals.append(km.inertia_)
    sil_vals.append(silhouette_score(X_scaled, km.labels_))

print(f"{'K':<5} {'Inertia':<15} {'Silhouette Score'}")
print("-" * 35)
for k, iner, sil in zip(K_range, inertia_vals, sil_vals):
    print(f"{k:<5} {iner:<15.2f} {sil:.4f}")

best_k = K_range[np.argmax(sil_vals)]
print(f"\n Best k = {best_k}  (Silhouette Score: {max(sil_vals):.4f})")


print(f"\n K-MEANS CLUSTERING  (k={best_k})")
print("-" * 45)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"Cluster Distribution:\n{df['Cluster'].value_counts().sort_index()}")
print(f"\nFinal Silhouette Score: {silhouette_score(X_scaled, df['Cluster']):.4f}")


print("\n CLUSTER PROFILING")
print("-" * 45)

cluster_profile = df.groupby('Cluster').agg(
    Count=('CustomerID', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Income=('Annual Income (k$)', 'mean'),
    Avg_Spending=('Spending Score (1-100)', 'mean'),
    Female_Pct=('Gender_Encoded', lambda x: round((1 - x.mean()) * 100, 1))).round(2)

def label_segment(row):
    if row['Avg_Income'] >= 60 and row['Avg_Spending'] >= 60:
        return 'VIP – High Income High Spenders'
    elif row['Avg_Income'] >= 60 and row['Avg_Spending'] < 40:
        return 'Savers – High Income Low Spenders'
    elif row['Avg_Income'] < 40 and row['Avg_Spending'] >= 60:
        return 'Impulse Buyers – Low Income High Spenders'
    elif row['Avg_Income'] < 40 and row['Avg_Spending'] < 40:
        return 'Budget – Low Income Low Spenders'
    else:
        return 'Average – Moderate Spenders'

cluster_profile['Segment_Label'] = cluster_profile.apply(label_segment, axis=1)
print(cluster_profile.to_string())

label_map = cluster_profile['Segment_Label'].to_dict()
df['Segment_Label'] = df['Cluster'].map(label_map)


print("\n EXPORTING FILES FOR POWER BI")
print("-" * 45)

df_export = df.drop(columns=['Gender_Encoded'])
df_export.to_csv("segmented_customers.csv", index=False)
print(" segmented_customers.csv  : Main file, import this in Power BI")

cluster_profile.reset_index(inplace=True)
cluster_profile.to_csv("cluster_profiles.csv", index=False)
print("cluster_profiles.csv   : Use for KPI cards & tables in Power BI")

elbow_df = pd.DataFrame({
    'K': list(K_range),
    'Inertia': inertia_vals,
    'Silhouette_Score': sil_vals
})
elbow_df.to_csv("elbow_silhouette_data.csv", index=False)
print("elbow_silhouette_data.csv  :Elbow/silhouette chart in Power BI")


print("\n" + "=" * 50)
print("           FINAL SUMMARY")
print("=" * 50)
print(f"Total Customers      : {len(df)}")
print(f"Features Used        : {features}")
print(f"Optimal Clusters (k) : {best_k}")
print(f"Silhouette Score     : {silhouette_score(X_scaled, df['Cluster']):.4f}")
print("\nSegment Labels:")
for _, row in cluster_profile.iterrows():
    print(f"  Cluster {int(row['Cluster'])}: {row['Segment_Label']}  ({int(row['Count'])} customers)")

print("\n Import into Power BI (Get Data -Text/CSV):")
print("   1. segmented_customers.csv")
print("   2. cluster_profiles.csv")
print("   3. elbow_silhouette_data.csv")
print("\n  Done!")