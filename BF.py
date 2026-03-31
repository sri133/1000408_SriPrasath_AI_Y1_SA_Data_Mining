import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Black Friday Analysis", layout="wide")

st.title("🛍️ Black Friday Sales Data Analysis")

# -------------------------------
# LOAD DATA (FULL DATASET FROM GITHUB)
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath_AI_Y1_SA_Data_Mining/main/black_friday.csv"
    df = pd.read_csv(url, low_memory=False)
    return df

df = load_data()

# -------------------------------
# 🎯 DASHBOARD OVERVIEW
# -------------------------------
st.header("📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Avg Purchase", int(df['Purchase'].mean()))
col3.metric("Total Transactions", len(df))

# -------------------------------
# SHOW FULL DATASET
# -------------------------------
st.header("📂 Full Dataset (All Rows)")
st.write("Total rows:", len(df))
st.dataframe(df, use_container_width=True, height=600)

# -------------------------------
# 🧹 DATA PREPROCESSING
# -------------------------------
st.header("🧹 Data Preprocessing")

st.markdown("""
- Missing values in **Product_Category_2** and **Product_Category_3** were filled with 0.
- These 0 values indicate that the product does not belong to a secondary or tertiary category.
- Converted categorical data where necessary.
- Selected important features for clustering: Age, Occupation, Purchase.
- No rows were removed to preserve full dataset integrity.
""")

# -------------------------------
# 🔎 SEARCH DATA
# -------------------------------
st.subheader("🔎 Search by User ID")

search_id = st.text_input("Enter User ID")

if search_id:
    result = df[df['User_ID'].astype(str).str.contains(search_id)]
    st.write("Results Found:", len(result))
    st.dataframe(result, use_container_width=True)
    
# -------------------------------
# DATA PREPROCESSING
# -------------------------------
st.header("🧹 Data Cleaning")

df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

age_map = {
    '0-17':1, '18-25':2, '26-35':3, '36-45':4,
    '46-50':5, '51-55':6, '55+':7
}
df['Age'] = df['Age'].map(age_map)

df.drop_duplicates(inplace=True)

# Normalize Purchase
scaler = StandardScaler()
df['Purchase_scaled'] = scaler.fit_transform(df[['Purchase']])

st.success("Data cleaned and processed!")

# -------------------------------
# 🔍 CLEAN FILTER SECTION
# -------------------------------
st.sidebar.header("🔍 Filters")

# Create a copy
filtered_df = df.copy()

# -------------------------------
# 🎯 BASIC FILTERS
# -------------------------------
gender = st.sidebar.multiselect(
    "Gender",
    df['Gender'].unique()
)

age = st.sidebar.multiselect(
    "Age Group",
    sorted(df['Age'].dropna().unique())
)

# -------------------------------
# 💰 PURCHASE FILTER
# -------------------------------
min_val, max_val = int(df['Purchase'].min()), int(df['Purchase'].max())

purchase_range = st.sidebar.slider(
    "Purchase Range",
    min_val, max_val,
    (min_val, max_val)
)

# -------------------------------
# 🧑‍💼 DEMOGRAPHIC FILTERS
# -------------------------------
occupation = st.sidebar.multiselect(
    "Occupation",
    sorted(df['Occupation'].unique())
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["All", 0, 1]
)

# -------------------------------
# 🌍 LOCATION FILTERS
# -------------------------------
city = st.sidebar.multiselect(
    "City Category",
    df['City_Category'].unique()
)

stay = st.sidebar.multiselect(
    "Years in Current City",
    df['Stay_In_Current_City_Years'].unique()
)

# -------------------------------
# 🛍️ PRODUCT FILTERS
# -------------------------------
category1 = st.sidebar.multiselect(
    "Product Category 1",
    df['Product_Category_1'].unique()
)

category2 = st.sidebar.multiselect(
    "Product Category 2",
    df['Product_Category_2'].unique()
)

# -------------------------------
# 🔄 APPLY FILTERS
# -------------------------------

if gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]

if age:
    filtered_df = filtered_df[filtered_df['Age'].isin(age)]

filtered_df = filtered_df[
    (filtered_df['Purchase'] >= purchase_range[0]) &
    (filtered_df['Purchase'] <= purchase_range[1])
]

if occupation:
    filtered_df = filtered_df[filtered_df['Occupation'].isin(occupation)]

if marital != "All":
    filtered_df = filtered_df[
        filtered_df['Marital_Status'] == marital
    ]

if city:
    filtered_df = filtered_df[filtered_df['City_Category'].isin(city)]

if stay:
    filtered_df = filtered_df[
        filtered_df['Stay_In_Current_City_Years'].isin(stay)
    ]

if category1:
    filtered_df = filtered_df[
        filtered_df['Product_Category_1'].isin(category1)
    ]

if category2:
    filtered_df = filtered_df[
        filtered_df['Product_Category_2'].isin(category2)
    ]

# -------------------------------
# 🔃 SORTING
# -------------------------------
sort_option = st.sidebar.selectbox(
    "Sort By",
    ["None", "Purchase High → Low", "Purchase Low → High"]
)

if sort_option == "Purchase High → Low":
    filtered_df = filtered_df.sort_values(by="Purchase", ascending=False)

elif sort_option == "Purchase Low → High":
    filtered_df = filtered_df.sort_values(by="Purchase", ascending=True)

# -------------------------------
# 🔄 RESET BUTTON
# -------------------------------
if st.sidebar.button("Reset Filters"):
    filtered_df = df.copy()

# -------------------------------
# 📂 DISPLAY FILTERED DATA
# -------------------------------
st.subheader("📂 Filtered Data (Full View)")
st.write("Total rows:", len(filtered_df))

# Column selector (advanced feature)
columns = st.multiselect(
    "Select Columns to Display",
    df.columns.tolist(),
    default=df.columns.tolist()
)

st.dataframe(filtered_df[columns], use_container_width=True, height=600)

# -------------------------------
# ⬇️ DOWNLOAD DATA FEATURE
# -------------------------------
st.subheader("⬇️ Download Data")

download_option = st.selectbox(
    "Select what to download",
    ["Filtered Data", "Full Dataset", "Top 100 Records"]
)

# -------------------------------
# 📂 SELECT DATASET
# -------------------------------
if download_option == "Filtered Data":
    download_df = filtered_df

elif download_option == "Full Dataset":
    download_df = df

elif download_option == "Top 100 Records":
    download_df = filtered_df.head(100)

# -------------------------------
# 📑 COLUMN SELECTION
# -------------------------------
download_columns = st.multiselect(
    "Select columns to download",
    download_df.columns.tolist(),
    default=download_df.columns.tolist()
)

# -------------------------------
# 📄 CONVERT TO CSV
# -------------------------------
csv = download_df[download_columns].to_csv(index=False).encode('utf-8')

# -------------------------------
# ⬇️ DOWNLOAD BUTTON
# -------------------------------
st.download_button(
    label="📥 Download Selected Data",
    data=csv,
    file_name="custom_data.csv",
    mime="text/csv",
)

# -------------------------------
# EDA
# -------------------------------
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Purchase Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Purchase'], bins=30, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Purchase by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Purchase', data=df, ax=ax)
    st.pyplot(fig)

st.subheader("Product Category Popularity")
fig, ax = plt.subplots()
df['Product_Category_1'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
st.pyplot(fig)

# -------------------------------
# 📊 SCATTER PLOT
# -------------------------------
st.subheader("📊 Purchase vs Occupation")

fig, ax = plt.subplots()
ax.scatter(df['Occupation'], df['Purchase'])
ax.set_xlabel("Occupation")
ax.set_ylabel("Purchase")

st.pyplot(fig)

# -------------------------------
# 💰 AVG PURCHASE PER CATEGORY
# -------------------------------
st.subheader("💰 Average Purchase per Product Category")

avg_purchase = df.groupby('Product_Category_1')['Purchase'].mean()

fig, ax = plt.subplots()
avg_purchase.plot(kind='bar', ax=ax)
ax.set_ylabel("Average Purchase")

st.pyplot(fig)

# -------------------------------
# 📉 ELBOW METHOD
# -------------------------------
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.subheader("📉 Elbow Method for Optimal Clusters")

X = df[['Age', 'Occupation', 'Purchase']]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method")

st.pyplot(fig)

# -------------------------------
# CLUSTERING
# -------------------------------
st.header("🤖 Customer Segmentation")

features = df[['Age', 'Occupation', 'Purchase_scaled']]

k = st.slider("Select number of clusters", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

st.write("Clustered Data (All Rows)")
st.dataframe(df[['User_ID', 'Age', 'Occupation', 'Purchase', 'Cluster']],
             use_container_width=True, height=600)

# Cluster plot
fig, ax = plt.subplots()
ax.scatter(df['Age'], df['Purchase'], c=df['Cluster'])
ax.set_xlabel("Age")
ax.set_ylabel("Purchase")
st.pyplot(fig)

# -------------------------------
# 🧠 CLUSTER INTERPRETATION
# -------------------------------
st.subheader("🧠 Cluster Insights")

cluster_summary = df.groupby('Cluster')['Purchase'].mean().sort_values()

for i, val in enumerate(cluster_summary):
    st.write(f"Cluster {cluster_summary.index[i]} → Avg Purchase: {round(val,2)}")

st.markdown("""
### 🧾 Interpretation:
- Low spend clusters → Budget Buyers  
- Medium spend clusters → Regular Customers  
- High spend clusters → Premium Buyers  
""")

# -------------------------------
# 🧠 CLUSTER LABELING
# -------------------------------
st.subheader("🧠 Customer Segments")

cluster_avg = df.groupby('Cluster')['Purchase'].mean()

labels = {}

for cluster, value in cluster_avg.items():
    if value < cluster_avg.mean():
        labels[cluster] = "Budget Buyers"
    elif value > cluster_avg.mean() * 1.2:
        labels[cluster] = "Premium Buyers"
    else:
        labels[cluster] = "Regular Customers"

df['Segment'] = df['Cluster'].map(labels)

st.dataframe(df[['Cluster', 'Segment']].drop_duplicates())

# -------------------------------
# ASSOCIATION RULES
# -------------------------------
st.header("🛒 Association Rule Mining")

basket = df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']].astype(str)
one_hot = pd.get_dummies(basket)

frequent_items = apriori(one_hot, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)

st.write("Association Rules (Full Table)")
st.dataframe(rules, use_container_width=True, height=600)

# -------------------------------
# 🔗 ASSOCIATION INSIGHTS
# -------------------------------
st.subheader("🔗 Association Insights")

if not rules.empty:
    top_rule = rules.sort_values(by="lift", ascending=False).iloc[0]

    st.write("Top Rule:")
    st.write(f"If {top_rule['antecedents']} → then {top_rule['consequents']}")

    st.write(f"Confidence: {round(top_rule['confidence'],2)}")
    st.write(f"Lift: {round(top_rule['lift'],2)}")

    st.markdown("""
### 💡 Business Use:
- Use this combination for combo offers
- Improve cross-selling strategy
""")
# -------------------------------
# ANOMALY DETECTION
# -------------------------------
st.header("🚨 Anomaly Detection")

Q1 = df['Purchase'].quantile(0.25)
Q3 = df['Purchase'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Purchase'] < (Q1 - 1.5 * IQR)) |
              (df['Purchase'] > (Q3 + 1.5 * IQR))]

st.write("Total anomalies detected:", len(outliers))

st.dataframe(outliers, use_container_width=True, height=600)

# -------------------------------
# 🚨 ANOMALY INSIGHTS
# -------------------------------
st.subheader("🚨 Anomaly Insights")

if len(outliers) > 0:
    avg_purchase = df['Purchase'].mean()
    outlier_avg = outliers['Purchase'].mean()

    st.write(f"Normal Avg Purchase: {round(avg_purchase,2)}")
    st.write(f"High Spender Avg: {round(outlier_avg,2)}")

    st.markdown("""
### 💡 Insight:
- High spenders significantly exceed average purchase
- These customers are valuable and should be targeted with premium offers
""")

# -------------------------------
# FILTER SECTION (FULL DATA FILTERING)
# -------------------------------
# -------------------------------
# 🔍 ADVANCED FILTERS
# -------------------------------
st.sidebar.header("🔍 Advanced Filters")

gender = st.sidebar.multiselect("Select Gender", df['Gender'].unique())
age = st.sidebar.multiselect("Select Age Group", sorted(df['Age'].dropna().unique()))
category = st.sidebar.multiselect("Product Category 1", df['Product_Category_1'].unique())

filtered_df = df.copy()

if gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]

if age:
    filtered_df = filtered_df[filtered_df['Age'].isin(age)]

if category:
    filtered_df = filtered_df[filtered_df['Product_Category_1'].isin(category)]

st.subheader("📂 Filtered Data (Advanced)")
st.write("Total rows:", len(filtered_df))
st.dataframe(filtered_df, use_container_width=True, height=600)

# -------------------------------
# 📌 SMART INSIGHTS
# -------------------------------
st.header("📌 Smart Insights (Auto Generated)")

top_age = df.groupby('Age')['Purchase'].mean().idxmax()
top_category = df['Product_Category_1'].value_counts().idxmax()

st.markdown(f"""
### 🔎 Key Findings:
- 💰 Age group **{top_age}** has the highest average spending.
- 🛍️ Product Category **{top_category}** is the most popular.
- 👥 Customer segmentation reveals multiple buying behaviors.
- 🚨 High spenders detected indicate premium customer segment.

### 📈 Business Recommendations:
- Target **Age {top_age}** with premium offers.
- Bundle products in Category **{top_category}** for cross-selling.
- Focus marketing on high-value customer clusters.
""")
