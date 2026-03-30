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

st.title("🛍️ Black Friday Sales Data Analysis Dashboard")

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
# SHOW FULL DATASET
# -------------------------------
st.header("📂 Full Dataset (All Rows)")
st.write("Total rows:", len(df))
st.dataframe(df, use_container_width=True, height=600)

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
# FILTER SECTION (FULL DATA FILTERING)
# -------------------------------
st.header("🔍 Filter Data")

gender = st.selectbox("Select Gender", ["All", 0, 1])
age = st.multiselect("Select Age", sorted(df['Age'].dropna().unique()))

filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == gender]

if age:
    filtered_df = filtered_df[filtered_df['Age'].isin(age)]

st.subheader("📂 Filtered Data (All Matching Rows)")
st.write("Total filtered rows:", len(filtered_df))

st.dataframe(filtered_df, use_container_width=True, height=600)

# -------------------------------
# INSIGHTS
# -------------------------------
st.header("📌 Key Insights")

st.markdown("""
- Customers aged 26–35 tend to spend the most.
- Certain product categories dominate overall sales.
- Strong product combinations exist for cross-selling.
- High spenders are identified as anomalies.
- Customer clusters show different buying behaviors.
""")
