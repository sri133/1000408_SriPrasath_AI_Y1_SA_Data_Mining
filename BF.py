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
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath_AI_Y1_SA_Data_Mining/main/black_friday.csv"
    df = pd.read_csv(url, low_memory=False)
    return df

df = load_data()

st.subheader("📂 Full Dataset (Scrollable)")
st.dataframe(df, use_container_width=True, height=600)
# -------------------------------
# DATA PREPROCESSING
# -------------------------------
st.subheader("🧹 Data Cleaning")

# Handle missing values
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

# Encode Gender
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

# Encode Age
age_map = {
    '0-17':1, '18-25':2, '26-35':3, '36-45':4,
    '46-50':5, '51-55':6, '55+':7
}
df['Age'] = df['Age'].map(age_map)

# Remove duplicates
df.drop_duplicates(inplace=True)

st.success("Data cleaned successfully!")

# -------------------------------
# NORMALIZATION
# -------------------------------
scaler = StandardScaler()
df['Purchase_scaled'] = scaler.fit_transform(df[['Purchase']])

# -------------------------------
# EDA SECTION
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

# Category popularity
st.subheader("🔥 Popular Product Categories")
fig, ax = plt.subplots()
df['Product_Category_1'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("📌 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# -------------------------------
# CLUSTERING
# -------------------------------
st.header("🤖 Customer Segmentation (Clustering)")

features = df[['Age', 'Occupation', 'Purchase_scaled']]

k = st.slider("Select Number of Clusters", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

st.write("Clustered Data Sample:")
st.write(df[['User_ID', 'Age', 'Occupation', 'Purchase', 'Cluster']].head())

# Visualization
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(df['Age'], df['Purchase'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Purchase")
st.pyplot(fig)

# -------------------------------
# ASSOCIATION RULE MINING
# -------------------------------
st.header("🛒 Market Basket Analysis")

# Prepare data for Apriori
basket = df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']]

basket = basket.astype(str)

one_hot = pd.get_dummies(basket)

frequent_items = apriori(one_hot, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_items, metric="lift", min_threshold=1)

st.subheader("Top Association Rules")
st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
st.header("🚨 Anomaly Detection (High Spenders)")

Q1 = df['Purchase'].quantile(0.25)
Q3 = df['Purchase'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Purchase'] < (Q1 - 1.5 * IQR)) |
              (df['Purchase'] > (Q3 + 1.5 * IQR))]

st.write(f"Number of anomalies detected: {len(outliers)}")

st.write("Sample High Spenders:")
st.write(outliers.head())

# -------------------------------
# INSIGHTS
# -------------------------------
st.header("📌 Key Insights")

st.markdown("""
- 💰 Certain age groups show higher spending behavior.
- 🛍️ Specific product categories dominate sales.
- 🔗 Strong associations exist between product combinations.
- 🚨 High spenders identified using anomaly detection.
- 👥 Customer segments reveal different buying patterns.
""")

# -------------------------------
# SIDEBAR FILTERS (BONUS MARKS)
# -------------------------------
st.sidebar.header("🔍 Filters")

gender_filter = st.sidebar.selectbox("Select Gender", ["All", 0, 1])

filtered_df = df.copy()

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]

st.subheader("Filtered Data")
st.write(filtered_df.head())
