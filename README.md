🛍️ Black Friday Sales Data Analysis App
📌 Project Overview

This project is an interactive data analysis web application built using Streamlit to analyze the Black Friday sales dataset. The application performs data preprocessing, exploratory data analysis (EDA), clustering, association rule mining, anomaly detection, and provides business insights through an intuitive user interface.

The goal of this project is not only to implement data mining techniques but also to demonstrate a clear understanding of the entire data pipeline, from raw data handling to meaningful decision-making insights.

🎯 Objectives
Analyze customer purchase behavior
Identify patterns in product categories and demographics
Segment customers using clustering techniques
Discover product associations for cross-selling
Detect anomalies in purchasing behavior
Provide actionable business insights
📂 Dataset Description

The dataset contains transactional data from Black Friday sales, including:

User_ID – Unique customer identifier
Product_ID – Unique product identifier
Gender – Male (M) / Female (F)
Age – Age group category
Occupation – Customer occupation code
City_Category – City type (A, B, C)
Stay_In_Current_City_Years – Years in current city
Marital_Status – 0 (Single), 1 (Married)
Product_Category_1, 2, 3 – Product category classifications
Purchase – Purchase amount
🧹 Data Preprocessing

Data preprocessing is a crucial step to ensure data quality and usability.

✔ Steps Performed:
Handling Missing Values
Product_Category_2 and Product_Category_3 contained missing values.
These were filled with 0.
Interpretation:
0 indicates that the product does not belong to a secondary or tertiary category.
Encoding Categorical Variables
Gender:
M → 0
F → 1
Age groups converted into numerical scale for analysis.
Duplicate Removal
Duplicate rows were removed to maintain data integrity.
Feature Scaling
Purchase values were normalized using StandardScaler for clustering.
Feature Selection
Selected features for clustering:
Age
Occupation
Purchase
🔍 Filtering System (Interactive Feature)

The app includes a comprehensive filtering system allowing users to dynamically explore data.

Filters Available:
Gender
Age Group
Purchase Range (slider)
Occupation
Marital Status
City Category
Years in Current City
Product Category 1 & 2
Additional Controls:
Sorting (High → Low, Low → High)
Reset Filters
Column Selection

👉 This ensures users can analyze specific subsets of the dataset dynamically.

📊 Exploratory Data Analysis (EDA)

The app provides multiple visualizations to understand the dataset.

✔ Visualizations Included:
Purchase Distribution (Histogram)
Shows how purchase values are distributed
Purchase by Gender (Boxplot)
Compares spending behavior between genders
Product Category Popularity (Bar Chart)
Identifies most purchased product categories
Correlation Heatmap
Displays relationships between numerical features
Scatter Plot (Purchase vs Occupation)
Helps identify trends and patterns
💰 Average Purchase Analysis
Displays average purchase per product category
Helps identify high-value product segments
📉 Elbow Method (Clustering Preparation)

The Elbow Method is used to determine the optimal number of clusters (K).

Plots WCSS (Within Cluster Sum of Squares) vs number of clusters
Helps visually identify the “elbow point”
🤖 Customer Segmentation (Clustering)
Algorithm Used:
K-Means Clustering
Features Used:
Age
Occupation
Scaled Purchase
Output:
Each customer is assigned a cluster label
Visualization:
Scatter plot showing cluster distribution
🧠 Cluster Interpretation

Clusters are analyzed based on average purchase values.

Segments Created:
Budget Buyers → Low spending customers
Regular Customers → متوسط spending customers
Premium Buyers → High spending customers

👉 This provides clear business understanding of customer groups.

🛒 Association Rule Mining
Algorithm Used:
Apriori Algorithm
Purpose:
Identify frequently bought product combinations
Metrics:
Support
Confidence
Lift
Output:
Rules such as:
“If Product A is bought → Product B is likely to be bought”
Business Use:
Product bundling
Cross-selling strategies
Recommendation systems
🚨 Anomaly Detection
Method Used:
Interquartile Range (IQR)
Purpose:
Detect unusual purchase values (outliers)
Insight:
High spenders identified as premium customers
Useful for targeted marketing strategies
📌 Smart Insights (Auto Generated)

The app automatically generates insights such as:

Highest spending age group
Most popular product category
Customer behavior patterns
Identification of high-value customers
Business Recommendations:
Target high-spending age groups
Promote popular categories
Use clustering for personalized marketing
⬇️ Data Download Feature

Users can download data based on their needs:

Options:
Filtered Data
Full Dataset
Top 100 Records
Additional Feature:
Select specific columns before downloading

👉 This enhances usability and real-world application.

🖥️ Technologies Used
Python
Streamlit
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Mlxtend

🏁 Conclusion

This project successfully demonstrates:

End-to-end data analysis pipeline
Strong understanding of preprocessing
Implementation of multiple data mining techniques
Interactive and user-friendly application
Real-world business insights
📈 Final Outcome

✔ All required rubric components implemented
✔ Multiple advanced features added
✔ Strong focus on both technical implementation and business understanding

Credits:

Student Name: Sri Prasath. P

Mentor Name: Arul Jothi

Course: Data Mining

School Name: Jain Vidyalaya IB World School
