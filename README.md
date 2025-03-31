# Customer Conversion Analysis for Online Shopping Using Clickstream Data     

## Overview
This project leverages clickstream data to predict customer conversion, estimate potential revenue, and segment customers—all within an interactive Streamlit web application. By combining robust data pipelines with state-of-the-art machine learning, the solution enables e-commerce businesses to make informed, data-driven decisions.

## Key Features
- **Data Preprocessing & Cleaning:** Handle missing values, encode categorical features, and scale numerical data.
- **Exploratory Data Analysis (EDA):** Visualize distributions, session patterns, and feature relationships.
- **Feature Engineering:** Derive session metrics and behavioral insights from clickstream data.
- **Machine Learning Models:**
  - **Classification:** Predict customer conversion (purchase vs. non-purchase).
  - **Regression:** Forecast potential revenue per customer.
  - **Clustering:** Segment customers for targeted marketing.
- **Pipeline Orchestration:** ZenML is used to manage end-to-end pipeline workflows.
- **Experiment Tracking:** MLflow logs parameters, models, and metrics for reproducibility.
- **Interactive Deployment:** Streamlit provides a user-friendly interface for real-time predictions and visualizations.

## Problem Statement
Develop an end-to-end solution that:
- Classifies customer behavior to predict purchase conversion.
- Estimates potential revenue per user.
- Segments customers based on their browsing patterns.

This empowers e-commerce platforms to optimize marketing strategies, enhance product recommendations, and boost overall customer engagement.

## Data Details
- **Source:** UCI Machine Learning Repository – Clickstream Data
- **Datasets:**
  - **train.csv:** For training the machine learning models.
  - **test.csv:** For evaluating model performance.
- **Key Variables:**
  - **YEAR, MONTH, DAY, ORDER**
  - **COUNTRY, SESSION ID**
  - **PAGE 1 (Main Category), PAGE 2 (Clothing Model)**
  - **COLOUR, LOCATION, MODEL PHOTOGRAPHY**
  - **PRICE, PRICE 2, PAGE**

## Approach

### Data Preprocessing
- **Missing Values:** Impute numerical features using mean/median and categorical features using mode.
- **Feature Encoding:** Apply One-Hot Encoding or Label Encoding.
- **Scaling:** Use MinMaxScaler or StandardScaler for numerical features.

### Exploratory Data Analysis (EDA)
- **Visualizations:** Generate bar charts, histograms, and pair plots.
- **Session & Time-based Analysis:** Evaluate session duration, page views, and time-related trends.
- **Correlation Analysis:** Use heatmaps to identify relationships between variables.

### Feature Engineering
- **Session Metrics:** Calculate session length, click counts, and time spent per category.
- **Behavioral Patterns:** Analyze clickstream sequences and bounce rates.
- **Derived Features:** Create additional metrics to better capture user behavior.

### Pipeline Orchestration & Experiment Management
- **ZenML:**  
  Automate and manage your data processing, model training, and deployment pipelines. ZenML ensures reproducibility and modularity in the workflow.
- **MLflow:**  
  Log experiment parameters, model artifacts, and performance metrics with MLflow to facilitate tracking, comparison, and versioning of different model runs.

### Model Building & Evaluation
- **Models:**
  - **Classification:** Logistic Regression, Decision Trees, Random Forest, XGBoost, etc.
  - **Regression:** Linear Regression, Ridge, Lasso, Gradient Boosting Regressors.
  - **Clustering:** K-means, DBSCAN, Hierarchical Clustering.
- **Evaluation Metrics:**
  - **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
  - **Regression:** MAE, MSE, RMSE, R-squared.
  - **Clustering:** Silhouette Score, Davies-Bouldin Index.
- **Pipeline Integration:** Utilize Scikit-learn Pipelines to streamline the entire process from preprocessing to model evaluation.

### Streamlit Application
- **Interactive UI:**  
  Develop a user-friendly interface to:
  - Upload CSV files or manually input data.
  - View real-time predictions for customer conversion.
  - Display revenue estimates and customer segmentation visualizations.
- **Deployment:**  
  Easily deploy the application to serve business users with intuitive insights.

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/Kamalesh-Gandhi/Customer-Segmentation-Analysis-Forecast-Model.git
cd your-repo
```
### Set Up a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Deployed this application in "Streamlit Cloud" platform. Use this link to access the UI
https://customer-segmentation-analysis-forecast-model-fskpcaj3pq3bkmha.streamlit.app
