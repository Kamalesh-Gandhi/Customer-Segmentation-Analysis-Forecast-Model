import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import os ,joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error


# ✅ Mapping for categorical values
COUNTRY_MAP = {
    1: "Australia", 2: "Austria", 3: "Belgium", 4: "British Virgin Islands", 5: "Cayman Islands",
    6: "Christmas Island", 7: "Croatia", 8: "Cyprus", 9: "Czech Republic", 10: "Denmark",
    11: "Estonia", 12: "unidentified", 13: "Faroe Islands", 14: "Finland", 15: "France",
    16: "Germany", 17: "Greece", 18: "Hungary", 19: "Iceland", 20: "India", 21: "Ireland",
    22: "Italy", 23: "Latvia", 24: "Lithuania", 25: "Luxembourg", 26: "Mexico", 27: "Netherlands",
    28: "Norway", 29: "Poland", 30: "Portugal", 31: "Romania", 32: "Russia", 33: "San Marino",
    34: "Slovakia", 35: "Slovenia", 36: "Spain", 37: "Sweden", 38: "Switzerland", 39: "Ukraine",
    40: "United Arab Emirates", 41: "United Kingdom", 42: "USA"
}

CATEGORY_MAP = {1: "Trousers", 2: "Skirts", 3: "Blouses", 4: "Sale"}
COLOR_MAP = {1: "Beige", 2: "Black", 3: "Blue", 4: "Brown", 5: "Burgundy", 6: "Gray", 7: "Green", 
             8: "Navy Blue", 9: "Multicolor", 10: "Olive", 11: "Pink", 12: "Red", 13: "Violet", 14: "White"}

LOCATION_MAP = {1: "Top Left", 2: "Top Middle", 3: "Top Right", 4: "Bottom Left", 5: "Bottom Middle", 6: "Bottom Right"}
PHOTOGRAPHY_MAP = {1: "En Face", 2: "Profile"}

# ✅ Reverse mapping for predictions
def reverse_map(selection, mapping):
    return {v: k for k, v in mapping.items()}[selection]

# Title
st.markdown("""
    <h1 style='text-align: center; color: #17202a;'>Customer Behavior Analyzer 📊</h1>
""", unsafe_allow_html=True)


# ✅ Function to Load the Best Model
def load_best_model(model_type: str):
    """
    Load the best classification or regression model.
    :param model_type: str ('classification' or 'regression')
    :return: Loaded Model
    """
    model_mapping = {
        "classification": "best_Random_Forest_classifier.pkl",
        "regression": "best_XGBoost_regressor.pkl",
        "clustering": "K-Means_clustering.pkl"
    }

    model_path = os.path.join("models", model_mapping.get(model_type, ""))
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"🚨 Best {model_type} model not found! in this {model_path}")
        return None

# ✅ Load Models for Prediction
best_classification_model = load_best_model("classification")
best_regression_model = load_best_model("regression")
best_clustering_model = load_best_model("clustering")

# ✅ Load Train Data to Extract Unique `page2_clothing_model` Values
train_data_path = "data/Train_data.csv"  # ✅ Make sure this path is correct
if os.path.exists(train_data_path):
    train_data = pd.read_csv(train_data_path)
    
    # Check if `page2_clothing_model` exists
    if "page2_clothing_model" in train_data.columns:
        unique_values = train_data["page2_clothing_model"].unique()
        
        # ✅ Load Label Encoder
        encoder_path = os.path.join("models", "Encoder.pkl")
        if os.path.exists(encoder_path):
            label_encoder_dict = joblib.load(encoder_path)
            if "page2_clothing_model" in label_encoder_dict:
                encoder = label_encoder_dict["page2_clothing_model"]  # ✅ Extract the encoder correctly
                encoded_values = encoder.transform(unique_values)  # ✅ Now transform works!
                clothing_model_map = dict(zip(unique_values, encoded_values))  # ✅ Mapping Original → Encoded
            else:
                st.error("🚨 'page2_clothing_model' key not found in the Encoder dictionary!")
        else:
            st.error("🚨 Encoder file not found! Encoding will not work properly.")
    else:
        st.error("🚨 Column `page2_clothing_model` not found in train data!")
else:
    st.error("🚨 Train data file not found!")

# Function to Set Background Image from Local File
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                        url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply Background Image
set_background("UI Images\main_backgroundimage.jpg")


# Sidebar Background Color
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #cecef4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Logo image 
st.sidebar.image("UI Images\sidebar_logoimage.png", use_container_width=True)

# Sidebar Navigation
page = st.sidebar.radio("**NAVIGATION**", ["🏠 Home", "📂 Bulk Customers Analyzer", "👤 Single Customer Analyzer"])

# ✅ Initialize default values for session state
default_values = {
    # Classification Session Keys (without target column "Purchase Completed")
    "classification_order": 10,
    "classification_country": "USA",
    "classification_session_id": 1001,
    "classification_page1_main_category": "Trousers",
    "classification_colour": "Beige",
    "classification_location": "Top Left",
    "classification_model_photography": "Profile",
    "classification_price": 50,
    "classification_price_2": 'Yes',
    "classification_page": 5,
    "classification_page2_clothing_model": "C20",

    # Regression Session Keys (without target column "price")
    "regression_order": 10,
    "regression_country": "USA",
    "regression_session_id": 1001,
    "regression_page1_main_category": "Trousers",
    "regression_colour": "Beige",
    "regression_location": "Top Left",
    "regression_model_photography": "Profile",
    "regression_price_2": 'Yes',
    "regression_page": 5,
    "regression_page2_clothing_model": "C20",
    "regression_purchase_completed": "Yes"
}
# Initialize session state
if "reset" not in st.session_state:
    st.session_state.reset = False
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset inputs
def reset_inputs():
    """Resets the input fields to default values."""

    # for key, value in default_values.items():
    #     st.session_state[key] = value  # ✅ Explicitly Reset Each Key
    st.rerun()

# Home Page Content
if page == "🏠 Home":
    st.markdown("<h2 class='stHeader'>Welcome to Customer Behavior Analyzer site</h2>", unsafe_allow_html=True)
    st.markdown("<p class='stSection'>This app helps businesses analyze customer behavior using machine learning techniques.</p>", unsafe_allow_html=True)
    
    st.write("""
        **Features:**
        - **Bulk Customers Analyzer:** Upload a CSV file to analyze multiple customers at once.
        - **Single Customer Analyzer:** Manually enter customer details for analysis.
            - **Classification:** Predict whether a customer will complete a purchase.
            - **Regression:** Estimate the potential revenue from a customer.
            - **Clustering:** Segment customers based on browsing behavior.
        
        **How to Use:**
        1. Use the sidebar to navigate between Home, Bulk Prediction, and Single Customer Prediction.
        2. In Bulk Prediction, upload a CSV file to analyze multiple customers.
        3. In Single Customer Prediction, input customer details manually.
        4. Run classification, regression, or clustering to get insights.
        
        Get started by selecting a feature from the sidebar!
    """)

# Bulk Prediction Page
if page == "📂 Bulk Customers Analyzer":
    st.header("Bulk Customers Analyzer - Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(data.head())
        
        col1, col2, col3 = st.columns(3)
        
        if col1.button("Run Classification", key="bulk_classification"):
            with st.spinner("Processing Classification..."):
                time.sleep(2)

                # ✅ Select only relevant columns for classification
                classification_features = ['order', 'country', 'session_id', 'page1_main_category', 'colour', 
                                           'location', 'model_photography', 'price', 'price_2', 'page', 'page2_clothing_model']

                data_for_classification = data[classification_features]
                predictions = best_classification_model.predict(data_for_classification)

                data["Predicted Purchase"] = predictions

                st.success("Classification Completed!")
                st.write("### Classification Predictions:")
                st.dataframe(data[["session_id", "Predicted Purchase"]])
                st.download_button("Download Classification Results", data.to_csv(index=False), "classification_results.csv")
        
        if col2.button("Run Regression", key="bulk_regression"):
            with st.spinner("Processing Regression..."):
                time.sleep(2)

                # ✅ Select only relevant columns for regression
                regression_features = ['order', 'country', 'session_id', 'page1_main_category', 'colour', 
                                       'location', 'model_photography', 'price_2', 'page', 'page2_clothing_model', 'Purchase Completed']

                data_for_regression = data[regression_features]
                predictions = best_regression_model.predict(data_for_regression)

                data["Estimated Revenue"] = predictions

                st.success("Regression Completed!")

                # ✅ Display results in tabular format
                st.write("### Regression Predictions:")
                st.dataframe(data[["session_id", "Estimated Revenue"]])

                st.download_button("Download Regression Results", data.to_csv(index=False), "regression_results.csv")
        
        if col3.button("Run Clustering", key="bulk_clustering"):
            with st.spinner("Processing Clustering..."):
                time.sleep(2)

                clustering_features = ['order', 'country', 'session_id', 'page1_main_category', 'colour', 
                            'location', 'model_photography', 'price_2', 'page', 'page2_clothing_model', 'Purchase Completed']
                data_clustering = data[clustering_features]
                data['Customer Segment'] = best_clustering_model.predict(data)

                st.success("Clustering Completed!")
                st.dataframe(data[['session_id','Customer Segment']])
                st.download_button("Download Clustering Results", data.to_csv(index=False), "clustering_results.csv")

                # **Cluster Distribution Bar Chart**
                st.write("### 🔍 Cluster Distribution")
                cluster_counts = data['Customer Segment'].value_counts()
                fig, ax = plt.subplots()
                cluster_counts.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_xlabel("Cluster ID")
                ax.set_ylabel("Number of Customers")
                ax.set_title("Customer Distribution Across Clusters")
                st.pyplot(fig)

                # **Cluster Percentage Pie Chart**
                st.write("### 📊 Cluster Percentage")
                fig, ax = plt.subplots()
                cluster_counts.plot(kind="pie", autopct='%1.1f%%', startangle=90, cmap="coolwarm", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Percentage of Customers in Each Cluster")
                st.pyplot(fig)

                # **Heatmap of Feature Correlations**
                st.write("### 🔥 Feature Importance Heatmap")
                fig, ax = plt.subplots(figsize=(8, 5))
                correlation_matrix = data_clustering.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                ax.set_title("Feature Importance in Clustering")
                st.pyplot(fig)



# 📌 Single Customer Prediction Page with Tabs
if page == "👤 Single Customer Analyzer":
    st.header("Single Customer Analyzer")

    if st.session_state.get("reset", False):
        for key, value in default_values.items():
            st.session_state[key] = value
        st.session_state.reset = False

    # 📌 Create Tabs for Classification & Regression
    tab1, tab2 = st.tabs(["**Purchase Prediction 🔍** ", "**Revenue Estimation 💰** "])

    with tab1:
        st.subheader("Predict Customer Purchase Behavior")

        col1, col2, col3 = st.columns(3)
        
        # Country Selection
        country = col1.selectbox("**Country**", list(COUNTRY_MAP.values()), key="classification_country")  
        
        # Main Category Selection
        category = col2.selectbox("**Main Category**", list(CATEGORY_MAP.values()), key="classification_page1_main_category") 
        
        # Clothing Model Code (Text Input)
        clothing_model = col3.selectbox("**Clothing Model**", list(clothing_model_map.keys()), key="classification_page2_clothing_model")  

        clothing_model_encoded_classification = clothing_model_map[clothing_model] # Convert back to encoded before prediction

        # Color Selection
        color = col1.selectbox("**Color**", list(COLOR_MAP.values()), key="classification_colour")
        
        # Location on Page
        location = col2.selectbox("**Location on Page**", list(LOCATION_MAP.values()), key="classification_location")

        # Model Photography
        model_photography = col3.selectbox("**Model Photography Type**", list(PHOTOGRAPHY_MAP.values()), key="classification_model_photography")  

        # Price Slider
        price = col1.number_input("**Product Price ($)**",min_value=0,  value=st.session_state["classification_price"], key="classification_price")  
        
        # Price Indicator (Above Avg?)
        price_indicator = col2.radio("**Is Price Above Average?**", ["Yes", "No"], 
                             index=0 if st.session_state["classification_price_2"] == "Yes" else 1,  
                             key="classification_price_2")    
        
        # Page Number within the Store
        page_number = col3.slider("**Page Number in Store**", 1, 5, st.session_state["classification_page"], key="classification_page")  

        # Session ID Input
        session_id = col1.number_input("**Session ID**", min_value=0, value=st.session_state["classification_session_id"], key="classification_session_id")  

        # Order Number
        order = col2.slider("**Order Number**", 1, 100, st.session_state["classification_order"], key="classification_order")  

        colA, colB = st.columns([2, 1])

    with colA:
        if st.button("Run Purchase Prediction", key="single_classification"):
            with st.spinner("Processing Purchase Prediction..."):
                time.sleep(2)
                # Convert UI selections to numerical values for prediction
                input_data = pd.DataFrame([[
                    order, reverse_map(country, COUNTRY_MAP), session_id, 
                    reverse_map(category, CATEGORY_MAP), clothing_model_encoded_classification, reverse_map(color, COLOR_MAP), 
                    reverse_map(location, LOCATION_MAP), reverse_map(model_photography, PHOTOGRAPHY_MAP), 
                    price, 1 if price_indicator == "Yes" else 0, page_number
                ]], columns=['order', 'country', 'session_id', 'page1_main_category','page2_clothing_model', 'colour', 
                             'location', 'model_photography', 'price', 'price_2', 'page'])
                prediction = best_classification_model.predict(input_data)[0]
                st.success(f"✅ Prediction: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")


    with colB:
        if st.button("♻️ Reset Inputs", key="reset_inputs_classification"):
            reset_inputs()

    with tab2:
        st.subheader("Estimate Customer Purchase Revenue")

        col1, col2, col3 = st.columns(3)
        
        # Country Selection
        country = col1.selectbox("**Country**", list(COUNTRY_MAP.values()), key="regression_country") 
        
        # Main Category Selection
        category = col2.selectbox("**Main Category**", list(CATEGORY_MAP.values()), key="regression_page1_main_category") 
         
        # Clothing Model Code (Text Input)
        clothing_model = col3.selectbox("**Clothing Model**", list(clothing_model_map.keys()), key="regression_page2_clothing_model")  

        clothing_model_encoded_regression = clothing_model_map[clothing_model]  # Convert back to encoded before prediction

        # Color Selection
        color = col1.selectbox("**Color**", list(COLOR_MAP.values()), key="regression_colour")  
        
        # Location on Page
        location = col2.selectbox("**Location on Page**", list(LOCATION_MAP.values()), key="regression_location") 

        # Model Photography
        model_photography = col3.selectbox("**Model Photography Type**", list(PHOTOGRAPHY_MAP.values()), key="regression_model_photography")  

        # Purchase Completed (Yes/No)
        purchase_completed = col1.radio("**Purchase Completed?**", ["Yes", "No"], 
                                index=0 if st.session_state["regression_purchase_completed"] == "Yes" else 1,  
                                key="regression_purchase_completed")  
  
        # Price Indicator (Above Avg?)
        price_indicator = col2.radio("**Is Price Above Average?**", ["Yes", "No"], 
                             index=0 if st.session_state["regression_price_2"] == "Yes" else 1,  
                             key="regression_price_2")    
        
        # Page Number within the Store
        page_number = col3.slider("**Page Number in Store**", 1, 5, st.session_state["regression_page"], key="regression_page")  

        # Session ID Input
        session_id = col1.number_input("**Session ID**", min_value=1000, max_value=9999, value=st.session_state["regression_session_id"], key="regression_session_id")  

        # Order Number
        order = col2.slider("**Order Number**", 1, 100, st.session_state["regression_order"], key="regression_order")  

        colA, colB = st.columns([2, 1])

        with colA:
            if st.button("Estimate Revenue", key="single_regression"):
                with st.spinner("Processing Revenue Estimation..."):
                    time.sleep(2)
                input_data = pd.DataFrame([[
                    order, reverse_map(country, COUNTRY_MAP), session_id, 
                    reverse_map(category, CATEGORY_MAP), clothing_model_encoded_regression , reverse_map(color, COLOR_MAP), 
                    reverse_map(location, LOCATION_MAP), reverse_map(model_photography, PHOTOGRAPHY_MAP), 
                    1 if price_indicator == "Yes" else 0, page_number, 1 if purchase_completed == "Yes" else 0
                ]], columns=['order', 'country', 'session_id', 'page1_main_category', 'page2_clothing_model', 'colour', 
                             'location', 'model_photography', 'price_2', 'page', 'Purchase Completed'])
                prediction = best_regression_model.predict(input_data)[0]
                st.success(f"💰 Estimated Revenue: ${prediction:.2f}")

        with colB:
            if st.button("♻️ Reset Inputs", key="reset_inputs_regression"):
                reset_inputs()
