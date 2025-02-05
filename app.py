import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set page title and layout
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")
st.title("📊 Advanced Data Analysis App with DSA")

# Sidebar for file upload and settings
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Initialize session state for data
if "data" not in st.session_state:
    st.session_state.data = None

# Load data
if uploaded_file is not None:
    start_time = time.time()  # Start timer for loading data
    if uploaded_file.name.endswith(".csv"):
        st.session_state.data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        st.session_state.data = pd.read_excel(uploaded_file)
    load_time = time.time() - start_time  # Calculate loading time
    st.sidebar.write(f"Time to load data: {load_time:.4f} seconds")

# Display data and analysis options
if st.session_state.data is not None:
    df = st.session_state.data

    # Show raw data
    st.subheader("Raw Data")
    st.write(df)

    # Data filtering
    st.sidebar.header("Data Filtering")
    filter_column = st.sidebar.selectbox("Select column to filter by", df.columns)
    unique_values = df[filter_column].unique()
    selected_values = st.sidebar.multiselect("Select values to keep", unique_values, default=unique_values)

    start_time = time.time()  # Start timer for filtering
    filtered_df = df[df[filter_column].isin(selected_values)]
    filter_time = time.time() - start_time  # Calculate filtering time
    st.sidebar.write(f"Time to filter data: {filter_time:.4f} seconds")

    # Show filtered data
    st.subheader("Filtered Data")
    st.write(filtered_df)

    # Download filtered data
    st.sidebar.header("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )

    # Advanced visualizations
    st.subheader("Advanced Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Scatter Plot")
        x_axis = st.selectbox("Select X-axis", df.columns, key="x_axis")
        y_axis = st.selectbox("Select Y-axis", df.columns, key="y_axis")
        start_time = time.time()  # Start timer for scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
        scatter_time = time.time() - start_time  # Calculate scatter plot time
        st.write(f"Time to generate scatter plot: {scatter_time:.4f} seconds")

    with col2:
        st.write("### Histogram")
        hist_column = st.selectbox("Select column for histogram", df.columns, key="hist_column")
        bins = st.slider("Number of bins", 5, 100, 20)
        start_time = time.time()  # Start timer for histogram
        fig, ax = plt.subplots()
        sns.histplot(filtered_df[hist_column], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)
        hist_time = time.time() - start_time  # Calculate histogram time
        st.write(f"Time to generate histogram: {hist_time:.4f} seconds")

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        start_time = time.time()  # Start timer for heatmap
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        heatmap_time = time.time() - start_time  # Calculate heatmap time
        st.write(f"Time to generate heatmap: {heatmap_time:.4f} seconds")
    else:
        st.warning("No numeric columns found for correlation heatmap.")

    # Machine Learning Section
    st.subheader("Machine Learning: Linear Regression")
    if not numeric_df.empty:
        target = st.selectbox("Select target variable", numeric_df.columns)
        features = st.multiselect("Select features", numeric_df.columns.drop(target))

        if features and target:
            X = numeric_df[features]
            y = numeric_df[target]

            # Train-test split
            start_time = time.time()  # Start timer for train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            split_time = time.time() - start_time  # Calculate split time
            st.write(f"Time to split data: {split_time:.4f} seconds")

            # Train model
            start_time = time.time()  # Start timer for model training
            model = LinearRegression()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time  # Calculate training time
            st.write(f"Time to train model: {train_time:.4f} seconds")

            # Make predictions
            start_time = time.time()  # Start timer for predictions
            y_pred = model.predict(X_test)
            predict_time = time.time() - start_time  # Calculate prediction time
            st.write(f"Time to make predictions: {predict_time:.4f} seconds")

            # Display results
            st.write("#### Model Performance")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

            # Plot predictions vs actual
            start_time = time.time()  # Start timer for plotting
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
            plot_time = time.time() - start_time  # Calculate plotting time
            st.write(f"Time to generate plot: {plot_time:.4f} seconds")
        else:
            st.warning("Please select at least one feature and a target variable.")
    else:
        st.warning("No numeric columns found for machine learning.")

else:
    st.info("Please upload a CSV or Excel file to get started.")