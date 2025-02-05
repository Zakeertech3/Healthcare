# 📊 Advanced Medical Cost Analysis App

Welcome to the **Advanced Medical Cost Analysis App**! This interactive Streamlit application allows users to explore, visualize, and analyze medical cost data efficiently. Whether you're a data analyst, researcher, or business professional, this app provides tools for deep data insights and simple machine learning models to predict medical expenses.


![Image](https://github.com/user-attachments/assets/3b30739b-80b3-4b39-8ed6-80adfa5fc1dc)
---

## 💼 **Business Requirement**

Healthcare costs are a major concern for both individuals and organizations. Understanding the factors that influence medical expenses can help:

- **Insurance Companies**: To adjust premiums and identify high-risk individuals.
- **Healthcare Providers**: To design preventive care programs and manage resources.
- **Policy Makers**: To develop strategies that reduce healthcare costs.

This app aims to:
1. **Identify key factors** affecting medical charges (e.g., age, BMI, smoking habits).
2. Provide **visual insights** to understand data distributions and relationships.
3. Offer a **predictive model** for estimating medical costs based on user-selected features.

---

## 💡 **What Type of Analysis Was Done?**

1. **Exploratory Data Analysis (EDA)** 🔍
   - Inspected data structure, types, and missing values.
   - Visualized distributions of numerical features like **age**, **BMI**, and **medical charges**.
   - Analyzed categorical features like **sex**, **smoker**, and **region**.

2. **Relationship Analysis** 📊
   - Used **boxplots** to see how features like smoking status, gender, and region affect medical costs.
   - Found that **smoking** had the largest impact on medical expenses.

3. **Correlation Analysis** 🔄
   - Created **heatmaps** to visualize correlations between numerical features and medical costs.
   - Encoded categorical variables (like **smoker** and **sex**) to include them in the correlation analysis.

4. **Predictive Modeling** 📊🔢
   - Built a **Linear Regression** model to predict medical costs based on selected features.
   - Displayed model performance metrics like **Mean Squared Error (MSE)** and **R² Score**.

---

## 🔧 **How to Run the App?**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Zakeertech3/Medical_Cost_Project.git
   cd Medical_Cost_Project
   ```

2. **Install Required Libraries**:
   ```bash
   pip install streamlit pandas numpy seaborn matplotlib scikit-learn
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

4. **Upload Your Dataset**:
   - Upload a **CSV** or **Excel** file in the sidebar.
   - Explore raw and filtered data, create visualizations, and build predictive models!

---

## 📑 **Features of the App**

1. **Data Upload & Display**:
   - Upload **CSV** or **Excel** files.
   - View raw and filtered data directly in the app.

2. **Advanced Visualizations**:
   - **Scatter Plots** to explore relationships between variables.
   - **Histograms** to understand the distribution of features.
   - **Correlation Heatmaps** to see how features relate to each other.

3. **Data Filtering**:
   - Filter data dynamically by selecting specific values for any column.
   - Download filtered data as a CSV.

4. **Machine Learning**:
   - Build a **Linear Regression Model** to predict medical costs.
   - Display performance metrics like **Mean Squared Error** and **R² Score**.
   - Visualize **actual vs predicted** medical costs.

---

## 📅 **Technologies Used**

- **Python** 🔬
- **Streamlit** 🔄
- **Pandas** & **NumPy** for data manipulation 📂
- **Seaborn** & **Matplotlib** for visualizations 🌍
- **Scikit-learn** for machine learning 💻

---

## 🌟 **Contributing**

Contributions are welcome! If you find a bug or want to add new features, feel free to open an issue or submit a pull request. 🚀

---

## 💎 **License**

This project is licensed under the MIT License.

---

**Happy Analyzing!** 🎉🚀
