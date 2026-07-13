# 📊 End-to-End Sales Forecasting & Demand Intelligence System

## 📌 Project Overview

This project is an end-to-end Sales Forecasting and Demand Intelligence System developed using Python and Machine Learning. The objective is to analyze historical Superstore sales data, forecast future sales, detect anomalies, segment product demand, and present insights through an interactive Streamlit dashboard.

The project combines Data Analytics, Time Series Forecasting, Machine Learning, Clustering, and Business Intelligence into one complete solution.

---

## 🎯 Project Objectives

- Analyze historical sales trends
- Forecast future sales using multiple forecasting models
- Compare forecasting model performance
- Detect unusual sales patterns (Anomalies)
- Segment products based on demand behavior
- Build an interactive Streamlit Dashboard
- Generate business recommendations for inventory planning

---

## 📂 Dataset

**Dataset:** Superstore Sales Dataset

The dataset contains:

- Order Date
- Ship Date
- Category
- Sub-Category
- Region
- Sales
- Profit
- Quantity
- Customer Information

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

---

## 🚀 Project Workflow

### Task 1
- Data Loading
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)

### Task 2
- Time Series Analysis
- Seasonal Decomposition
- Stationarity Test (ADF)
- Differencing

### Task 3
- SARIMA Forecasting
- Facebook Prophet Forecasting
- XGBoost Forecasting
- Model Comparison using MAE, RMSE, and MAPE

### Task 4
- Category & Region Level Forecasting
- 3-Month Sales Prediction
- Forecast Comparison

### Task 5
- Anomaly Detection
- Isolation Forest
- Z-Score Method

### Task 6
- Product Demand Segmentation
- K-Means Clustering
- Elbow Method
- PCA Visualization

### Task 7
- Interactive Streamlit Dashboard
- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

### Task 8
- Executive Business Report
- Business Recommendations

---

## 📊 Dashboard Features

✔ Sales Overview Dashboard

✔ Total Sales by Year

✔ Monthly Sales Trend

✔ Region & Category Filters

✔ Forecast Explorer

✔ MAE & RMSE Display

✔ Anomaly Detection Report

✔ Product Demand Segmentation

---

## 📁 Project Structure

```
SalesForecasting_Project/

│── analysis.ipynb
│── app.py
│── train.csv
│── requirements.txt
│── README.md
│── summary.pdf
│── summary.docx
│── product_demand_segmentation.csv
│── charts/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SalesForecasting_Project.git
```

Go to project folder

```bash
cd SalesForecasting_Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📈 Machine Learning Models

- SARIMA
- Facebook Prophet
- XGBoost Regressor
- Isolation Forest
- K-Means Clustering

---

## 📌 Evaluation Metrics

- MAE
- RMSE
- MAPE

---

## 💼 Business Recommendations

- Maintain higher inventory for high-demand products.
- Reduce inventory for declining demand products.
- Monitor sales anomalies to identify unusual business events.
- Use forecasting results for demand planning.
- Improve inventory management using product demand segmentation.

---

## 👨‍💻 Author

**Sujit Kushwaha**

B.Tech – Computer Science & Engineering

IEC College of Engineering & Technology

Dr. A.P.J. Abdul Kalam Technical University (AKTU)

---

## ⭐ Acknowledgement

This project was developed as part of a Machine Learning & Data Analytics Internship to demonstrate practical implementation of sales forecasting, demand intelligence, anomaly detection, clustering, and interactive dashboard development using Python.
