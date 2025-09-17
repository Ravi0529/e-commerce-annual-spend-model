# E-Commerce Customer Spending Prediction Model

## 📊 Project Overview

This project implements a **Linear Regression model** to predict annual customer spending in an e-commerce business. The model analyzes customer behavior patterns and membership data to forecast yearly spending amounts, helping businesses understand customer value and make data-driven decisions.

## 🗂️ Project Structure

```
1-E-commerce-linear-regression/
├── data/
│   └── ecommerce.xlsx          # Dataset containing customer information
├── yearly_spend-prediction_model.ipynb  # Main Jupyter notebook with model implementation
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```

## 📋 Dataset Description

The dataset (`ecommerce.xlsx`) contains **500 customer records** with the following features:

| Feature              | Type    | Description                                  |
| -------------------- | ------- | -------------------------------------------- |
| Email                | Object  | Customer email address                       |
| Address              | Object  | Customer address                             |
| Avatar               | Object  | Customer avatar information                  |
| Avg. Session Length  | Float64 | Average time spent per session (minutes)     |
| Time on App          | Float64 | Time spent on mobile app (minutes)           |
| Time on Website      | Float64 | Time spent on website (minutes)              |
| Length of Membership | Float64 | Years of membership                          |
| Yearly Amount Spent  | Float64 | **Target variable** - Annual spending amount |

## 🛠️ Code Implementation Details

### 1. **Data Loading and Exploration**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/ecommerce.xlsx")
df.head()
df.info()
df.describe()
```

### 2. **Data Visualization**

- **Scatter plots with marginal histograms** for key variables:
  - Time on Website vs Yearly Amount Spent
  - Time on App vs Yearly Amount Spent
  - Length of Membership vs Yearly Amount Spent
- **Pairplot** showing relationships between all numeric variables
- **Correlation analysis** through scatter matrix

### 3. **Feature Selection**

Selected 4 key features for the model:

```python
X = df[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = df["Yearly Amount Spent"]
```

### 4. **Data Splitting**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

- **Training set**: 70% of data
- **Test set**: 30% of data
- **Random state**: 42 for reproducibility

### 5. **Model Training**

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
```

### 6. **Feature Importance Analysis**

Model coefficients reveal feature importance:

```
                           Coef
Avg. Session Length   25.724256
Time on App           38.597135
Time on Website        0.459148
Length of Membership  61.674732
```

**Key Insights:**

- **Length of Membership** has the highest impact (61.67)
- **Time on App** is the second most important (38.60)
- **Time on Website** has minimal impact (0.46)

### 7. **Model Evaluation**

```python
predictions = lin_reg.predict(X_test)

# Performance Metrics:
Mean Absolute Error: 8.43
MSE: 103.92
RMSE: 10.19
```

### 8. **Model Validation**

- **Scatter plot** of predictions vs actual values
- **Residual analysis** with histogram and Q-Q plot
- **Normality testing** of residuals

## 📈 Model Performance

| Metric   | Value  | Interpretation                    |
| -------- | ------ | --------------------------------- |
| **MAE**  | 8.43   | Average prediction error is $8.43 |
| **MSE**  | 103.92 | Mean squared error                |
| **RMSE** | 10.19  | Root mean squared error           |

## 🔍 Key Findings

1. **Length of Membership** is the strongest predictor of yearly spending
2. **Time on App** significantly influences spending behavior
3. **Time on Website** has negligible impact on spending
4. **Average Session Length** moderately affects spending patterns
5. Model achieves reasonable accuracy with RMSE of ~$10

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages (see requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook yearly_spend-prediction_model.ipynb
   ```

## 📦 Dependencies

Key libraries used:

- **pandas** (2.3.2) - Data manipulation
- **numpy** (2.3.3) - Numerical computing
- **matplotlib** (3.10.6) - Data visualization
- **scikit-learn** (1.7.2) - Machine learning
- **scipy** (1.16.2) - Statistical functions

---

_This project serves as a foundation for understanding customer behavior patterns and implementing predictive analytics in e-commerce businesses._
