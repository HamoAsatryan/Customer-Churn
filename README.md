# 📉 Customer Churn Prediction

This project focuses on predicting customer churn — identifying which customers are likely to leave the company.  
The goal is to analyze the data, compare models, and deploy a model for real-world predictions.

---

## 📂 Project Structure

customer-churn/
│── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv # dataset
│── notebooks/
│ └── churn_analysis.ipynb # exploratory analysis and visualizations
│── models/
│ └── churn_model.joblib # saved trained model
│── src/
│ ├── train.py # training script
│ └── predict.py # script for predicting new data
│── README.md
│── requirements.txt



---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib, Seaborn  
- imbalanced-learn (SMOTE)  
- Joblib

---

## 📊 Results

The following models were compared on the test set:

| Model                | Accuracy | F1     | ROC-AUC |
|----------------------|----------|--------|---------|
| Logistic Regression  | 0.740    | 0.613  | 0.836   |
| Random Forest        | 0.776    | 0.586  | 0.823   |
| XGBoost              | 0.773    | 0.570  | 0.812   |

- **Logistic Regression** was chosen as the final model due to its best performance on F1 and ROC-AUC, which are critical for churn prediction.

---

## 📈 Visualizations (from the notebook)

- Feature distributions and correlations  
- Churn distribution by contract type  
- ROC curves for all models  
- Feature importance for Random Forest / XGBoost  

*(See `notebooks/churn_analysis.ipynb` for all plots.)*

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your_username/customer-churn.git
