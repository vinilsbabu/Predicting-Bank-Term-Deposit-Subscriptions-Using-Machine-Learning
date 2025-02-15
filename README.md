# Predicting Bank Term Deposit Subscriptions Using Machine Learning

## Project Overview
This project focuses on predicting customer subscriptions to bank term deposits based on direct marketing campaign data. The dataset, sourced from a Portuguese banking institution, contains information from phone-based marketing campaigns. The goal is to preprocess the data, perform exploratory data analysis (EDA), engineer features, and build machine learning models to achieve accurate predictions.

---

## Dataset Description
The dataset consists of 45,211 entries and 17 features. Below are the key features:

- **age**: Client's age.
- **job**: Type of job.
- **marital**: Marital status.
- **education**: Education level.
- **default**: Has credit in default? (yes/no).
- **balance**: Average yearly account balance (in euros).
- **housing**: Has a housing loan? (yes/no).
- **loan**: Has a personal loan? (yes/no).
- **contact**: Communication type (e.g., cellular, telephone).
- **day**: Last contact day of the month.
- **month**: Last contact month.
- **duration**: Last contact duration (in seconds).
- **campaign**: Number of contacts performed during the campaign.
- **pdays**: Days since last contact in a previous campaign.
- **previous**: Number of previous contacts.
- **poutcome**: Outcome of the previous marketing campaign.
- **y**: Target variable (term deposit subscription: yes/no).

---

## Project Workflow

### 1. Data Preprocessing
- **Data Loading**: Loaded the dataset and checked for missing values and duplicates.
- **Handling Missing Values**: Replaced 'unknown' values with NaN and addressed them accordingly.
- **Encoding**: Applied one-hot encoding for categorical features.
- **Class Balancing**: Tackled class imbalance in the target variable using upsampling techniques.

### 2. Exploratory Data Analysis (EDA)
- **Target Variable Analysis**: Identified a significant imbalance in term deposit subscriptions.
- **Numerical Features Analysis**: Visualized distributions and identified skewness in `balance` and `duration`.
- **Correlation Analysis**: Used heatmaps to evaluate relationships among numerical features.
- **Categorical Features Analysis**: Examined distributions for features such as job, marital status, and education.

### 3. Feature Engineering
- **Encoding**: Encoded the target variable (`yes` -> 1, `no` -> 0).
- **Feature Selection**: Selected key features based on Random Forest.
  - Selected features include: 'month_1', 'poutcome_2', 'month_7', 'previous', 'duration', 'education_1', 'day', 'balance', 'education_2', 'loan_1', 'month_8', 'job_4', 'job_9', 'age', 'pdays', 'housing_1', 'marital_1', 'month_10', 'campaign'.
- **Feature Scaling**: Applied Min-Max Scaling.

### 4. Machine Learning Models
Implemented the following classification models:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting**
- **AdaBoost**
- **Multi-Layer Perceptron (MLP)**
- **Naive Bayes**


### 5. Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC Score.
- **Regression Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared, Root Mean Squared Error (RMSE).

---

## Key Findings
- The dataset is highly imbalanced, with most clients not subscribing to term deposits.
- Features such as `duration`, `balance`, and `number of contacts` proved to be strong predictors.
- Random Forest and Gradient Boosting achieved the best performance in classification models.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bank-term-deposit-prediction.git
2. Navigate to the project directory:
   cd bank-term-deposit-prediction
3. Install required libraries:
   pip install -r requirements.txt
4. Run the preprocessing and EDA scripts.
   python src/preprocessing.py
   python src/eda.py
5. Train and evaluate models by running
   python src/train_models.py

---

## Requirements
- Python 3.8+.
- Required Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib

---

## Contributing
Contributions are welcome! If you find a bug or want to add a feature, feel free to fork the repository, create a pull request, or open an issue.

---

## Acknowledgements
This project uses the Bank Marketing Dataset from the UCI Machine Learning Repository.


Let me know if you'd like further refinements! 🚀





- Random Forest and Gradient Boosting achieved the best performance in classification models.
