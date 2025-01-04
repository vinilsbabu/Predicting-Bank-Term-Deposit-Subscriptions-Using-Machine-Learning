# Predicting Bank Term Deposit Subscriptions Using Machine Learning

## Project Overview
This project focuses on predicting customer subscriptions to bank term deposits based on direct marketing campaign data. The dataset, sourced from a Portuguese banking institution, contains information from phone-based marketing campaigns. The goal is to preprocess the data, perform exploratory data analysis (EDA), and prepare it for machine learning models.

## Dataset Description
The dataset includes 45,211 entries and 17 features. Key features are:

- **age**: Client's age.
- **job**: Type of job.
- **marital**: Marital status.
- **education**: Education level.
- **default**: Has credit in default? (yes/no).
- **balance**: Average yearly account balance (in euros).
- **housing**: Has a housing loan? (yes/no).
- **loan**: Has a personal loan? (yes/no).
- **contact**: Communication type (cellular, telephone, etc.).
- **day**: Last contact day of the month.
- **month**: Last contact month.
- **duration**: Last contact duration (in seconds).
- **campaign**: Number of contacts performed during the campaign.
- **pdays**: Days since last contact in a previous campaign.
- **previous**: Number of previous contacts.
- **poutcome**: Outcome of the previous marketing campaign.
- **y**: Target variable (yes/no for term deposit subscription).

## Project Workflow

### 1. Data Preprocessing
- **Loaded Dataset**: Imported the dataset and checked for basic information, missing values, and duplicates.
- **Handled Missing Values**: Replaced 'unknown' values with NaN for easier handling.
- **Encoded Categorical Variables**: Converted categorical features to numerical representations using one-hot encoding.
- **Balanced Classes**: Addressed target variable imbalance using upsampling techniques.

### 2. Exploratory Data Analysis (EDA)
- **Target Variable Analysis**: Checked the distribution of term deposit subscriptions (imbalanced, with a majority of "no").
- **Numerical Features Analysis**:
  - Plotted distributions for features like age, balance, duration, campaign, pdays, and previous.
  - Observed skewed distributions in balance and duration.
- **Correlation Analysis**:
  - Visualized relationships among numerical features using a correlation heatmap.
  - Found low correlations among most features, suggesting limited linear relationships.
- **Categorical Features Analysis**:
  - Examined distributions of job, marital status, education, and other categorical features.

### Key Findings
- The dataset is highly imbalanced, with more clients not subscribing to term deposits.
- Variables like duration, balance, and number of contacts show potential as strong predictors.
- Many clients had no prior contact, evident from pdays and previous features.

## File Structure
- `data/`: Contains the dataset (`bank_data.csv`).
- `notebooks/`: Includes Jupyter notebooks for preprocessing and EDA.
- `src/`: Contains Python scripts for preprocessing and analysis.
- `README.md`: Project overview and instructions.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bank-term-deposit-prediction.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing and EDA scripts:
   ```bash
   python src/preprocessing.py
   python src/eda.py
   ```

## Requirements
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Future Steps
- **Feature Engineering**: Create new features to enhance predictive power.
- **Model Building**: Train and evaluate machine learning models (e.g., logistic regression, decision trees, random forest).
- **Model Optimization**: Use hyperparameter tuning to improve performance.

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
This project uses the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) from the UCI Machine Learning Repository.

