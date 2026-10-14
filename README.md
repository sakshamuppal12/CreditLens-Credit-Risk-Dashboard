# CreditLens — Credit Risk Intelligence Dashboard

CreditLens is a Python-based data science project that explores loan-performance data, examines data quality, compares credit-risk classification models, and presents analytical findings through an interactive Streamlit dashboard.

## Project Overview

The project analyzes application information and recorded loan outcomes to explore patterns associated with good and bad loans.

### Key Features

- **Data Quality Analysis:** Examines missing values, duplicate records, and data consistency.
- **Data Integration:** Investigates the relationship between application records and loan-performance records.
- **Machine Learning:** Compares Logistic Regression and Gradient Boosting across different feature sets.
- **Model Evaluation:** Reports ROC-AUC, Accuracy, Precision, and Recall using cross-validated predictions.
- **Visual Analytics:** Generates ROC curves, risk-group outcome charts, and logistic regression coefficient plots.
- **Interactive Dashboard:** Uses Streamlit to display data summaries, diagnostics, and project limitations.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Excel data processing

## Project Structure

```text
taplend_project/
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── res.json
├── roc.png
├── quint.png
└── coef.png
```

Additional files may be generated when the model script runs. Processed data files containing applicant information should not be published.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the dataset

Place the authorized assessment workbook in the location expected by `model.py`, or configure the workbook path in the script.

The source workbook is intentionally excluded from the repository.

### 5. Generate model outputs

```bash
python model.py
```

This generates model evaluation results and diagnostic charts used by the dashboard.

### 6. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## Model Evaluation

The project compares Logistic Regression and Gradient Boosting using three feature sets:

- Scores only
- Application only
- All features

Evaluation includes ROC-AUC, Accuracy, Precision, and Recall. Results are exploratory and depend on the selected features, outcome definition, record-matching assumptions, and validation procedure.

## Limitations

- Application-to-loan record linkage requires further verification.
- Missing values and data quality may affect the analysis.
- Potential target leakage must be reviewed.
- Cross-validation results do not establish performance on future, unseen borrowers.
- The project is an analytical prototype, not a production-ready lending decision system.

## Responsible Data Use

Do not commit source workbooks, personal applicant information, banking details, credentials, environment files, or sensitive processed datasets to a public repository.

## Author

**Saksham Uppal**  
B.Tech Computer Science and Engineering  
Lovely Professional University

## Disclaimer

This project is developed for educational and analytical purposes. It should not be used as the sole basis for real-world lending decisions.
