# Smart Loan Default AI

An interactive machine learning application that estimates credit-card payment default probability using a Random Forest classifier. Built with Python, Scikit-learn, Pandas, and Streamlit.

## Project Overview

Smart Loan Default AI explores how historical customer and repayment data can be used to estimate the likelihood of payment default in the following month.

The application provides:
- Interactive default-probability predictions
- A configurable classification threshold based on experimental evaluation
- Model comparison metrics and visualizations
- Threshold analysis for precision, recall, and F1-score
- A Streamlit dashboard with multiple navigation pages

## Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, Random Forest
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Streamlit charts
- **Model Persistence:** Joblib
- **Development Environment:** VS Code and Jupyter Notebook

## Dataset

This project uses the UCI Machine Learning Repository's **Default of Credit Card Clients** dataset.

The dataset contains 30,000 customer records and 23 predictor features after excluding the ID and target columns.

Target variable: `default payment next month`

- `0`: No payment default in the following month
- `1`: Payment default in the following month

Refer to the original dataset documentation for feature definitions, category encodings, and usage terms.

## Project Structure

```text
Smart-Loan-Default-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Dataset instructions
├── notebooks/
│   └── 01_data_exploration.ipynb
└── outputs/
    ├── random_forest.joblib
    ├── logistic_regression.joblib
    ├── model_comparison.csv
    └── threshold_comparison.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart-Loan-Default-AI.git
cd Smart-Loan-Default-AI
```

Replace `YOUR_USERNAME` with your GitHub username and use the repository URL you create.

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Model Evaluation

The project compares Logistic Regression and Random Forest using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Actual results are available in `outputs/model_comparison.csv`. Threshold experiment results are available in `outputs/threshold_comparison.csv`.

See these files for measured values rather than relying on assumed or illustrative metrics.

## Responsible Use and Limitations

This project is an educational machine learning prototype, not a production lending or credit-scoring system.

- Predictions are estimates, not guarantees.
- Model performance depends on the dataset and evaluation methodology.
- Classification thresholds affect false positives and false negatives.
- Historical data can contain bias and may not represent current customers.
- Real lending decisions require rigorous validation, fairness assessment, privacy safeguards, and appropriate legal and regulatory review.

Do not use this application to approve or reject real loan applications.

## Future Improvements

- Add validation-based threshold selection
- Improve model explainability using feature analysis
- Add model monitoring and data drift checks
- Evaluate fairness across relevant customer groups
- Deploy the dashboard to a hosting platform

## Author

**Piyush Raj**

B.Tech Computer Science Engineering | Aspiring Data Analyst and Machine Learning Practitioner

- [GitHub](https://github.com/TPiyushRaj-CS)
- [LinkedIn](https://linkedin.com/in/piyush-thakur-a30418343)
- [Portfolio](https://tpiyushraj-cs.github.io/piyush-portfolio/)
