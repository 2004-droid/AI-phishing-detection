# AI-Powered Phishing Detection System

A team project (5 students, CSIT375) that combines machine learning with cybersecurity. Paste a URL and the app tells you whether it looks **legitimate** or like **phishing**, with a risk level and the warning signs behind the verdict.

## Results

Models were trained on about 57,000 labelled URLs (80/20 stratified train/test split, 5-fold cross-validation).

| Model | Features | Test F1 |
|---|---|---|
| Random Forest (200 trees) | all 111 | **0.960** (accuracy 0.957) |
| Gradient Boosting | all 111 | 0.952 |
| Logistic Regression | all 111 | 0.906 |
| Random Forest | 98 URL-text only | 0.913 |
| Gradient Boosting | 98 URL-text only | 0.904 |
| Logistic Regression | 98 URL-text only | 0.872 |

Random Forest was the best model, and cross-validation showed no sign of overfitting. The web app only sees a pasted URL, so it uses the 98 URL-text ("lexical") features and not the 13 features that need live internet lookups (such as domain age).

## Project structure

| Folder | What it contains |
|---|---|
| `1 - Data Set and processing` | Data cleaning and the 80/20 split, plus the cleaned, train and test CSV files |
| `2 - Machine Learning Model` | `375Models.ipynb`: trains and compares Random Forest, Logistic Regression and Gradient Boosting |
| `3 - Testing and results` | Notebook that tests the model on unseen data (accuracy, precision, recall, F1) |
| `4 - Application and interface` | Streamlit web app (`app.py`) |

## How to run

```
pip install streamlit pandas scikit-learn joblib matplotlib notebook
```

- **Notebooks:** open them in Jupyter or VS Code and choose Run All. They find the CSV files automatically.
- **App:** open a terminal in `4 - Application and interface` and run `streamlit run app.py` (or `python -m streamlit run app.py`). The first start trains the model from the CSV and can take up to a minute.

## My role

I built the machine learning model (folder 2): training and comparing the three models, cross-validation, and the URL-only versions used by the app.

## Dataset

Data from the phishing URL dataset of Vrbančič et al. (2020), "Datasets for phishing websites detection", *Data in Brief*. All credit for the data belongs to the original authors.

## Team

[Your name], [teammate 1], [teammate 2], [teammate 3], [teammate 4]
