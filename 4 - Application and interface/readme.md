# AI-Powered Phishing Detection: Streamlit App (Task 4)

A small web app where you enter a URL and it tells you whether the URL looks like phishing or legitimate.

## How to run

1. Install Python 3, then the libraries:

   ```
   pip install streamlit pandas scikit-learn joblib
   ```

2. Open a terminal **in this folder** (the one that contains `app.py`) and run:

   ```
   streamlit run app.py
   ```

3. A page opens in your browser. Type a URL, click **Analyze URL**, and read the result.

## What you see

- A verdict (phishing or legitimate) with the model's phishing probability
- A risk level (Low, Medium, High) and advice, for example "Do not open this link"
- A list of warning signs found in the URL (IP address instead of a domain, very long URL, "@" symbol, shortened link, and so on)

## How it works

- The app turns the URL text into the 98 URL-text ("lexical") features that the model was trained on (URL length, number of dots, hyphens, slashes, and so on).
- It does not use the 13 features that need live internet lookups (such as domain age), because the app only has the pasted URL.
- The model is a Random Forest (200 trees) trained on `phishing_train.csv`. On the test set it scored about 0.91 F1 with these URL-text features.

## About the model files

- If `rf_lexical.joblib` and `columns_lexical.joblib` are next to `app.py`, the app loads them.
- If they are **not** there, the app trains the same model once when it starts, using `phishing_train.csv` from the folder `../1 - Data Set and processing/` (or from this folder). This can take from a few seconds to about a minute on the first start.
