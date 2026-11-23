
import re
 
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent          # the folder that contains app.py
MODEL_FILE = HERE / "rf_lexical.joblib"
COLUMNS_FILE = HERE / "columns_lexical.joblib"
# If the model files are missing, the model is trained once from the dataset instead.
DATA_FOLDERS = [HERE, HERE.parent / "1 - Data Set and processing"]
# These 13 features need live internet lookups, so the app does not use them.
NETWORK_COLS = ["time_response", "domain_spf", "asn_ip", "time_domain_activation",
                "time_domain_expiration", "qty_ip_resolved", "qty_nameservers",
                "qty_mx_servers", "ttl_hostname", "tls_ssl_certificate",
                "qty_redirects", "url_google_index", "domain_google_index"]
PHISHING_LABEL = 1  # phishing_train.csv

 
st.set_page_config(page_title="AI Phishing Detection System", page_icon="🛡️", layout="centered")
 
 
def train_from_csv():
    """Same model as in the notebook: Random Forest (200 trees) on the 98 URL-text features."""
    from sklearn.ensemble import RandomForestClassifier
    for folder in DATA_FOLDERS:
        csv_path = folder / "phishing_train.csv"
        if csv_path.exists():
            break
    else:
        raise FileNotFoundError("rf_lexical.joblib was not found and neither was phishing_train.csv. "
                                "Put one of them next to app.py.")
    train = pd.read_csv(csv_path)
    x_train = train.drop(columns=["phishing"])
    cols = [c for c in x_train.columns if c not in NETWORK_COLS]
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(x_train[cols], train["phishing"])
    return rf, cols


@st.cache_resource(show_spinner="Training the model from the dataset (first start only, can take up to a minute)...")
def load_assets():
    if MODEL_FILE.exists() and COLUMNS_FILE.exists():
        try:
            return joblib.load(MODEL_FILE), joblib.load(COLUMNS_FILE)
        except Exception:
            pass          # e.g. the files were saved with another scikit-learn version
    return train_from_csv()
 
 
model, EXPECTED_COLS = load_assets()
 
# feature extraction
# Rebuilds the dataset's lexical features from a raw URL. Two conventions of the
# training data matter: URLs have no "http(s)://" prefix, and a missing part

SPECIAL = {
    "dot": ".", "hyphen": "-", "underline": "_", "slash": "/", "questionmark": "?",
    "equal": "=", "at": "@", "and": "&", "exclamation": "!", "space": " ",
    "tilde": "~", "comma": ",", "plus": "+", "asterisk": "*", "hashtag": "#",
    "dollar": "$", "percent": "%",
}
TLDS = {"com", "net", "org", "info", "biz", "edu", "gov", "io", "co", "uk", "de", "ru", "cn",
        "tk", "xyz", "top", "site", "online", "club", "app", "us", "ca", "au", "in", "br",
        "fr", "it", "nl", "es", "jp"}
SHORTENERS = ["bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly", "cutt.ly"]
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
 
 
def clean_url(raw):
    url = raw.strip().strip("[]\"'")
    return re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
 
 
def extract_lexical_features(raw):
    url = clean_url(raw)
    m = re.match(r"([^/?#]*)([^?#]*)(?:\?([^#]*))?", url)
    domain, path, query = m.group(1), m.group(2), m.group(3) or ""
    cut = path.rfind("/")
    directory = path[:cut] if cut > 0 else ""
    file_name = path[cut + 1:] if cut >= 0 else path
 
    parts = {"url": url, "domain": domain, "directory": directory, "file": file_name, "params": query}
    absent = {"directory": not directory, "file": not file_name, "params": not query}
    host = domain.split(":")[0].lower()
 
    feats = {}
    for col in EXPECTED_COLS:
        m2 = re.fullmatch(r"qty_(.+)_(url|domain|directory|file|params)", col)
        if m2 and m2.group(1) in SPECIAL:
            part = m2.group(2)
            feats[col] = -1 if absent.get(part) else parts[part].count(SPECIAL[m2.group(1)])
        elif col == "length_url":
            feats[col] = len(url)
        elif col == "domain_length":
            feats[col] = len(domain)
        elif col in ("directory_length", "file_length", "params_length"):
            part = col.replace("_length", "")
            feats[col] = -1 if absent[part] else len(parts[part])
        elif col == "qty_vowels_domain":
            feats[col] = sum(c in "aeiou" for c in domain.lower())
        elif col == "domain_in_ip":
            feats[col] = int(bool(IP_RE.match(host)))
        elif col == "server_client_domain":
            feats[col] = int("server" in host or "client" in host)
        elif col == "qty_tld_url":
            feats[col] = sum(lbl in TLDS for lbl in host.split(".")[1:]) or 1
        elif col == "tld_present_params":
            feats[col] = -1 if absent["params"] else int(bool(re.search(r"\.(?:%s)\b" % "|".join(TLDS), query)))
        elif col == "qty_params":
            feats[col] = -1 if absent["params"] else len(query.split("&"))
        elif col == "email_in_url":
            feats[col] = int(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", url)))
        elif col == "url_shortened":
            feats[col] = int(any(s == host for s in SHORTENERS))
        else:
            feats[col] = 0
    return feats
 
 
def red_flags(raw):
    """Simple human-readable indicators shown next to the model's verdict."""
    url = clean_url(raw)
    host = url.split("/")[0].split("?")[0].split(":")[0].lower()
    flags = []
    if IP_RE.match(host):
        flags.append("Uses an IP address instead of a domain name")
    if "@" in url:
        flags.append("Contains an '@' symbol (can hide the real destination)")
    if host in SHORTENERS:
        flags.append("Shortened link (hides the real destination)")
    if host.count("-") >= 3:
        flags.append("Suspicious amount of hyphens in the domain name")
    if len(url) > 75:
        flags.append("Unusually long URL")
    if re.search(r"[a-z][013]|[013][a-z]", host.split(".")[0]) and re.search(r"[a-z]", host):
        flags.append("Digits mixed into the name (possible look-alike, e.g. paypa1)")
    if raw.strip().lower().startswith("http://"):
        flags.append("Lack of HTTPS")
    return flags
 
 
# UI BLOCK
st.title("🛡️ AI-Powered Phishing Detection System")
st.markdown("---")
url_input = st.text_input("Enter Web URL to Analyze:", placeholder="e.g. https://www.wikipedia.org")
 
if st.button("Analyze URL"):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        row = pd.DataFrame([extract_lexical_features(url_input)], columns=EXPECTED_COLS)
        p_phish = float(model.predict_proba(row)[0][list(model.classes_).index(PHISHING_LABEL)])
        flags = red_flags(url_input)
 
        risk = "High" if p_phish >= 0.7 else "Medium" if p_phish >= 0.4 else "Low"
        if flags and risk == "Low":
            risk = "Medium"
 
        if p_phish >= 0.5:
            st.error(f"🚨 **Phishing URL detected** (phishing probability: {p_phish * 100:.1f}%)")
        else:
            st.success(f"✅ **Looks legitimate** (phishing probability: {p_phish * 100:.1f}%)")
 
        st.metric("Risk level", risk)
        if risk == "High":
            st.error("**Do not open this link.** Do not enter passwords or personal details. Consider reporting it to your IT/security team.")
        elif risk == "Medium":
            st.warning("**Be careful.** Do not click it unless you can verify the sender. We recommend typing the official website address instead.")
        else:
            st.info("No major warning signs, but stay careful with unknown or unexpected links")
 
        if flags:
            st.markdown("**Warning signs found:**")
            for f in flags:
                st.markdown(f"- {f}")
 