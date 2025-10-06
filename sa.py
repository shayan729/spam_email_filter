import streamlit as st
import re
import os
import joblib
import uuid
import quopri
import requests
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import nltk
from dotenv import load_dotenv
from textblob import TextBlob
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Load environment variables ---
load_dotenv()
VT_API_KEY = os.getenv("VT_API_KEY")

# --- Page Config ---
st.set_page_config(page_title="Spam Email Filter", layout="wide")

st.title("📧 Spam Email Analyzer")
st.write("Analyze `.eml` files or plain email text using ML, NLP, and VirusTotal integration.")

# --- Load Models ---
@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_spam_model.joblib")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    nlp = spacy.load("en_core_web_sm")
    return svm_model, tfidf_vectorizer, nlp

try:
    svm_model, tfidf_vectorizer, nlp = load_models()
    st.sidebar.success("✅ Models Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model Load Error: {e}")
    st.stop()

# --- Ensure NLTK Data ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

lem = WordNetLemmatizer()
sw = set(stopwords.words('english'))

# --- Helper Functions ---
def parse_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject, html, text = msg.get('subject', ''), "", ""
    links, attachments = set(), []
    for part in msg.walk():
        if part.get_content_disposition() == 'attachment':
            if filename := part.get_filename():
                attachments.append(filename)
            continue
        if payload := part.get_payload(decode=True):
            try:
                decoded = payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
            except Exception:
                decoded = quopri.decodestring(payload).decode('utf-8', errors='replace')
            if part.get_content_type() == "text/plain":
                text += decoded
            elif part.get_content_type() == "text/html":
                html += decoded
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            if href := a.get('href', '').strip():
                if not href.startswith(('mailto:', 'tel:')):
                    links.add(href)
        for tag in soup(["script", "style"]):
            tag.decompose()
        text += "\n" + soup.get_text(separator='\n', strip=True)
    links.update(re.findall(r'https?://[^\s/$.?#].[^\s]*', text))
    return {
        "full_text": f"Subject: {subject}\n{text}".strip(),
        "links": sorted(list(links)),
        "attachments": attachments
    }

def extract_from_text(email_text):
    links = set(re.findall(r'https?://[^\s/$.?#].[^\s]*', email_text))
    attachments = set(re.findall(r'filename="?([^"]+)"?', email_text, re.IGNORECASE))
    return {"links": sorted(list(links)), "attachments": sorted(list(attachments))}

def analyze_headers(text):
    if match := re.search(r'subject:(.*)', text, re.IGNORECASE):
        for kw in ['free', 'winner', 'congratulations', 'claim', 'prize', 'urgent']:
            if kw in match.group(1).lower():
                return (True, f"Suspicious keyword '{kw}' in subject.")
    return (False, "Passed.")

def analyze_content_nlp(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.8:
        return (True, "Extreme positive sentiment detected.")
    if blob.sentiment.polarity < -0.6:
        return (True, "Extreme negative sentiment detected.")
    urgency_keywords = ['urgent', 'required', 'important', 'action', 'expires', 'immediately', 'verify', 'suspended']
    urgency_score = sum(1 for word in urgency_keywords if word in text.lower())
    if urgency_score >= 3:
        return (True, f"High urgency score ({urgency_score}).")
    doc = nlp(text)
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    credential_keywords = ['password', 'login', 'username', 'account', 'credential']
    if organizations:
        found_org = organizations[0]
        for keyword in credential_keywords:
            if keyword in text.lower():
                return (True, f"Brand impersonation detected: '{found_org}' with credential request.")
    return (False, "Passed.")

def scan_links_attachments(text):
    if short := re.findall(r'(bit\.ly|tinyurl\.com|goo\.gl)', text):
        return (True, f"Found suspicious shortened link: '{short[0]}'.")
    if suspicious := re.findall(r'(\w+\.(exe|zip|scr|msi))', text, re.IGNORECASE):
        return (True, f"Found suspicious attachment: '{suspicious[0][0]}'.")
    return (False, "Passed.")

def predict_with_ml_model(text):
    def preprocess(t):
        t = re.sub('[^a-zA-Z]', ' ', t).lower()
        return ' '.join([lem.lemmatize(w) for w in t.split() if w not in sw])
    pred = svm_model.predict(tfidf_vectorizer.transform([preprocess(text)]).toarray())
    return (True, "Classified as SPAM.") if pred[0] == 'spam' else (False, "Classified as HAM.")

def perform_analysis(email_text):
    pipeline = [
        {'name': 'Header', 'fn': analyze_headers},
        {'name': 'Content', 'fn': analyze_content_nlp},
        {'name': 'Link/Attach', 'fn': scan_links_attachments},
        {'name': 'ML Model', 'fn': predict_with_ml_model}
    ]
    results, final_verdict, found_spam = [], 'ham', False
    for stage in pipeline:
        if not found_spam:
            is_spam, reason = stage['fn'](email_text)
            if is_spam:
                status, final_verdict, found_spam = 'failed', 'spam', True
            else:
                status = 'passed'
        else:
            status, reason = 'skipped', 'Skipped due to earlier detection.'
        results.append({'stage': stage['name'], 'status': status, 'reason': reason})
    return {'verdict': final_verdict, 'stages': results}

# --- Sidebar Input ---
mode = st.sidebar.radio("Choose Mode:", ["Analyze Text", "Upload .eml File"])

if mode == "Analyze Text":
    email_text = st.text_area("Paste Email Text Here:", height=250)
    if st.button("🔍 Analyze"):
        if not email_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                res = perform_analysis(email_text)
                meta = extract_from_text(email_text)
            st.subheader(f"Verdict: {'🚫 SPAM' if res['verdict']=='spam' else '✅ HAM'}")
            for stage in res["stages"]:
                st.markdown(f"**{stage['stage']}** — {stage['status'].upper()} : {stage['reason']}")
            st.write("**Links Found:**", meta["links"])
            st.write("**Attachments Found:**", meta["attachments"])

elif mode == "Upload .eml File":
    eml_file = st.file_uploader("Upload .eml File", type=["eml"])
    if eml_file and st.button("🔍 Analyze File"):
        temp_path = f"temp_{uuid.uuid4()}.eml"
        with open(temp_path, "wb") as f:
            f.write(eml_file.read())
        data = parse_eml(temp_path)
        res = perform_analysis(data['full_text'])
        os.remove(temp_path)
        st.subheader(f"Verdict: {'🚫 SPAM' if res['verdict']=='spam' else '✅ HAM'}")
        for stage in res["stages"]:
            st.markdown(f"**{stage['stage']}** — {stage['status'].upper()} : {stage['reason']}")
        st.write("**Links Found:**", data["links"])
        st.write("**Attachments:**", data["attachments"])
