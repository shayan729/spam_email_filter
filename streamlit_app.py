import streamlit as st
import re
import os
import joblib
import quopri
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import nltk
import requests
import time

# --- NLP IMPORTS ---
from textblob import TextBlob
import spacy

# --- Page Configuration (Run only once) ---
st.set_page_config(
    page_title="Spam Email Filter",
    page_icon="🛡️",
    layout="wide"
)

# --- Resource Loading (Cached for performance) ---
@st.cache_resource
def load_resources():
    """Loads all models and data, assuming they are pre-installed via requirements.txt."""
    # NLTK data handling
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        st.info("NLTK data not found. Downloading...")
        nltk.download('stopwords')
        nltk.download('wordnet')

    # Load models
    try:
        svm_model = joblib.load('svm_spam_model.joblib')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        # spaCy model is now guaranteed to be installed by requirements.txt
        nlp_model = spacy.load("en_core_web_sm")
        return svm_model, tfidf_vectorizer, nlp_model
    except (FileNotFoundError, IOError) as e:
        st.error(f"A critical resource is missing: {e}. Please ensure all model files are in the GitHub repository.")
        return None, None, None

svm_model, tfidf_vectorizer, nlp = load_resources()
if not all([svm_model, tfidf_vectorizer, nlp]):
    st.stop() # Stop execution if resources failed to load

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Load API Key from Streamlit Secrets ---
VT_API_KEY = st.secrets.get("VT_API_KEY", os.getenv("VT_API_KEY"))

# --- All Backend Logic Functions ---
def parse_eml(file_bytes):
    msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
    subject, html, text = msg.get('subject', ''), "", ""; links, attachments = set(), []
    for part in msg.walk():
        if part.get_content_disposition() == 'attachment':
            if filename := part.get_filename(): attachments.append(filename)
            continue
        if payload := part.get_payload(decode=True):
            try: decoded = payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
            except Exception: decoded = quopri.decodestring(payload).decode('utf-8', errors='replace')
            if part.get_content_type() == "text/plain": text += decoded
            elif part.get_content_type() == "text/html": html += decoded
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            if href := a.get('href', '').strip():
                if not href.startswith(('mailto:', 'tel:')): links.add(href)
        for tag in soup(["script", "style"]): tag.decompose()
        text += "\n" + soup.get_text(separator='\n', strip=True)
    links.update(re.findall(r'https?://[^\s/$.?#].[^\s]*', text))
    return {"full_text": f"Subject: {subject}\n{text}".strip(), "links": sorted(list(links)), "attachments": attachments}

def extract_from_text(email_text):
    links = set(re.findall(r'https?://[^\s/$.?#].[^\s]*', email_text))
    attachments = set(re.findall(r'filename="?([^"]+)"?', email_text, re.IGNORECASE))
    return {"links": sorted(list(links)), "attachments": sorted(list(attachments))}

def analyze_headers(text):
    if match := re.search(r'subject:(.*)', text, re.IGNORECASE):
        for kw in ['free', 'winner', 'congratulations', 'claim', 'prize', 'urgent']:
            if kw in match.group(1).lower(): return (True, f"Suspicious keyword '{kw}' in subject.")
    return (False, "Passed.")

def analyze_content_nlp(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.8: return (True, "Extreme positive sentiment (e.g., lottery win).")
    if blob.sentiment.polarity < -0.6: return (True, "Extreme negative sentiment (e.g., threat).")
    urgency_score = sum(1 for word in ['urgent', 'required', 'action', 'expires', 'immediately', 'verify', 'suspended'] if word in text.lower())
    if urgency_score >= 3: return (True, f"High urgency score ({urgency_score}) detected.")
    doc = nlp(text)
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    if organizations:
        for keyword in ['password', 'login', 'username', 'account', 'credential']:
            if keyword in text.lower(): return (True, f"Brand impersonation: '{organizations[0]}' mentioned with credential request ('{keyword}').")
    return (False, "Passed.")

def scan_links_attachments(text):
    if short := re.findall(r'(bit\.ly|tinyurl\.com|goo\.gl)', text): return (True, f"Found suspicious shortened link: '{short[0]}'.")
    if suspicious := re.findall(r'(\w+\.(exe|zip|scr|msi))', text, re.IGNORECASE): return (True, f"Found suspicious attachment: '{suspicious[0][0]}'.")
    return (False, "Passed.")

def predict_with_ml_model(text):
    # --- BUG FIX: Defined preprocess function inside for correct scoping ---
    lem, sw = WordNetLemmatizer(), set(stopwords.words('english'))
    def preprocess(t):
        t = re.sub('[^a-zA-Z]', ' ', t).lower()
        return ' '.join([lem.lemmatize(w) for w in t.split() if w not in sw])
    pred = svm_model.predict(tfidf_vectorizer.transform([preprocess(text)]).toarray())
    return (True, "Classified as SPAM.") if pred[0] == 'spam' else (False, "Classified as HAM.")

def perform_analysis(email_text):
    pipeline = [{'name': 'Header', 'fn': analyze_headers}, {'name': 'Content', 'fn': analyze_content_nlp}, {'name': 'Link/Attach', 'fn': scan_links_attachments}, {'name': 'ML Model', 'fn': predict_with_ml_model}]
    results, final_verdict, found_spam = [], 'ham', False
    for stage in pipeline:
        if not found_spam:
            is_spam, reason = stage['fn'](email_text)
            if is_spam: status, final_verdict, found_spam = 'failed', 'spam', True
            else: status = 'passed'
        else: status, reason = 'skipped', 'Skipped due to earlier detection.'
        results.append({'stage': stage['name'], 'status': status, 'reason': reason})
    return {'verdict': final_verdict, 'stages': results}

def submit_to_vt(endpoint, data=None, files=None):
    if not VT_API_KEY: st.error("VirusTotal API key is not configured in secrets."); return None
    headers = {"accept": "application/json", "x-apikey": VT_API_KEY}
    try:
        response = requests.post(f"https://www.virustotal.com/api/v3/{endpoint}", data=data, files=files, headers=headers)
        if response.status_code == 409:
            st.info("Item already analyzed by VT. Fetching existing report.")
            return response.json()
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e: st.error(f"VirusTotal API Error: {e}"); return None

def get_vt_report(analysis_id):
    headers = {"accept": "application/json", "x-apikey": VT_API_KEY}
    try:
        response = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}", headers=headers)
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException: return None

# --- STREAMLIT UI ---
st.title("🛡️ Spam Email Analyzer")
st.markdown("An intelligent, multi-stage tool to detect email threats using NLP and Machine Learning.")

# Initialize session state
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
if 'original_file' not in st.session_state: st.session_state.original_file = None

tab1, tab2 = st.tabs(["Paste Raw Content", "Upload .eml File"])
with tab1:
    email_text_input = st.text_area("Paste the full email source:", height=300, key="email_text")
    if st.button("Analyze Pasted Text", type="primary"):
        if email_text_input:
            with st.spinner('Analyzing...'):
                st.session_state.original_file = None
                # --- BUG FIX: Create a consistent data dictionary ---
                analysis_data = extract_from_text(email_text_input)
                analysis_data["full_text"] = email_text_input
                analysis_data.update(perform_analysis(email_text_input))
                st.session_state.analysis_result = analysis_data
        else: st.warning("Please paste email content to analyze.")
with tab2:
    uploaded_file = st.file_uploader("Choose a .eml file", type=['eml'], key="email_file")
    if st.button("Analyze .eml File", type="primary"):
        if uploaded_file:
            with st.spinner('Analyzing...'):
                st.session_state.original_file = uploaded_file
                file_bytes = uploaded_file.getvalue()
                # --- BUG FIX: Create a consistent data dictionary ---
                analysis_data = parse_eml(file_bytes)
                analysis_data.update(perform_analysis(analysis_data['full_text']))
                st.session_state.analysis_result = analysis_data
        else: st.warning("Please upload a .eml file to analyze.")

# --- Display Results ---
if st.session_state.analysis_result:
    st.divider()
    st.header("Analysis Results", anchor=False)
    res = st.session_state.analysis_result
    if res['verdict'] == 'spam': st.error("**Verdict: SPAM**", icon="🚨")
    else: st.success("**Verdict: HAM (Looks Safe)**", icon="✅")

    with st.expander("View Analysis Pipeline Details"):
        for stage in res['stages']:
            icon = "✅" if stage['status'] == 'passed' else "🚨" if stage['status'] == 'failed' else "⏩"
            st.info(f"**{stage['stage']} Analysis:** {stage['reason']}", icon=icon)

    col1, col2 = st.columns(2)
    with col1:
        if res['links']:
            st.subheader("Extracted Links", anchor=False)
            for link in res['links']:
                if st.button(f"Scan: {link}", key=f"link_{link}"):
                    response = submit_to_vt("urls", data={"url": link})
                    if response: st.session_state[f"vt_poll_{response['data']['id']}"] = 'pending'
    with col2:
        if res['attachments'] and st.session_state.original_file:
            st.subheader("Detected Attachments", anchor=False)
            for attachment in res['attachments']:
                if st.button(f"Scan: {attachment}", key=f"att_{attachment}"):
                    file_obj = st.session_state.original_file
                    file_obj.seek(0)
                    response = submit_to_vt("files", files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)})
                    if response:
                        analysis_id = response.get('error', {}).get('id') or response['data']['id']
                        st.session_state[f"vt_poll_{analysis_id}"] = 'pending'

# --- VT Polling and Display Logic ---
for key, value in st.session_state.items():
    if key.startswith('vt_poll_'):
        analysis_id = key.split('_')[-1]
        if value == 'pending':
            with st.spinner(f"Waiting for VirusTotal report ({analysis_id[:10]}...)..."):
                report = get_vt_report(analysis_id)
                if report and report['data']['attributes']['status'] == 'completed':
                    st.session_state[key] = report
                    st.rerun()
                else:
                    time.sleep(5)
                    st.rerun()
        elif isinstance(value, dict): # Report is complete
            report = value
            stats = report['data']['attributes'].get('stats', {})
            is_file = "file_info" in report.get("meta", {})
            st.subheader(f"VirusTotal Report ({analysis_id[:10]}...)", anchor=False)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Malicious", stats.get('malicious', 0)); c2.metric("Suspicious", stats.get('suspicious', 0))
            c3.metric("Undetected", stats.get('undetected', 0)); c4.metric("Harmless", stats.get('harmless', 0))
            if is_file:
                sha256 = report['meta']['file_info']['sha256']
                st.markdown(f"[View Full Report](https://www.virustotal.com/gui/file/{sha256})")
            else:
                url_id = report['data']['links']['item'].split('/')[-1]
                st.markdown(f"[View Full Report](https://www.virustotal.com/gui/url/{url_id})")

