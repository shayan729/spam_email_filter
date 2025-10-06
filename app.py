import re
import os
import joblib
from flask import Flask, render_template, request, jsonify
import uuid
import quopri
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import nltk
import requests
from dotenv import load_dotenv
import subprocess
# --- NEW: Import NLP libraries ---
from textblob import TextBlob
import spacy

load_dotenv()
VT_API_KEY = os.getenv("VT_API_KEY")

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Models & NLTK/spaCy Data ---
try:
    svm_model = joblib.load('svm_spam_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("✅ Model and vectorizer loaded successfully.")

except (FileNotFoundError, IOError) as e:
    print(f"❌ Critical startup error: {e}. Please ensure model files and spaCy model are present.")
    exit()
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    print("✅ NLTK data found.")
except LookupError:
    print("⚠️ NLTK data not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- (Parser and other analysis stages remain unchanged) ---
def parse_eml(file_path):
    with open(file_path, 'rb') as f: msg = BytesParser(policy=policy.default).parse(f)
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

# --- NEW: Upgraded Content Analysis with NLP ---
def analyze_content_nlp(text):
    """
    Performs advanced content analysis using NLP.
    - Sentiment Analysis (emotional manipulation)
    - Urgency Score (pressure tactics)
    - Named Entity Recognition (brand impersonation)
    """
    # 1. Sentiment Analysis with TextBlob
    blob = TextBlob(text)
    # Polarity is between -1 (very negative) and 1 (very positive)
    # Subjectivity is between 0 (objective) and 1 (subjective)
    if blob.sentiment.polarity > 0.8:
        return (True, "Extreme positive sentiment detected (e.g., lottery win).")
    if blob.sentiment.polarity < -0.6:
        return (True, "Extreme negative sentiment detected (e.g., threat, account closure).")

    # 2. Urgency Score
    urgency_keywords = ['urgent', 'required', 'important', 'action', 'expires', 'immediately', 'verify', 'suspended']
    urgency_score = sum(1 for word in urgency_keywords if word in text.lower())
    if urgency_score >= 3:
        return (True, f"High urgency score ({urgency_score}) indicating pressure tactics.")

    # 3. Named Entity Recognition for Brand Impersonation (with spaCy)
    doc = nlp(text)
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    credential_keywords = ['password', 'login', 'username', 'account', 'credential']
    
    if organizations:
        # Check if a known organization is mentioned alongside a request for credentials
        found_org = organizations[0] # Focus on the first detected org
        for keyword in credential_keywords:
            if keyword in text.lower():
                return (True, f"Brand impersonation detected: '{found_org}' mentioned with credential request ('{keyword}').")

    return (False, "Passed.")

def scan_links_attachments(text):
    if short := re.findall(r'(bit\.ly|tinyurl\.com|goo\.gl)', text): return (True, f"Found suspicious shortened link: '{short[0]}'.")
    if suspicious := re.findall(r'(\w+\.(exe|zip|scr|msi))', text, re.IGNORECASE): return (True, f"Found suspicious attachment: '{suspicious[0][0]}'.")
    return (False, "Passed.")
def predict_with_ml_model(text):
    lem, sw = WordNetLemmatizer(), set(stopwords.words('english'))
    def preprocess(t):
        t = re.sub('[^a-zA-Z]', ' ', t).lower()
        return ' '.join([lem.lemmatize(w) for w in t.split() if w not in sw])
    pred = svm_model.predict(tfidf_vectorizer.transform([preprocess(text)]).toarray())
    return (True, "Classified as SPAM.") if pred[0] == 'spam' else (False, "Classified as HAM.")

# --- UPDATED: Analysis Pipeline to use the new NLP function ---
def perform_analysis(email_text):
    pipeline = [
        {'name': 'Header', 'fn': analyze_headers},
        {'name': 'Content', 'fn': analyze_content_nlp}, # Using the new NLP function here
        {'name': 'Link/Attach', 'fn': scan_links_attachments},
        {'name': 'ML Model', 'fn': predict_with_ml_model}
    ]
    results, final_verdict, found_spam = [], 'ham', False
    for stage in pipeline:
        if not found_spam:
            is_spam, reason = stage['fn'](email_text)
            if is_spam: status, final_verdict, found_spam = 'failed', 'spam', True
            else: status = 'passed'
        else: status, reason = 'skipped', 'Skipped due to earlier detection.'
        results.append({'stage': stage['name'], 'status': status, 'reason': reason})
    return {'verdict': final_verdict, 'stages': results}


# --- (All Flask routes remain unchanged) ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/analyze_text', methods=['POST'])
def analyze_text_route():
    text = request.json.get('email_text', '');
    if not text: return jsonify({'error': 'No text provided.'}), 400
    res = perform_analysis(text); res.update(extract_from_text(text)); return jsonify(res)
@app.route('/analyze_file', methods=['POST'])
def analyze_file_route():
    if 'eml_file' not in request.files: return jsonify({'error': 'No file part.'}), 400
    file = request.files['eml_file']
    if file.filename == '' or not file.filename.endswith('.eml'): return jsonify({'error': 'Invalid file.'}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    try:
        file.save(path); data = parse_eml(path); res = perform_analysis(data['full_text']); res.update(data); return jsonify(res)
    except Exception as e: return jsonify({'error': f'Failed to process file: {e}'}), 500
    finally:
        if os.path.exists(path): os.remove(path)
@app.route('/scan_url', methods=['POST'])
def scan_url():
    if not VT_API_KEY: return jsonify({'error': 'API key not configured.'}), 500
    url_to_scan = request.json.get('url')
    if not url_to_scan: return jsonify({'error': 'URL not provided.'}), 400
    vt_url = "https://www.virustotal.com/api/v3/urls"
    payload = { "url": url_to_scan }
    headers = { "accept": "application/json", "x-apikey": VT_API_KEY, "content-type": "application/x-www-form-urlencoded" }
    try:
        response = requests.post(vt_url, data=payload, headers=headers); response.raise_for_status(); return jsonify(response.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409: return jsonify(e.response.json())
        return jsonify({'error': f'VirusTotal submission failed: {e}'}), e.response.status_code
    except requests.exceptions.RequestException as e: return jsonify({'error': f'VirusTotal connection failed: {e}'}), 500
@app.route('/scan_attachment', methods=['POST'])
def scan_attachment():
    if not VT_API_KEY: return jsonify({'error': 'API key not configured.'}), 500
    if 'file' not in request.files: return jsonify({'error': 'No file provided.'}), 400
    file_to_scan = request.files['file']
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    file_to_scan.save(temp_path)
    try:
        with open(temp_path, "rb") as f:
            vt_url = "https://www.virustotal.com/api/v3/files"
            files = { "file": (file_to_scan.filename, f, file_to_scan.content_type) }
            headers = { "accept": "application/json", "x-apikey": VT_API_KEY }
            try:
                response = requests.post(vt_url, files=files, headers=headers); response.raise_for_status(); return jsonify(response.json())
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 409:
                    error_data = e.response.json()
                    if existing_id := error_data.get('error', {}).get('id'): return jsonify({'data': {'id': existing_id}})
                return jsonify({'error': f'VirusTotal submission failed: {e}'}), e.response.status_code
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
@app.route('/get_vt_report/<analysis_id>', methods=['GET'])
def get_vt_report(analysis_id):
    if not VT_API_KEY: return jsonify({'error': 'API key not configured.'}), 500
    vt_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
    headers = { "accept": "application/json", "x-apikey": VT_API_KEY }
    try:
        response = requests.get(vt_url, headers=headers); response.raise_for_status(); return jsonify(response.json())
    except requests.exceptions.RequestException as e: return jsonify({'error': f'VirusTotal report retrieval failed: {e}'}), 500

if __name__ == "__main__":
    app.run(debug=True)

