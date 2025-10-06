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
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy NLP model loaded successfully.")
except (FileNotFoundError, IOError) as e:
    print(f"❌ Critical startup error: {e}. Attempting to download spaCy model...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    except Exception as download_e:
        print(f"❌ Failed to download spaCy model: {download_e}")
        exit()

try:
    nltk.data.find('corpora/stopwords');
    nltk.data.find('corpora/wordnet')
    print("✅ NLTK data found.")
except LookupError:
    print("⚠️ NLTK data not found. Downloading...");
    nltk.download('stopwords');
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --- (Parser and text extraction functions remain unchanged) ---
def parse_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject, html, text = msg.get('subject', ''), "", "";
    links, attachments = set(), []
    for part in msg.walk():
        if part.get_content_disposition() == 'attachment':
            if filename := part.get_filename(): attachments.append(filename)
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
                if not href.startswith(('mailto:', 'tel:')): links.add(href)
        for tag in soup(["script", "style"]): tag.decompose()
        text += "\n" + soup.get_text(separator='\n', strip=True)
    links.update(re.findall(r'https?://[^\s/$.?#].[^\s]*', text))
    return {"full_text": f"Subject: {subject}\n{text}".strip(), "links": sorted(list(links)),
            "attachments": sorted(list(attachments))}


def extract_from_text(email_text):
    links = set(re.findall(r'https?://[^\s/$.?#].[^\s]*', email_text))
    attachments = set(re.findall(r'filename="?([^"]+)"?', email_text, re.IGNORECASE))
    return {"links": sorted(list(links)), "attachments": sorted(list(attachments))}


# --- Analysis Stages (scan_links_attachments is updated) ---
def analyze_headers(text):
    suspicious_keywords = ['free', 'winner', 'congratulations', 'claim', 'prize', 'urgent']
    subject_line = ""
    if match := re.search(r'subject:(.*)', text, re.IGNORECASE): subject_line = match.group(1).lower()
    found_keywords = [kw for kw in suspicious_keywords if kw in subject_line]
    if found_keywords: return {'is_spam': True, 'reason': f"Suspicious keyword '{found_keywords[0]}' in subject."}
    return {'is_spam': False, 'reason': "Subject line appears clean.",
            'score': f"{len(found_keywords)}/{len(suspicious_keywords)} flags"}


def analyze_content_nlp(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.8: return {'is_spam': True, 'reason': "Extreme positive sentiment detected."}
    if polarity < -0.6: return {'is_spam': True, 'reason': "Extreme negative sentiment detected."}
    urgency_keywords = ['urgent', 'required', 'important', 'action', 'expires', 'immediately', 'verify', 'suspended']
    urgency_score = sum(1 for word in urgency_keywords if word in text.lower())
    if urgency_score >= 3: return {'is_spam': True,
                                   'reason': f"High urgency score ({urgency_score}) indicates pressure tactics."}
    doc = nlp(text)
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    if organizations:
        for keyword in ['password', 'login', 'username', 'account', 'credential']:
            if keyword in text.lower(): return {'is_spam': True,
                                                'reason': f"Brand impersonation: '{organizations[0]}' mentioned with credential request ('{keyword}')."}
    score_details = f"Sentiment Polarity: {polarity:.2f} | Urgency Score: {urgency_score}/{len(urgency_keywords)}"
    return {'is_spam': False, 'reason': "Content appears neutral and non-manipulative.", 'score': score_details}


# --- UPDATED: This function now uses the pre-parsed link and attachment lists ---
def scan_links_attachments(text, links, attachments):
    if short := re.findall(r'(bit\.ly|tinyurl\.com|goo\.gl)', text):
        return {'is_spam': True, 'reason': f"Found suspicious shortened link: '{short[0]}'."}

    for attachment_name in attachments:
        if re.search(r'\.(exe|zip|scr|msi)$', attachment_name, re.IGNORECASE):
            return {'is_spam': True, 'reason': f"Found suspicious attachment: '{attachment_name}'."}

    # Use the accurate lengths of the lists for the score
    score = f"0 suspicious items found (scanned {len(links)} links, {len(attachments)} attachments)."
    return {'is_spam': False, 'reason': "No high-risk links or attachments found.", 'score': score}


def predict_with_ml_model(text):
    lem, sw = WordNetLemmatizer(), set(stopwords.words('english'))

    def preprocess(t):
        t = re.sub('[^a-zA-Z]', ' ', t).lower()
        return ' '.join([lem.lemmatize(w) for w in t.split() if w not in sw])

    pred = svm_model.predict(tfidf_vectorizer.transform([preprocess(text)]).toarray())
    if pred[0] == 'spam': return {'is_spam': True, 'reason': "SVM model classified as SPAM."}
    return {'is_spam': False, 'reason': "SVM model classified as HAM.", 'score': 'Classification: HAM'}


# --- UPDATED: The main pipeline now passes the extra data to the analysis stages ---
def perform_analysis(email_text, links, attachments):
    # Note: links and attachments are now passed in
    pipeline = [
        {'name': 'Header', 'fn': lambda text, l, a: analyze_headers(text)},
        {'name': 'Content', 'fn': lambda text, l, a: analyze_content_nlp(text)},
        {'name': 'Link/Attach', 'fn': scan_links_attachments},  # This function will use the extra args
        {'name': 'ML Model', 'fn': lambda text, l, a: predict_with_ml_model(text)}
    ]
    all_results, any_spam_found = [], False
    for stage in pipeline:
        result = stage['fn'](email_text, links, attachments)  # Pass all args to each function
        status = 'failed' if result['is_spam'] else 'passed'
        if result['is_spam']: any_spam_found = True
        all_results.append({
            'stage': stage['name'], 'status': status,
            'reason': result['reason'], 'score': result.get('score', None)
        })
    final_verdict = 'spam' if any_spam_found else 'ham'
    return {'verdict': final_verdict, 'stages': all_results}


# --- UPDATED: Flask routes now pass the parsed data correctly ---
@app.route('/')
def index(): return render_template('index.html')


@app.route('/analyze_text', methods=['POST'])
def analyze_text_route():
    text = request.json.get('email_text', '')
    if not text: return jsonify({'error': 'No text provided.'}), 400

    # Get links/attachments first
    extracted_data = extract_from_text(text)
    links = extracted_data['links']
    attachments = extracted_data['attachments']

    # Pass everything to the analysis function
    res = perform_analysis(text, links, attachments)
    res.update(extracted_data)  # Add lists to the final result for display
    return jsonify(res)


@app.route('/analyze_file', methods=['POST'])
def analyze_file_route():
    if 'eml_file' not in request.files: return jsonify({'error': 'No file part.'}), 400
    file = request.files['eml_file']
    if not file.filename or not file.filename.endswith('.eml'): return jsonify({'error': 'Invalid file.'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    try:
        file.save(path)
        # Get links/attachments first from the reliable parser
        parsed_data = parse_eml(path)
        email_text = parsed_data['full_text']
        links = parsed_data['links']
        attachments = parsed_data['attachments']

        # Pass everything to the analysis function
        res = perform_analysis(email_text, links, attachments)
        res.update(parsed_data)  # Add lists to the final result for display
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {e}'}), 500
    finally:
        if os.path.exists(path): os.remove(path)


# --- (VirusTotal routes remain unchanged) ---
@app.route('/scan_url', methods=['POST'])
def scan_url():
    if not VT_API_KEY: return jsonify({'error': 'API key not configured.'}), 500
    url_to_scan = request.json.get('url')
    if not url_to_scan: return jsonify({'error': 'URL not provided.'}), 400
    vt_url = "https://www.virustotal.com/api/v3/urls"
    payload = {"url": url_to_scan}
    headers = {"accept": "application/json", "x-apikey": VT_API_KEY,
               "content-type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(vt_url, data=payload, headers=headers);
        response.raise_for_status();
        return jsonify(response.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409: return jsonify(e.response.json())
        return jsonify({'error': f'VirusTotal submission failed: {e}'}), e.response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'VirusTotal connection failed: {e}'}), 500


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
            files = {"file": (file_to_scan.filename, f, file_to_scan.content_type)}
            headers = {"accept": "application/json", "x-apikey": VT_API_KEY}
            try:
                response = requests.post(vt_url, files=files, headers=headers);
                response.raise_for_status();
                return jsonify(response.json())
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 409:
                    error_data = e.response.json()
                    if existing_id := error_data.get('error', {}).get('id'): return jsonify(
                        {'data': {'id': existing_id}})
                return jsonify({'error': f'VirusTotal submission failed: {e}'}), e.response.status_code
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.route('/get_vt_report/<analysis_id>', methods=['GET'])
def get_vt_report(analysis_id):
    if not VT_API_KEY: return jsonify({'error': 'API key not configured.'}), 500
    vt_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
    headers = {"accept": "application/json", "x-apikey": VT_API_KEY}
    try:
        response = requests.get(vt_url, headers=headers);
        response.raise_for_status();
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'VirusTotal report retrieval failed: {e}'}), 500


if __name__ == "__main__":
    app.run(debug=True)

