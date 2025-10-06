import os
import re
import quopri
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

def extract_from_eml(file_path, output_dir="attachments"):
    # --- Step 1: Read and parse the .eml file ---
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # --- Step 2: Extract metadata ---
    subject = msg.get('subject', '')
    sender = msg.get('from', '')
    to = msg.get('to', '')
    date = msg.get('date', '')

    # --- Step 3: Initialize containers ---
    text_content = ""
    html_content = ""

    # --- Step 4: Walk through parts ---
    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = part.get_content_disposition()

        # skip attachments
        if content_disposition == 'attachment':
            continue

        # decode payload
        payload = part.get_payload(decode=True)
        if not payload:
            continue

        # decode quoted-printable if necessary
        try:
            payload_decoded = payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
        except Exception:
            payload_decoded = quopri.decodestring(payload).decode('utf-8', errors='replace')

        if content_type == "text/plain":
            text_content += payload_decoded
        elif content_type == "text/html":
            html_content += payload_decoded

    # --- Step 5: Clean and extract readable text from HTML ---
    clean_text = text_content
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        # get only visible text (strip scripts, styles)
        for tag in soup(["script", "style"]):
            tag.decompose()
        visible_text = soup.get_text(separator='\n', strip=True)
        clean_text += "\n" + visible_text

    # --- Step 6: Extract URLs from both plain & HTML ---
    links = set()
    links.update(re.findall(r'(https?://\S+)', clean_text))
    if html_content:
        for a in soup.find_all('a', href=True):
            links.add(a['href'])

    # --- Step 7: Save attachments ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    attachments = []
    for part in msg.iter_attachments():
        filename = part.get_filename()
        if filename:
            file_path_out = os.path.join(output_dir, filename)
            with open(file_path_out, 'wb') as fp:
                fp.write(part.get_payload(decode=True))
            attachments.append(file_path_out)

    # --- Step 8: Return structured data ---
    return {
        "subject": subject.strip(),
        "from": sender.strip(),
        "to": to.strip(),
        "date": date.strip(),
        "text_content": clean_text.strip(),
        "links": list(links),
        "attachments": attachments
    }

# Example usage
if __name__ == "__main__":
    eml_path = "sample_mail.eml"
    data = extract_from_eml(eml_path)

    print("Subject:", data['subject'])
    print("From:", data['from'])
    print("To:", data['to'])
    print("Date:", data['date'])
    print("\nExtracted Links:", data['links'])
    print("\nAttachments Saved:", data['attachments'])
    print("\nExtracted Email Text:\n") 
    print(data['text_content'][:1000])  

import requests

url = "https://www.virustotal.com/api/v3/files"

files = { "file": ("tech_assessment_confirmation.pdf", open("tech_assessment_confirmation.pdf", "rb"), "application/pdf") }
payload = { "password": "" }
headers = {
    "accept": "application/json",
    "x-apikey": VT_API_KEY
}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.text)