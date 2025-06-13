from flask import Flask, request, render_template_string
import joblib
import re
import os
import subprocess

# Run spam2.py only if models don't exist
if not all([os.path.exists(p) for p in ["spam_classifier_model.joblib", "tfidf_vectorizer.joblib"]]):
    subprocess.run(["python", "spam2.py"])

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("spam_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# HTML template
HTML_PAGE = """
<!doctype html>
<title>Email Spam Detector</title>
<h2>Enter email content to detect spam</h2>
<form method=post action="/predict">
  <textarea name=text rows="10" cols="70" placeholder="Paste your email content here..."></textarea><br><br>
  <input type=submit value="Check for Spam">
</form>
{% if prediction is not none %}
  <h3>Prediction: <span style="color:{{ color }}">{{ prediction }}</span></h3>
{% endif %}
"""

# Text cleaning function (same as in spam2.py)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # Remove non-alphabetic chars
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra whitespace
    return text

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    raw_text = request.form.get("text", "")
    cleaned_text = clean_text(raw_text)
    vect = vectorizer.transform([cleaned_text])
    pred = model.predict(vect)[0]

    label = "Spam" if pred == 1 else "Not Spam"
    color = "red" if pred == 1 else "green"

    return render_template_string(HTML_PAGE, prediction=label, color=color)

if __name__ == '__main__':
    app.run(debug=True)
