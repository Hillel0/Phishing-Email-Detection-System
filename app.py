import os
from flask import Flask, render_template, request, jsonify
from assignment1 import PhishingDetector
from flask_cors import CORS

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)
detector = PhishingDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    email_content = request.form.get('email_content')
    if not email_content:
        return jsonify({'error': 'No email content provided'}), 400

    # Get analysis results from the detector
    results = detector.analyze_email(email_content)
    
    # Add a simple AI prediction based on the risk score
    ai_opinion = "Likely Phishing" if results['risk_score'] > 50 else "Likely Safe"

    # Return all results including an AI opinion
    return jsonify({
        'risk_score': results['risk_score'],
        'is_suspicious': results['is_suspicious'],
        'indicators': results['indicators'],
        'ai_prediction': ai_opinion
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500, debug=True)