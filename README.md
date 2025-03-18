# Phishing Email Detection System

A comprehensive phishing detection system that combines rule-based analysis with machine learning capabilities to identify email-based phishing attempts. This project was developed as part of a coding assignment for a job interview, with extended functionality beyond the original requirements.

## 📋 Project Overview

This project implements a phishing detection system for email content that integrates traditional rule-based detection with advanced machine learning techniques. The system analyzes incoming emails to identify phishing attempts using multiple detection methods and provides clear, actionable results.
The assignment required building a tool that scans email content for common phishing indicators, with the core requirements being:

1. Accept a text file containing email content
2. Detect suspicious links, spoofed sender addresses, and urgent language
3. Provide a summary of detected indicators

I completed all the base requirements and additionally implemented a web-based user interface (bonus requirement) along with a custom DistilBERT-based machine learning component as my personal enhancement to the project.

## 🏗️ Architecture

The system consists of three main components:

1. **Phishing Detection Engine (`assignment1.py`):**
   - Rule-based detection engine
   - URL analysis
   - Sender verification
   - Language pattern analysis

2. **User Interface (`app.py` - Bonus):**
   - Flask server
   - Browser-based interface for email analysis

3. **Machine Learning Model (`phishing.py` - Personal Addition):**
   - DistilBERT-based classification model
   - Binary prediction (safe/phishing)
   - Pre-trained model support

## ✨ Key Features

### 1. URL Analysis
- Detection of IP-based URLs
- Identification of uncommon domain TLDs
- Impersonation checking against legitimate domain list
- Detection of minor changes in domain names

### 2. Language Analysis
- Detection of urgent language patterns
- Keyword-based scoring system

### 3. Machine Learning Integration
- DistilBERT transformer model
- Binary classification (safe/phishing)
- Pre-trained model with fine-tuning capability
- For long emails, analyzes first 512 characters
- Real-time prediction integration

### Risk Scoring System
- URL risks - 25 points per suspicious URL
- Urgent language - 15 points per suspicious phrase
- Spoofed sender - 30 points
- Emails with scores ≥ 50 are marked as suspicious

## 🖥️ Implemented Bonus Features

### 1. Browser-Based User Interface
- Clean, intuitive interface
- Real-time analysis feedback
- Risk score visualization

### 2. Machine Learning Enhancement
- AI-based prediction alongside rule-based analysis
- Ability to retrain on new data

## 🚀 System Setup

### 1. Prerequisites
```bash
pip install flask torch transformers pandas scikit-learn datasets
```

### 2. Starting the Server
```bash
python app.py
```
- Server runs at http://127.0.0.1:5500

### 3. Using the System
- Upload or paste email content
- System provides immediate analysis
- Results include:
  - Risk score
  - Identified indicators
  - AI prediction

## 📚 Project Structure

```
project-root/
├── _pycache_/          # Python compiled files
├── results/            # Results output directory
├── saved_model/        # Saved ML model files
├── templates/          # Web interface templates
│   └── index.html      # Main UI template
├── test_dataset/       # Testing data
├── train_dataset/      # Training data
├── app.py              # Web server application
├── assignment1.py      # Core detection engine
├── Phishing_Email.csv  # Email dataset
├── phishing.py         # Machine learning component
├── README.txt          # Original readme file
├── requirements.txt    # Required dependencies
└── t.py                # Test script
```

## 🛡️ Technologies Used

- **Python** - Core programming language
- **Flask** - Web server framework
- **PyTorch & Transformers** - ML infrastructure
- **DistilBERT** - NLP model for text classification
- **Pandas & Scikit-learn** - Data processing

## 🔍 Additional Information

The system uses a multi-layered approach to phishing detection:
1. **Rules-based analysis** examines structural elements like URLs and sender info
2. **Content analysis** identifies suspicious language patterns
3. **AI prediction** leverages machine learning trained on real phishing examples

This combined approach delivers more accurate results than either method alone.

## 🚧 Future Enhancements

- API for integration with email clients
- Support for more languages
- Enhanced model training on larger datasets
- Browser extension for inline email checking
