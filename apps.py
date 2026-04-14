import logging
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from resumes_parser import extract_resume_details
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from ats_checker import check_ats_compatibility, extract_text_from_pdf, extract_text_from_docx, benchmark_against_industry, check_file_format
import tempfile
import traceback
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all origins (simplified for Render)
CORS(app)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {str(e)}")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download spaCy model at runtime
def download_spacy_model(model_name="en_core_web_sm"):
    try:
        import spacy
        spacy.load(model_name)
        logger.info(f"spaCy model '{model_name}' is already available")
    except OSError:
        logger.info(f"Downloading spaCy model: {model_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            logger.info(f"Successfully downloaded spaCy model: {model_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download spaCy model '{model_name}': {str(e)}")
            raise Exception(f"Failed to download spaCy model: {model_name}")

# Call this after app initialization
try:
    download_spacy_model()
except Exception as e:
    logger.error(f"spaCy model setup failed: {str(e)}")
    # Continue running the app, as the model might not be critical for all routes

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_phone_number(text):
    """Extract phone number from text using regex patterns"""
    if not text:
        logger.warning("No text provided for phone number extraction")
        return "Not Found"
        
    patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{4}',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{10}',
        r'\+?\d{1,3}[-.\s]?\d{9,10}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            logger.debug(f"Phone number found: {matches[0]}")
            return matches[0]
    
    logger.debug("No phone number found in text")
    return "Not Found"

def cleanResume(txt):
    """Enhanced text cleaning function with lemmatization and improved regex"""
    if not txt:
        logger.warning("No text provided for cleaning")
        return ""
    
    if isinstance(txt, list):
        txt = " ".join(str(item) for item in txt)
        
    cleanText = txt.lower()
    cleanText = re.sub(r'http\S+', ' ', cleanText)
    cleanText = re.sub(r'www\.\S+', ' ', cleanText)
    cleanText = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[^\w\s]', ' ', cleanText)
    cleanText = re.sub(r'\d+', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    
    words = cleanText.split()
    filtered_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            try:
                lemma = lemmatizer.lemmatize(word)
                filtered_words.append(lemma)
            except Exception as e:
                logger.warning(f"Error lemmatizing word '{word}': {str(e)}")
                filtered_words.append(word)
    
    return ' '.join(filtered_words)

def extract_key_features(resume_data):
    """Extract and weight important sections from resume data"""
    logger.debug("Extracting key features from resume data")
    features = {}
    
    raw_text = resume_data.get("Raw Text", "")
    
    skills = resume_data.get("Skills", [])
    if isinstance(skills, list):
        features['skills'] = " ".join(str(skill) for skill in skills)
    else:
        features['skills'] = str(skills)
    
    for key in ["Education", "Projects", "Certifications"]:
        value = resume_data.get(key, "")
        if isinstance(value, list):
            features[key.lower()] = " ".join(str(item) for item in value)
        else:
            features[key.lower()] = str(value)
    
    experience = resume_data.get("Experience", [])
    if isinstance(experience, list) and experience and experience != ["Not Found"]:
        experience_text = []
        for exp in experience:
            if isinstance(exp, dict):
                exp_text = f"{exp.get('Job Title', '')} {exp.get('Company', '')}"
                responsibilities = exp.get('Key Responsibilities', [])
                if responsibilities:
                    exp_text += " " + " ".join(responsibilities)
                experience_text.append(exp_text)
        features['experience'] = " ".join(experience_text)
    else:
        features['experience'] = ""

    for key in features:
        try:
            features[key] = cleanResume(features[key])
        except Exception as e:
            logger.warning(f"Error cleaning feature '{key}': {str(e)}")
            features[key] = ""
    
    weighted_text = ""
    weighted_text += (features['skills'] + " ") * 3
    weighted_text += (features['experience'] + " ") * 2
    weighted_text += features['education'] + " "
    weighted_text += features['projects'] + " " if 'projects' in features else ""
    weighted_text += features['certifications'] + " " if 'certifications' in features else ""
    
    if len(weighted_text.strip()) < 100:
        try:
            if isinstance(raw_text, list):
                raw_text = " ".join(str(item) for item in raw_text)
            cleaned_raw = cleanResume(raw_text)
            weighted_text += " " + cleaned_raw
        except Exception as e:
            logger.warning(f"Error cleaning raw text: {str(e)}")
    
    logger.debug(f"Feature text extracted, length: {len(weighted_text)} characters")
    return weighted_text.strip()

# Base directory relative to the script's location
BASE_PATH = os.path.join(os.path.dirname(__file__), "model")

# Define model paths
MODEL_PATHS = {
    "categorization": {
        "nb": os.path.join(BASE_PATH, "categorization/naive_bayes_categorization_model.pkl"),
        "logistic": os.path.join(BASE_PATH, "categorization/logistic_categorization_model.pkl"),
        "knn": os.path.join(BASE_PATH, "categorization/knn_categorization.pkl"),
        "label_encoder": os.path.join(BASE_PATH, "categorization/label_encoder_categorization.pkl"),
        "tfidf": os.path.join(BASE_PATH, "tfidf_shared/tfidf_vectorizer_categorization.pkl")
    },
    "recommendation": {
        "nb": os.path.join(BASE_PATH, "recommendation/naive_bayes_recommendation_model.pkl"),
        "logistic": os.path.join(BASE_PATH, "recommendation/logistic_recommendation_model.pkl"),
        "knn": os.path.join(BASE_PATH, "recommendation/knn_recommendation.pkl"),
        "label_encoder": os.path.join(BASE_PATH, "recommendation/label_encoder_job.pkl"),
        "tfidf": os.path.join(BASE_PATH, "tfidf_shared/tfidf_vectorizer_job_recommendation.pkl")
    }
}

def verify_model_files():
    """Verify all model files exist before loading"""
    missing_files = []
    for category, paths in MODEL_PATHS.items():
        for model_type, path in paths.items():
            if not os.path.exists(path):
                missing_files.append(path)
                logger.error(f"Missing model file: {path}")
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} model files: {', '.join(missing_files)}")
        return False
    logger.info("All model files verified successfully")
    return True

def load_model(path):
    """Load a model with improved error handling"""
    try:
        if not os.path.exists(path):
            logger.error(f"Model file does not exist: {path}")
            return None
        model = joblib.load(path)
        logger.info(f"Successfully loaded model from: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {str(e)}")
        return None

# Verify and load models at startup
logger.info("Starting model verification and loading...")
if not verify_model_files():
    logger.warning("Some model files are missing. Application may fail for certain operations.")

models = {
    category: {model: load_model(path) for model, path in paths.items()}
    for category, paths in MODEL_PATHS.items()
}

@app.route("/")
def home():
    logger.debug("Serving home page")
    return render_template("index.html")

@app.route("/results.html")
def results():
    logger.debug("Serving results page")
    return render_template("results.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to verify server is running"""
    logger.debug("Health check requested")
    return jsonify({"status": "ok", "message": "Server is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    try:
        logger.info("Received predict request")
        if "resume" not in request.files:
            logger.error("No resume file uploaded")
            return jsonify({"error": "No resume file uploaded"}), 400

        file = request.files["resume"]
        if not file.filename.lower().endswith((".pdf", ".docx")):
            logger.error(f"Unsupported file format: {file.filename}")
            return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

        # Save file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_path = tmp.name
            file.save(temp_path)
        logger.info(f"Saved resume to temporary file: {temp_path}")

        # Extract resume details
        resume_data = extract_resume_details(temp_path)
        if not resume_data or "error" in resume_data:
            logger.error(f"Failed to extract resume details: {resume_data.get('error', 'Unknown error')}")
            return jsonify({"error": "Failed to extract resume details", "details": resume_data.get("error", "")}), 400

        logger.debug(f"Resume data extracted: {list(resume_data.keys())}")

        # Extract display information
        name = str(resume_data.get("Name", "Not Found"))
        email = str(resume_data.get("Email", "Not Found"))
        
        phone = resume_data.get("Contact Number", None)
        if not phone or phone in ["Not Found", "None"]:
            raw_text = resume_data.get("Raw Text", "")
            if raw_text:
                phone = extract_phone_number(raw_text)
                logger.debug(f"Extracted phone from raw text: {phone}")
        else:
            phone = str(phone)
        
        skills = resume_data.get("Skills", [])
        skills_display = ", ".join(str(skill) for skill in skills) if isinstance(skills, list) else str(skills)
        
        education = resume_data.get("Education", "Not Found")
        if isinstance(education, list):
            education = "\n".join(str(edu) for edu in education)
            
        experience = resume_data.get("Experience", "Not Found")
        if isinstance(experience, list):
            experience_display = []
            for exp in experience:
                if isinstance(exp, dict):
                    exp_text = f"{exp.get('Job Title', 'Not Found')} at {exp.get('Company', 'Not Found')} ({exp.get('Duration', 'Not Found')})"
                    responsibilities = exp.get('Key Responsibilities', [])
                    if responsibilities:
                        exp_text += "\nResponsibilities: " + "; ".join(responsibilities)
                    experience_display.append(exp_text)
            experience = "\n\n".join(experience_display)
        else:
            experience = str(experience)
        
        # Extract features
        try:
            feature_text = extract_key_features(resume_data)
            logger.debug(f"Feature text extracted, length: {len(feature_text)} characters")
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raw_text = resume_data.get("Raw Text", "")
            if isinstance(raw_text, list):
                raw_text = " ".join(str(item) for item in raw_text)
            feature_text = re.sub(r'[^\w\s]', ' ', raw_text.lower())
            feature_text = re.sub(r'\s+', ' ', feature_text).strip()
            logger.info("Using fallback text cleaning")

        if not feature_text:
            logger.error("No meaningful features extracted from resume")
            return jsonify({"error": "Failed to extract meaningful features from resume"}), 400

        # Get model selections
        categorization_model = request.form.get("categorization_model", "nb")
        recommendation_model = request.form.get("recommendation_model", "nb")
        logger.info(f"Selected models - Categorization: {categorization_model}, Recommendation: {recommendation_model}")

        if categorization_model not in models["categorization"] or recommendation_model not in models["recommendation"]:
            logger.error(f"Invalid model selection: Cat={categorization_model}, Rec={recommendation_model}")
            return jsonify({"error": "Invalid model selection"}), 400

        # Load TF-IDF vectorizers
        cat_tfidf = models["categorization"].get("tfidf")
        rec_tfidf = models["recommendation"].get("tfidf")
        if not cat_tfidf or not rec_tfidf:
            missing_files = []
            if not cat_tfidf:
                missing_files.append(MODEL_PATHS["categorization"]["tfidf"])
            if not rec_tfidf:
                missing_files.append(MODEL_PATHS["recommendation"]["tfidf"])
            logger.error(f"Missing TF-IDF vectorizers: {missing_files}")
            return jsonify({"error": f"Missing TF-IDF vectorizers: {', '.join(missing_files)}"}), 500

        # Transform features
        try:
            input_tfidf_cat = cat_tfidf.transform([feature_text])
            input_tfidf_rec = rec_tfidf.transform([feature_text])
            logger.debug("Feature text transformed successfully")
        except Exception as e:
            logger.error(f"Failed to transform features with TF-IDF: {str(e)}")
            return jsonify({"error": "Failed to transform features for prediction"}), 500

        # Make predictions
        cat_model = models["categorization"].get(categorization_model)
        rec_model = models["recommendation"].get(recommendation_model)
        if not cat_model or not rec_model:
            logger.error("Selected models could not be loaded")
            return jsonify({"error": "Selected model could not be loaded"}), 500

        try:
            category_pred = int(cat_model.predict(input_tfidf_cat)[0])
            job_pred = int(rec_model.predict(input_tfidf_rec)[0])
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({"error": "Prediction failed"}), 500
        
        # Get probabilities
        cat_probs = {}
        job_probs = {}
        try:
            if hasattr(cat_model, 'predict_proba'):
                cat_prob_values = cat_model.predict_proba(input_tfidf_cat)[0]
                category_encoder = models["categorization"].get("label_encoder")
                cat_classes = category_encoder.classes_
                cat_probs = {
                    str(category_encoder.inverse_transform([i])[0]): round(float(prob) * 100, 2)
                    for i, prob in enumerate(cat_prob_values) if prob > 0.05
                }
        except Exception as e:
            logger.warning(f"Failed to get category probabilities: {str(e)}")
        
        try:
            if hasattr(rec_model, 'predict_proba'):
                job_prob_values = rec_model.predict_proba(input_tfidf_rec)[0]
                job_encoder = models["recommendation"].get("label_encoder")
                job_classes = job_encoder.classes_
                job_probs = {
                    str(job_encoder.inverse_transform([i])[0]): round(float(prob) * 100, 2)
                    for i, prob in enumerate(job_prob_values) if prob > 0.05
                }
        except Exception as e:
            logger.warning(f"Failed to get job probabilities: {str(e)}")
        
        logger.debug(f"Predictions - Category: {category_pred}, Job: {job_pred}")

        # Decode predictions
        category_encoder = models["categorization"].get("label_encoder")
        job_encoder = models["recommendation"].get("label_encoder")
        if not category_encoder or not job_encoder:
            logger.error("Label encoders not loaded")
            return jsonify({"error": "Label encoders not loaded"}), 500

        try:
            category_label = str(category_encoder.inverse_transform([category_pred])[0])
            job_label = str(job_encoder.inverse_transform([job_pred])[0])
        except Exception as e:
            logger.error(f"Failed to decode predictions: {str(e)}")
            return jsonify({"error": "Failed to decode predictions"}), 500
        
        logger.info(f"Decoded predictions - Category: {category_label}, Job: {job_label}")

        response = {
            "Name": name,
            "Email": email,
            "Contact Number": phone,
            "Skills": skills_display,
            "Education": education,
            "Experience": experience,
            "Categorization": category_label,
            "Job Recommendation": job_label
        }
        
        if cat_probs:
            response["Category Probabilities"] = cat_probs
        if job_probs:
            response["Job Recommendation Probabilities"] = job_probs

        logger.info("Prediction completed successfully")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "details": traceback.format_exc()}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_path}: {str(e)}")

@app.route("/ats_check", methods=["POST"])
def ats_check():
    temp_path = None
    try:
        logger.info("Received ATS check request")
        if "resume" not in request.files:
            logger.error("No resume file uploaded")
            return jsonify({"error": "No resume file uploaded"}), 400

        file = request.files["resume"]
        if not file.filename.lower().endswith((".pdf", ".docx")):
            logger.error(f"Unsupported file format: {file.filename}")
            return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_path = tmp.name
            file.save(temp_path)
        logger.info(f"Saved resume to temporary file: {temp_path}")

        format_check = check_file_format(temp_path)
        
        if temp_path.endswith(".pdf"):
            resume_text = extract_text_from_pdf(temp_path)
        elif temp_path.endswith(".docx"):
            resume_text = extract_text_from_docx(temp_path)
        else:
            resume_text = ""
        
        if not resume_text:
            logger.error("Failed to extract text from resume")
            return jsonify({"error": "Failed to extract text from resume"}), 400

        job_description = request.form.get("job_description", "")
        industry = request.form.get("industry", "")
        logger.debug(f"Job description length: {len(job_description)}, Industry: {industry}")

        ats_results = check_ats_compatibility(resume_text, job_description)
        
        if industry:
            benchmark = benchmark_against_industry(resume_text, industry)
            if 'details' not in ats_results:
                ats_results['details'] = {}
            ats_results['details']['industry_benchmark'] = benchmark
            if 'missing_keywords' in benchmark and benchmark.get('industry_relevance', 0) < 0.5:
                missing_keywords = benchmark.get('missing_keywords', [])
                if missing_keywords:
                    ats_results['recommendations'].append(
                        f"Add more {benchmark['industry']}-specific keywords: {', '.join(missing_keywords[:3])}"
                    )

        response = {
            "analysis": {
                "score": ats_results.get('score', 0),
                "warnings": ats_results.get('warnings', []),
                "recommendations": ats_results.get('recommendations', []),
                "details": ats_results.get('details', {})
            },
            "report": "\n".join([
                "=== 📊 ATS Compatibility Report ===",
                f"🟢 Overall Score: {ats_results.get('score', 0)}/100",
                "\n⚠️ Warnings:",
                "\n".join(f"- {w}" for w in ats_results.get('warnings', [])),
                "\n💡 Recommendations:",
                "\n".join(f"- {r}" for r in ats_results.get('recommendations', []))
            ]),
            "format_check": {
                "format": format_check.get("format", "unknown"),
                "size_mb": format_check.get("size_mb", 0),
                "issues": format_check.get("issues", [])
            }
        }
        
        logger.info("ATS check completed successfully")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in ATS check: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred during ATS check: {str(e)}", "details": traceback.format_exc()}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_path}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Flask application")
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(debug=False, host='0.0.0.0', port=port)
