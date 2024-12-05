from flask import Flask, render_template, request
import re
import string
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk

# Configure NLTK data directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Ensure necessary NLTK data is available
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

app = Flask(__name__)

# Helper function to preprocess text
def preprocess_text(input_text):
    """Clean and preprocess the input text."""
    cleaned_text = str(input_text).lower()
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)  # Remove HTML tags
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\w*\d\w*', '', cleaned_text)  # Remove words with numbers
    table = str.maketrans('', '', string.punctuation)
    cleaned_text = cleaned_text.translate(table)  # Remove punctuation
    return cleaned_text

def predict_with_distilbert(sentence):
    """Make predictions using the DistilBERT model."""
    try:
        # Preprocess the input sentence
        sentence = preprocess_text(sentence)
        inputs = tokenizer_bert(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model_bert(**inputs)
            logits = outputs.logits
        # Predict class and confidence
        predicted_class = torch.argmax(logits, dim=1).item()
        probability = torch.softmax(logits, dim=1)[0][predicted_class].item()
        return "REAL" if predicted_class == 1 else "FAKE", probability
    except Exception as e:
        print(f"Error in DistilBERT prediction: {e}")
        return "Error", 0.0

# Load DistilBERT model and tokenizer
try:
    model_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model_bert.load_state_dict(torch.load('distilbert_finetuned_model.pth', map_location=torch.device('cpu')))
    tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
except Exception as e:
    print(f"Error loading DistilBERT model or tokenizer: {e}")
    model_bert = None
    tokenizer_bert = None

@app.route('/', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    prediction_label = None
    probability = None
    text = ""  # Initialize text to avoid reference errors

    if request.method == 'POST':
        text = request.form.get('text', '')  # Safely get 'text' input
        try:
            if model_bert:
                prediction_label, probability = predict_with_distilbert(text)
            else:
                prediction_label = "Error"
                probability = 0.0
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction_label = "Error"
            probability = 0.0

    return render_template(
        'index.html',
        prediction_text=text,
        prediction_label=prediction_label,
        prediction_probability=f"{probability:.2f}" if probability else "N/A"
    )

if __name__ == '__main__':
    app.run(debug=True, port=8199)
