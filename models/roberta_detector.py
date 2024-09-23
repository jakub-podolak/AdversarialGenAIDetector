from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "roberta-large"  # RoBERTa large, as in the RADAR paper
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Function to predict whether text is AI-generated or human-generated
def classify_text(text):
    # Not sure what the best value for max_length should be
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Label mapping - 0: Human-generated, 1: AI-generated)
    labels = ['Human-generated', 'AI-generated']
    return labels[predicted_class]

# Here is an example how to use it for prediction
text = "The AI model was able to detect human-like text with high accuracy."
prediction = classify_text(text)
print(f"Prediction: {prediction}")

# ToDo: Code for loading the dataset and fine-tuning
