import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class RobertaSentinelModel:
    def __init__(self, model_checkpoint_path: str):
        # Load the tokenizer for RoBERTa
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load the model from the checkpoint
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        
        # Put the model in evaluation mode
        self.model.eval()

    def predict(self, text: str):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class (assuming binary classification for simplicity)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        return predicted_class

if __name__ == '__main__':
    print("Loading the RobertaSentinelModel")
    model = RobertaSentinelModel("roberta_mlp.pt")
    
    # Example text to classify
    text = "This is a test sentence."
    
    # Get the prediction
    prediction = model.predict(text)
    print(f"Predicted class: {prediction}")