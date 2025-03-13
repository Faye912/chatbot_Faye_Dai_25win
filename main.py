from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("emotion_model")
tokenizer = BertTokenizer.from_pretrained("emotion_model")
model.eval()

class TextInput(BaseModel):
    text: str
    
EMOTION_LABELS = ["admiration", "amusement", "anger", "annoyance", "approval","caring", 
                  "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", 
                  "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", 
                  "nervousness", "optimism", "pride", "realization", "relief", "remorse", 
                  "sadness", "surprise", "neutral"]




@app.post("/predict/")
async def predict_emotion(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=-1).item()
    return {"text": input.text, "predicted_emotion": EMOTION_LABELS[predicted_class]}

@app.get("/")
def root():
    return {"message": "Emotion classifier API is running!"}