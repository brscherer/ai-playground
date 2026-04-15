from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model/saved")
model = AutoModelForSequenceClassification.from_pretrained("model/saved")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    predicted_class = torch.argmax(probs).item()

    return "positive" if predicted_class == 1 else "negative"

print(predict("This movie was amazing!"))