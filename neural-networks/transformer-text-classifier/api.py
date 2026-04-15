from fastapi import FastAPI
from inference import predict

app = FastAPI()

@app.get("/predict")
def classify(text: str):
    result = predict(text)
    return {"sentiment": result}