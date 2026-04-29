import joblib
import json

tokenizer = joblib.load("models/tokenizer.pkl")

tokenizer_json = tokenizer.to_json()

with open("models/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

print("Tokenizer converted successfully")