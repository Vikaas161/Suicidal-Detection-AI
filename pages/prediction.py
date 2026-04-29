import streamlit as st
import torch
import joblib
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
from keras.models import load_model
import numpy as np

st.set_page_config(page_title="Suicide AI", layout="wide")

@st.cache_resource
def load_model():
    lr_model = None
    nb_model = None
    svm_model = None
    tfidf = None

    from keras.models import load_model
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.mixed_precision import Policy

    custom_objects = {
        "InputLayer": lambda **kwargs: InputLayer(
            input_shape=kwargs.get("batch_shape", [None, 100])[1:]
        ),
        "DTypePolicy": Policy,
        "Policy": Policy
    }

    lstm_model = load_model(
        "models/lstm_model.h5",
        compile=False,
        custom_objects=custom_objects
    )

    bilstm_model = load_model(
        "models/bilstm_model.h5",
        compile=False,
        custom_objects=custom_objects
    )

    cnn_model = load_model(
        "models/cnn_model.h5",
        compile=False,
        custom_objects=custom_objects
    )
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    import json

    with open("models/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = f.read()

    tokenizer = tokenizer_from_json(tokenizer_json)
    bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
    bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")
    explainer = LimeTextExplainer(class_names=["Non-Suicidal", "Suicidal"])

    llm = pipeline("text-generation", model="google/flan-t5-large")

    return lr_model, nb_model, svm_model, tfidf, lstm_model, bilstm_model, cnn_model, tokenizer, bert_model, bert_tokenizer, explainer, llm

lr_model, nb_model, svm_model, tfidf, lstm_model, bilstm_model, cnn_model, tokenizer, bert_model, bert_tokenizer, explainer, llm = load_model()

def prediction(text, model_name):
    if model_name == "Logistic Regression":
        if lr_model is None or tfidf is None:
            return "Model unavailable"

    elif model_name == "Naive Bayes":
        if nb_model is None or tfidf is None:
            return "Model unavailable"

    elif model_name == "SVM":
        if svm_model is None or tfidf is None:
            return "Model unavailable"
    elif model_name == "LSTM":
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = (lstm_model.predict(padded) > 0.5).astype(int)[0][0]
    elif model_name == "BILSTM":
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = (bilstm_model.predict(padded) > 0.5).astype(int)[0][0]
    elif model_name == "CNN":
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = (cnn_model.predict(padded) > 0.5).astype(int)[0][0]
    elif model_name == "BERT":
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return "Suicidal" if pred == 1 else "Non-Suicidal"

def bert_predict_proba(texts):
    inputs = bert_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

def explain_text(text):
    exp = explainer.explain_instance(
        text,
        bert_predict_proba,
        num_features=6
    )
    return exp


st.title("Suicidal Detection Prediction")

model_choice = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Naive Bayes",
        "SVM",
        "LSTM",
        "BILSTM",
        "CNN",
        "BERT"
    ]
)

text = st.text_area("Enter Text")
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter Text")
    else:
        result = prediction(text, model_choice)
        if result == "Non-Suicidal":
            st.success(f"{model_choice} Prediction: {result}")
        else:
            st.error(f"{model_choice} Prediction: {result}")

        response = llm(
            f"""
            Instruction: Explain why the following text indicates suicidal ideation.

            Text: {text}

            Explanation:
            """,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )

        st.write(response[0]['generated_text'])
        if model_choice == "BERT":
            exp = explain_text(text)
            st.subheader("Visual Explanation")
            st.components.v1.html(exp.as_html(), height=400)