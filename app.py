"""
app.py — Interactive Gradio demo for the fine-tuned emotion classifier.

Run locally after training:
    python app.py

Deploy on Hugging Face Spaces:
    1. Create a new Space (SDK: Gradio) and push this repo's files to it.
    2. Push your trained model to the Hub: `python train.py --push_to_hub --hub_model_id you/emotion-classifier`
    3. Set MODEL_SOURCE below to "you/emotion-classifier" so the Space
       doesn't need the (large) model folder committed to git.
"""
import os

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local path (default, works right after running train.py) OR a Hub repo id
# such as "your-username/emotion-classifier" once you've pushed the model.
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "./emotion-classifier-model")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJI = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}

if MODEL_SOURCE.startswith("./") and not os.path.isdir(MODEL_SOURCE):
    raise FileNotFoundError(
        f"No model found at '{MODEL_SOURCE}'.\n"
        f"Run `python train.py` first to produce it, or set MODEL_SOURCE to a "
        f"Hugging Face Hub repo id (e.g. 'your-username/emotion-classifier') "
        f"after pushing your trained model with `python train.py --push_to_hub ...`."
    )

print(f"Loading model from: {MODEL_SOURCE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
model.eval()

# Prefer the label order baked into the model config (set by train.py's
# id2label), falling back to LABEL_NAMES if it's ever missing.
id2label = model.config.id2label if model.config.id2label else dict(enumerate(LABEL_NAMES))


def predict(text: str):
    if not text or not text.strip():
        return {name: 0.0 for name in LABEL_NAMES}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {f"{EMOJI.get(id2label[i], '')} {id2label[i]}": float(probs[i]) for i in range(len(probs))}


examples = [
    "I can't believe I actually got the internship offer today!",
    "I miss my old friends so much, nothing feels the same anymore.",
    "Why would you say that to me, I am so done with this.",
    "My hands are shaking, I don't know what's going to happen next.",
    "I never expected to see you here, what a small world!",
    "Being with you makes every ordinary day feel special.",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter a sentence",
        placeholder="Type how you're feeling, or paste any short piece of text...",
        lines=3,
    ),
    outputs=gr.Label(num_top_classes=6, label="Predicted emotion"),
    examples=examples,
    title="🎭 Emotion Classifier — Fine-Tuned DistilBERT",
    description=(
        "A DistilBERT model fine-tuned on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) "
        "dataset to classify text into 6 emotions: sadness, joy, love, anger, fear, and surprise. "
        "Type a sentence below and see the model's confidence for each class."
    ),
    article=(
        "Built as a portfolio project — fine-tuning code, evaluation, and this demo are all in the "
        "[GitHub repo](https://github.com/) this Space was built from."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
