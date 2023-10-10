import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

print(model)

raw_inputs = [
    "I've been waiting for a huggingface course my whole life.",
    "She hates this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)

predicted_class_id = logits.argmax(axis=-1)
print(predicted_class_id)

print([model.config.id2label[predicted_class_id[k].item()] for k in range(len(raw_inputs))])
