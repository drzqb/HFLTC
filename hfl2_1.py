from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a huggingface course my whole life.",
    "She hates this so much!",
    "He loves these books."
]

inputs = tokenizer(raw_inputs,
                   padding=True,
                   truncation=True,
                   return_tensors="pt",
                   # return_token_type_ids=True,
                   )

print(inputs)

from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

print(model)

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits)

probilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(probilities)

res = probilities.argmax(dim=-1)
print(res)

predictions = [model.config.id2label[k.item()] for k in res]
print(predictions)

model.config.id2label = {1: "HAPPY", 0: "WORRY"}
predictions = [model.config.id2label[k.item()] for k in res]
print(predictions)
