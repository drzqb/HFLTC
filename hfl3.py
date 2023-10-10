checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

print(model)

seq1_ids = [[200, 200, 200]]
seq2_ids = [[200, 200]]

batched_ids = [[200, 200, 200],
               [200, 200, tokenizer.pad_token_id]]

with torch.no_grad():
    outputs1 = model(torch.tensor(seq1_ids))
    outputs2 = model(torch.tensor(seq2_ids))
    outputs_batched = model(torch.tensor(batched_ids))
    outputs_batched_fix = model(torch.tensor(batched_ids), attention_mask=torch.tensor([
        [1, 1, 1],
        [1, 1, 0]
    ]))

print(outputs1.logits)
print(outputs2.logits)
print(outputs_batched.logits)
print(outputs_batched_fix.logits)
