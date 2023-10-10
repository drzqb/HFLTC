from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a huggingface course my whole life.",
    "She hates this so much!",
]

inputs = tokenizer(raw_inputs,
                   padding=True,
                   truncation=True,
                   return_tensors="pt",
                   # return_token_type_ids=True,
                   )

print(inputs)

rev = tokenizer.decode([101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172,
                        2607, 2026, 2878, 2166, 1012, 102])
print(rev)

from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)

print(model)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
