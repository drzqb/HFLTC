import warnings

warnings.filterwarnings("ignore")

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc", cache_dir="./")

# print(raw_datasets)

raw_datasets_train = raw_datasets["train"]
# print(raw_datasets_train.features)

# print(raw_datasets["train"][100])

from transformers import AutoTokenizer

checkpoint = "./bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer("this is the first sentence.", "this is the second sentence.")


# print(inputs)

# print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# print(tokenized_datasets)

# print(tokenized_datasets["train"][:2])

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
# print(samples)

batch = data_collator(samples)
batch_shape = {k: v.shape for k, v in batch.items()}
# print(batch_shape)

from transformers import TrainingArguments

# import evaluate
# print(evaluate.list_evaluation_modules("metric"))
# accuracy_metric = evaluate.load("accuracy")

from datasets import load_metric

accuracy_metric = load_metric("glue", "mrpc")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    # acc = np.mean(np.equal(predictions, labels))
    # return {"accuracy": acc}
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    "test-trainer",
    save_strategy="epoch",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# print(model)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# trainer.save_model("test-trainer")


predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.metrics["test_accuracy"], predictions.metrics["test_f1"])
