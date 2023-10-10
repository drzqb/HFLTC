import warnings

warnings.filterwarnings("ignore")

from datasets import load_dataset

raw_datasets = load_dataset("csv", data_files={
    "test": "THUCNews/thucnews_test.csv",
})

raw_datasets = raw_datasets.shuffle()

from transformers import AutoTokenizer

checkpoint = "./thucnewsmodel"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence"])


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from datasets import load_metric

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_metric.compute(
            predictions=predictions,
            references=labels
        )["accuracy"],
        "f1": f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="micro"
        )["f1"]
    }


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=14)
# print(model)

from transformers import Trainer

trainer = Trainer(
    model,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

predictions = trainer.predict(tokenized_datasets["test"])
print(predictions.metrics["test_accuracy"], predictions.metrics["test_f1"])

print(predictions.predictions.argmax(axis=-1))
print(predictions.label_ids)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
actual=predictions.label_ids
pred=predictions.predictions.argmax(axis=-1)
print({
    "accuracy": accuracy_score(actual, pred),
    "precision": precision_score(actual, pred, average="micro"),
    "recall": recall_score(actual, pred, average="micro"),
    "f1": f1_score(actual, pred, average="micro"),
})

