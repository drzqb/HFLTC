import warnings

warnings.filterwarnings("ignore")

from datasets import load_dataset

raw_datasets = load_dataset("csv", data_files={
    "train": "THUCNews/thucnews_train.csv",
    "validation": "THUCNews/thucnews_val.csv",
    "test": "THUCNews/thucnews_test.csv",
})
print(raw_datasets)

raw_datasets = raw_datasets.shuffle()

from transformers import AutoTokenizer

checkpoint = "./bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# inputs = tokenizer("体育比赛精彩纷呈！")
# print(inputs)

# print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))


def tokenize_function(example):
    return tokenizer(example["sentence"])


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# print(tokenized_datasets)

print(tokenized_datasets["train"][:30])


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:30]
print(samples)
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}
print(samples)

# batch = data_collator(samples)
# batch_shape = {k: v.shape for k, v in batch.items()}
# print(batch_shape)

from transformers import TrainingArguments

# from datasets import load_metric
# import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_metrics(eval_pred):
    # accuracy_metric = load_metric("accuracy")
    # precision_metric = load_metric("precision")
    # recall_metric = load_metric("recall")
    # f1_metric = load_metric("f1")
    # accuracy_metric = evaluate.load("accuracy")
    # precision_metric = evaluate.load("precision")
    # recall_metric = evaluate.load("recall")
    # f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="micro"),
        "recall": recall_score(labels, predictions, average="micro"),
        "f1": f1_score(labels, predictions, average="micro"),
    }
    # return {
    #     "accuracy": accuracy_metric.compute(
    #         predictions=predictions,
    #         references=labels
    #     )["accuracy"],
    # "precision": precision_metric.compute(
    #     predictions=predictions,
    #     references=labels,
    #     average="micro"
    # )["precision"],
    # "recall": recall_metric.compute(
    #     predictions=predictions,
    #     references=labels,
    #     average="micro"
    # )["recall"],
    # "f1": f1_metric.compute(
    #     predictions=predictions,
    #     references=labels,
    #     average="micro"
    # )["f1"]
    # }


training_args = TrainingArguments(
    "thucnewsmodel",
    save_strategy="no",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    # num_train_epochs=5,
)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=14)
print(model)

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

trainer.save_model("thucnewsmodel")

predictions = trainer.predict(tokenized_datasets["test"])
print(
    predictions.metrics["test_accuracy"],
    predictions.metrics["test_precision"],
    predictions.metrics["test_recall"],
    predictions.metrics["test_f1"]
)
