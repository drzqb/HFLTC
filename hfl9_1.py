import warnings
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="micro"),
        "recall": recall_score(labels, predictions, average="micro"),
        "f1": f1_score(labels, predictions, average="micro"),
    }


def predict(checkpoint, test_data_path):
    def tokenize_function(example):
        return tokenizer(example["sentence"])

    raw_datasets = load_dataset("csv", data_files={
        "test": test_data_path,
    })

    raw_datasets = raw_datasets.shuffle()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=14,
    )

    trainer = Trainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_datasets["test"])
    print(predictions.metrics)


if __name__ == "__main__":
    predict("models/thucnewsmodel9/run-2/checkpoint-1868",
            "data/THUCNews/thucnews_test.csv")
