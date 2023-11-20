'''
    自动调参实验
'''
import warnings

warnings.filterwarnings("ignore")

from datasets import load_dataset

raw_datasets = load_dataset("csv", data_files={
    "train": "data/THUCNews/thucnews_train.csv",
    "validation": "data/THUCNews/thucnews_val.csv",
    "test": "data/THUCNews/thucnews_test.csv",
})

raw_datasets = raw_datasets.shuffle()

from transformers import AutoTokenizer

checkpoint = "./bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence"])


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="micro"),
        "recall": recall_score(labels, predictions, average="micro"),
        "f1": f1_score(labels, predictions, average="micro"),
    }


training_args = TrainingArguments(
    "models/thucnewsmodel9",
    save_strategy="epoch",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

from transformers import AutoModelForSequenceClassification


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=14)
    return model


from transformers import Trainer

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


def default_hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64, 128]),
        "optim": trial.suggest_categorical("optim", ["sgd", "adamw_hf","adamw_torch"]),
    }


trainer.hyperparameter_search(
    hp_space=default_hp_space_optuna,
    compute_objective=lambda x: x["eval_f1"],
    direction="maximize",
    n_trials=10,
)

