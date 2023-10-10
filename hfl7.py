from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
import os

device = "cuda"
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

mycheckpoint = "thucnewsmodel7"
if not os.path.exists(mycheckpoint):
    os.makedirs(mycheckpoint)

label2id = {
    '体育': 0,
    '娱乐': 1,
    '家居': 2,
    '彩票': 3,
    '房产': 4,
    '教育': 5,
    '时尚': 6,
    '时政': 7,
    '星座': 8,
    '游戏': 9,
    '社会': 10,
    '科技': 11,
    '股票': 12,
    '财经': 13,
}

id2label = {v: k for k, v in label2id.items()}


class MyDataset(Dataset):
    '''
    从csv文件读取文本数据
    '''

    def __init__(self, csvfile):
        self.csvdata = pd.read_csv(csvfile)

    def __len__(self):
        return len(self.csvdata)

    def __getitem__(self, idx):
        data = self.csvdata.values[idx]
        return data


def collate_fn(data):
    '''
    整理batch数据并编码
    :param data: 输入batch文本数据
    :return: 返回编码数据
    '''
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = tokenizer(sents, padding=True, return_tensors="pt")

    labels = torch.LongTensor(labels)
    data["labels"] = labels

    data.to(device)

    return data


def train():
    dataset_train = MyDataset("./THUCNews/thucnews_train.csv")
    dataset_val = MyDataset("./THUCNews/thucnews_val.csv")

    loadertrain = DataLoader(
        dataset=dataset_train,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=False
    )
    loaderval = DataLoader(
        dataset=dataset_val,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=14,
                                                               id2label=id2label,
                                                               label2id=label2id,
                                                               )

    model.save_pretrained(mycheckpoint)
    tokenizer.save_pretrained(mycheckpoint)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(loadertrain)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=0,
    )

    for epoch in range(num_epochs):
        total_loss = 0.
        model.train()

        for batch in tqdm(loadertrain):
            model.zero_grad()

            outputs = model(**batch)

            loss = outputs["loss"]

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

        model.eval()

        correct = 0
        total = 0
        for batch in tqdm(loaderval):
            with torch.no_grad():
                outputs = model(**batch)
            out = outputs["logits"].argmax(dim=1)
            correct += (out == batch["labels"]).sum().item()
            total += len(batch["labels"])

        avg_train_loss = total_loss / len(loadertrain)
        avg_eval_acc = correct / total
        print("\nepoch: ", epoch + 1,
              "     loss: ", avg_train_loss,
              "   eval_acc: ", avg_eval_acc)

    model.save_pretrained(mycheckpoint)


def eval():
    dataset_test = MyDataset("./THUCNews/thucnews_test.csv")

    loadertest = DataLoader(
        dataset=dataset_test,
        batch_size=16,
        collate_fn=collate_fn,
        drop_last=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(mycheckpoint)

    model.to(device)

    model.eval()

    correct = 0
    total = 0
    for batch in tqdm(loadertest):
        with torch.no_grad():
            outputs = model(**batch)
        out = outputs["logits"].argmax(dim=1)
        correct += (out == batch["labels"]).sum().item()
        total += len(batch["labels"])

    avg_eval_acc = correct / total
    print("test_acc: ", avg_eval_acc)


if __name__ == "__main__":
    # train()

    eval()
