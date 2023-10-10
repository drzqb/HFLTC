from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm

device = "cuda"

mycheckpoint = "thucnewsmodel7"
tokenizer = AutoTokenizer.from_pretrained(mycheckpoint)


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


def predict(sentence):
    data = tokenizer(sentence, return_tensors="pt")

    print(data)
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

    model = AutoModelForSequenceClassification.from_pretrained(mycheckpoint)
    # print(model)

    output = model(**data)
    predict = output['logits'][0].argmax().item()
    print(id2label[predict])


def inference(sentence):
    classifier = pipeline("text-classification",
                          model=mycheckpoint,
                          tokenizer=mycheckpoint)

    res = classifier(sentence)[0]
    print(res['label'])


if __name__ == "__main__":
    # eval()

    inference("姚明是哪个篮球队的？")
