import warnings

warnings.filterwarnings("ignore")

from transformers import pipeline

classifier = pipeline(task="sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english",
                      )
print(classifier)

out = classifier([
    "I've been waiting for a huggingface course my whole life.",
    "She hates this so much!",
])

print(out)
