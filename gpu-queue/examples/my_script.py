import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

text = [
    "Brevity is the soul of wit.",
    "Amor, ch'a nullo amato amar perdona."
]

model_ckpt = "/nfs/data/project/models/papluca--xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

preds = torch.softmax(logits, dim=1)

# Map raw predictions to languages
id2lang = model.config.id2label
vals, idxs = torch.max(preds, dim=1)
{id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)}

print("test DONE")