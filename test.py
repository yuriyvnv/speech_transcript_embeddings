from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.keys())
print("TUDO", outputs)