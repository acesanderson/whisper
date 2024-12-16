"""
Following instructions from the official HuggingFace repo:
    https://huggingface.co/openai/whisper-large-v3-turbo
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

MP3_FILE = "dr_visit_12-16-2024.mp3"

if torch.cuda.is_available():
    print("CUDA is available.")
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    print("CUDA is not available.")
    device = "mps"
    torch_dtype = torch.float32


model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# result = pipe(sample, return_timestamps=True)
# print(result["text"])

# Put filename here.
result = pipe(MP3_FILE, return_timestamps=True)
print(result["text"])
