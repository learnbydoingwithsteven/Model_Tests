import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import time

def print_model_size(model, label=""):
    """Prints the size of a PyTorch model in MB."""
    torch.save(model.state_dict(), "temp_model.p")
    size_mb = os.path.getsize("temp_model.p") / 1e6
    print(f"Size of {label} model: {size_mb:.2f} MB")
    os.remove("temp_model.p")

# 1. Load a pre-trained model
model_name = "distilbert-base-uncased"
print(f"Loading pre-trained model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Print original model size
print_model_size(model, "original")

# 2. Apply dynamic quantization
# The model is quantized on-the-fly for inference
print("\nApplying dynamic quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Print quantized model size
print_model_size(quantized_model, "quantized")

# 3. Compare inference speed
print("\nComparing inference time...")
dummy_text = "This is a test sentence for inference."
inputs = tokenizer(dummy_text, return_tensors="pt")

# Original model inference
start_time = time.time()
with torch.no_grad():
    original_output = model(**inputs)
end_time = time.time()
print(f"Original model inference time: {end_time - start_time:.4f} seconds")

# Quantized model inference
start_time = time.time()
with torch.no_grad():
    quantized_output = quantized_model(**inputs)
end_time = time.time()
print(f"Quantized model inference time: {end_time - start_time:.4f} seconds")

print("\nQuantization test complete. The quantized model is smaller and can be faster on CPU.")
