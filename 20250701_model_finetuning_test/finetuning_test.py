import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset

# 1. Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
print(f"Loading pre-trained model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Create a dummy dataset for fine-tuning
# In a real scenario, you would load your own dataset here.
print("\nCreating a dummy dataset...")
sentences = [
    "This is a positive sentence.",
    "I love using transformers!",
    "This is a negative sentence.",
    "I hate this product."
]
labels = [1, 1, 0, 0]  # 1 for positive, 0 for negative

# Tokenize the dataset
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.tensor(labels)

# Create a DataLoader
dataset = TensorDataset(input_ids, attention_mask, labels)
data_loader = DataLoader(dataset, batch_size=2)

# 3. Set up the fine-tuning components
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Move model to CPU (or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Fine-tuning on: {device}")

# 4. Fine-tuning loop
print("\nStarting fine-tuning loop...")
model.train()  # Set the model to training mode
for epoch in range(2):  # Train for 2 epochs as a demonstration
    for batch in data_loader:
        # Unpack batch and move to the correct device
        b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    print(f"Epoch {epoch + 1} complete. Loss: {loss.item():.4f}")

print("\nFine-tuning complete.")

# 5. Test the fine-tuned model
model.eval() # Set the model to evaluation mode
test_sentence = "I enjoy learning new things."
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1).item()

print(f'\nTest sentence: "{test_sentence}"')
print(f'Predicted label: {"Positive" if prediction == 1 else "Negative"}')
