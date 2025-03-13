
#%%
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

#%%
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available
print(f"Using device: {device}")
#%%
# Load the dataset
dataset = load_dataset("go_emotions")

# Define a function to sample the dataset
def sample_dataset(dataset, sample_size):
    dataset = dataset.shuffle(seed=42)  # Ensure randomness
    sampled_data = dataset.select(range(min(sample_size, len(dataset))))  
    return sampled_data

# Define sample sizes for train, validation, and test
train_size, val_size, test_size = 6000, 2000, 2000  

# Subsample each split
small_train = sample_dataset(dataset["train"], train_size)
small_val = sample_dataset(dataset["validation"], val_size)
small_test = sample_dataset(dataset["test"], test_size)

# Create a new dataset dictionary with the reduced dataset
small_dataset = {
    "train": small_train,
    "validation": small_val,
    "test": small_test
}

# Tokenizer initialization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#%%
# Preprocessing function for tokenization and label handling
def preprocess_function(examples):
    
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True)

    # One-hot encode the labels if they are multi-label
    labels = [torch.tensor(label).sum(dim=0).tolist() if isinstance(label, list) else 0 for label in examples["labels"]]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Apply the preprocessing function to all splits
tokenized_datasets = {split: data.map(preprocess_function, batched=True) for split, data in small_dataset.items()}

# Remove the 'text' column as it's no longer needed
for split in tokenized_datasets:
    tokenized_datasets[split] = tokenized_datasets[split].remove_columns(["text"])

# Set the format for PyTorch
for split in tokenized_datasets:
    tokenized_datasets[split].set_format("torch")

# Get the number of labels from the dataset
num_labels = len(dataset["train"].features["labels"].feature.names)

# Split dataset into train, validation, and test
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


#%%

# Training 
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,  # Increase the number of epochs
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,  # Ensures the best model is saved
    metric_for_best_model="accuracy",  # Select metric to monitor for saving the best model
    warmup_steps=500,  # Learning rate warmup
    logging_steps=10,  # Log training process
    lr_scheduler_type="linear",  # Linear learning rate scheduler
    logging_first_step=True,  # Log the first step
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# %%
# Evaluate model
def evaluate_model():
    predictions = trainer.predict(test_dataset)
    logits, labels = predictions.predictions, predictions.label_ids
    predicted_classes = np.argmax(logits, axis=-1)
    print("Classification Report:\n", classification_report(labels, predicted_classes))

evaluate_model()
#%%

# Move model to MPS
model.to(device)

#%%
# Save model
def save_model():
    model.save_pretrained("emotion_model")
    tokenizer.save_pretrained("emotion_model")
    print("Model saved successfully.")

save_model()



# type: ignore #%%

# %%
