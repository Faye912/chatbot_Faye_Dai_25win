
#%%
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

#%%
# Load dataset
dataset = load_dataset("go_emotions")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

num_labels = len(dataset["train"].features["labels"].feature.names)


#%%
# define training function
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True)
    
    tokenized_inputs["labels"] = [label[0] if isinstance(label, list) and len(label) > 0 
                                  else 0 for label in examples["labels"]]
    return tokenized_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])  
tokenized_datasets.set_format("torch")  

# Split dataset into train and validation
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
num_labels=num_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

#%%
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
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

# Save model
def save_model():
    model.save_pretrained("emotion_model")
    tokenizer.save_pretrained("emotion_model")
    print("Model saved successfully.")

save_model()

# Load model
def load_model():
    loaded_model = BertForSequenceClassification.from_pretrained("emotion_model")
    loaded_tokenizer = BertTokenizer.from_pretrained("emotion_model")
    return loaded_model, loaded_tokenizer

#%%
# Example inference function
def predict_emotion(text):
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    print(f"Predicted emotion category: {predicted_class}")

# Example usage
predict_emotion("I am feeling so happy today!")

