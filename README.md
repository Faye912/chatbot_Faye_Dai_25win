# Emotion Detection with BERT

This project involves building a **text-based emotion detection model** using the **GoEmotions dataset** and deploying it as an API using **FastAPI**. The model predicts emotions in text and can be accessed through a web-based API for real-time emotion analysis.

## Features

- **Emotion Classification**: Classifies text into one of the 27 emotions (e.g., joy, sadness, anger, etc.) using a pre-trained BERT model fine-tuned on the GoEmotions dataset.
- **API Integration**: The model is exposed through a REST API using **FastAPI**.
- **Real-Time Emotion Prediction**: Send a POST request with text input to get the predicted emotion.

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

- Python 3.7+
- `torch` (with MPS support for Apple Silicon users, or CPU)
- `transformers`
- `fastapi`
- `uvicorn`
- `pydantic`
- `scikit-learn`

You can install the dependencies via:

```bash
pip install -r requirements.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install required packages

Run the following command to install all dependencies:

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

If you haven't trained the model yet, run the following code to fine-tune BERT on the GoEmotions dataset.

```python
# train_model.py
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset and preprocess
dataset = load_dataset("go_emotions")

# Set up the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=27)

# Fine-tune the model (use Trainer API)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

# Save the model
model.save_pretrained("emotion_model")
tokenizer.save_pretrained("emotion_model")
```

Run the training script:
```bash
python train_model.py
```

### 4. Deploy the Model with FastAPI

Start the FastAPI server to deploy the emotion detection model:

```bash
uvicorn app:app --reload
```

This will run the API on **http://127.0.0.1:8000**. The API will have the following endpoints:

- **POST /predict**: Classify the emotion of a given text input.

Example request:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I am feeling so happy today!"
}'
```

Response:
```json
{
  "emotion": "joy"
}
```

## API Endpoints

### 1. `/predict`

- **Method**: POST
- **Request Body**: JSON object with a text field containing the input text to be classified.
  
Example:
```json
{
  "text": "I am feeling great!"
}
```

- **Response**: JSON object with the predicted emotion.

Example:
```json
{
  "emotion": "joy"
}
```

### 2. `/docs`

The **Swagger UI** documentation for the API is available at **http://127.0.0.1:8000/docs** for testing the endpoints interactively.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **GoEmotions Dataset**: The dataset used to fine-tune the model comes from the GoEmotions dataset by Google Research.
- **Transformers Library**: Hugging Face’s transformers library was used to load and fine-tune BERT.

---

### Notes

- Ensure your environment has sufficient resources (e.g., memory, GPU/CPU) to run the model efficiently.
- If using Apple Silicon (M1/M2/M3), ensure that MPS (Metal Performance Shaders) support is enabled in PyTorch.
