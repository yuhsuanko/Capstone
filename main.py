import torch
import pandas as pd
import configparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import logging

# === Set up logging ===
logging.basicConfig(
    filename='inference.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s | %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# === Read config.ini (need to adjust path if config.ini in different path) ===
config = configparser.ConfigParser()
config.read("/content/drive/MyDrive/Colab Notebooks/Capstone/config.ini")

MODEL_ID = config["model"]["model_id"]
CHECKPOINT_PATH = config["model"]["checkpoint_path"]
TEST_CSV = config["data"]["test_csv"]
TEXT_COL = config["data"]["text_column"]
LABEL_COL = config["data"]["label_column"]
SAVE_REPORT_PATH = config["output"]["save_report_path"]
SAVE_PRED_PATH = config["output"]["save_predictions_path"]

# === Load base model + LoRA adapter ===
def load_model_and_tokenizer(checkpoint_path, base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=2,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

# === Inference function for one input ===
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=-1).item()
    return label, probs[0].tolist()

# === Main entry point ===
def main():
    logging.info("Loading test data...")
    df = pd.read_csv(TEST_CSV)

    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH, MODEL_ID)

    preds, probs, labels = [], [], []

    logging.info("Running inference...")
    for text in tqdm(df[TEXT_COL], desc="Predicting"):
        label, prob = predict(text, model, tokenizer)
        preds.append(label)
        probs.append(prob)

    # If ground truth labels are present, generate classification report
    if LABEL_COL in df.columns:
        labels = df[LABEL_COL].tolist()
        report = classification_report(labels, preds, digits=4)
        logging.info("\n" + report)
        with open(SAVE_REPORT_PATH, "w") as f:
            f.write(report)
        logging.info(f"Saved classification report to: {SAVE_REPORT_PATH}")

    # Save prediction results
    df["predicted_label"] = preds
    df["prob_class_0"] = [p[0] for p in probs]
    df["prob_class_1"] = [p[1] for p in probs]
    df.to_csv(SAVE_PRED_PATH, index=False)
    logging.info(f"Saved predictions to: {SAVE_PRED_PATH}")

if __name__ == "__main__":
    main()
