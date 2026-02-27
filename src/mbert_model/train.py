from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from src.data_utils import load_and_prepare_data, split_data

MODEL_NAME = "bert-base-multilingual-cased"
DATA_PATH = "./data/spam_ham_dataset.csv"
SAVE_PATH = "./models/mbert/"


class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    print("Loading data...")
    df = load_and_prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    train_dataset = EmailDataset(train_encodings, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(2):
        print(f"Epoch {epoch + 1}/2")
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} complete")

    print("Saving model...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Training complete.")


if __name__ == "__main__":
    main()
