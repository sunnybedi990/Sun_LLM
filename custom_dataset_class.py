from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class QADataset(Dataset):
    def __init__(self, qa_pairs):
        self.qa_pairs = qa_pairs

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        # Return tokenized question and answer
        return {
            "input_ids": question["input_ids"],
            "attention_mask": question["attention_mask"],
            "token_type_ids": question.get("token_type_ids", []),
            "labels": answer["input_ids"]  # Using answer input IDs as labels
        }
