import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
import custom_dataset_class
import model_maker
from prepare_files import load_and_preprocess_data
from transformers import LongformerTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from transformers import AutoTokenizer

from transformer_encoder import TransformerEncoder, TransformerDecoder, Seq2SeqTransformer

print("Tokenizer")
# Choose a tokenizer (e.g., a tokenizer from a pre-trained model like BERT or GPT)
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
print("Processing Dataset...")
qa_pairs = load_and_preprocess_data()

# Tokenize the questions and answers
print("Processing Tokenizing...")
if os.path.exists('tokenized_qa_pairs.pkl'):
    with open('tokenized_qa_pairs.pkl', 'rb') as file:
        tokenized_qa_pairs = pickle.load(file)
else:
    # Perform tokenization if file doesn't exist
    tokenized_qa_pairs = [
        (tokenizer(question, max_length=512, truncation=True),
         tokenizer(answer, max_length=512, truncation=True))
        for question, answer in tqdm(qa_pairs, desc="Tokenizing")
    ]
    # Save the tokenized pairs for future use
    with open('tokenized_qa_pairs.pkl', 'wb') as file:
        pickle.dump(tokenized_qa_pairs, file)
# Example dataset class

print("Loading Dataset...")

def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask
def collate_batch(batch):
    # Initialize containers for the various components
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for item in batch:
        input_ids.append(torch.tensor(item["input_ids"]))
        attention_masks.append(torch.tensor(item["attention_mask"]))
        token_type_ids.append(torch.tensor(item["token_type_ids"]))
        labels.append(torch.tensor(item["labels"]))

    # Pad the sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "token_type_ids": token_type_ids,
        "labels": labels
    }

dataset = custom_dataset_class.QADataset(tokenized_qa_pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

print("Starting the model training")
# Create dataset and dataloader
#dataset = TextDataset(texts, labels)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Model parameters
vocab_size = len(tokenizer)  # or tokenizer.vocab_size based on the tokenizer used
d_model = 512  # Embedding dimension
nhead = 8  # Number of attention heads
num_encoder_layers = 3  # Number of transformer layers
num_decoder_layers = 3
dim_feedforward = 2048  # Hidden layer size in feed forward network
num_classes = 10  # Number of output classes
num_epochs = 10 # Number of
# Model instantiation
#model = model_maker.TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes)
encoder = TransformerEncoder(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward)
decoder = TransformerDecoder(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward)
model = Seq2SeqTransformer(encoder, decoder)
# Loss function and optimizer
#loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Assuming loss_function is defined as CrossEntropyLoss
loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(num_epochs):
    total_loss = 0
    # Wrap your dataloader with tqdm for a progress bar
    data_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in data_iterator:
        src = batch["input_ids"]  # Source sequence (question)
        trg = batch["labels"]     # Target sequence (answer)
        src_mask = batch["attention_mask"]  # Source sequence mask

        # Assuming trg_mask needs to be generated based on the target sequence
        trg_input = trg[:, :-1]  # Shifted target sequences for decoder input
        trg_len = trg_input.size(1)
        trg_mask = generate_square_subsequent_mask(trg_len)
        print(f"trg shape: {trg.shape}, trg_mask shape: {trg_mask.shape}")  # Debugging

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(src, trg[:, :-1], src_mask=src_mask, trg_mask=trg_mask)

        # Reshape outputs and target_sequences for loss calculation
        # outputs shape: [batch_size, trg_len - 1, vocab_size]
        #target_sequences shape: [batch_size, trg_len]
        outputs = outputs.view(-1, outputs.size(-1))
        targets = trg[:, 1:].reshape(-1)

        # Calculate loss
        loss = loss_function(outputs, targets)
        total_loss += loss.item()

        # Backward pass: compute gradient and update model parameters
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")

print("Model Trained")
# Save the model
torch.save(model.state_dict(), "models/Sun_LLM/model.pth")

# Save the tokenizer
# Assuming tokenizer has a save_pretrained method
tokenizer.save_pretrained("models/Sun_LLM")
print("Model Saved")