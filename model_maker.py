import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Embedding and positional encoding on input_ids
        src = self.embedding(input_ids) * math.sqrt(d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        # Transpose attention_mask to match the expected shape (batch_size, seq_length)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            # If the shape is transposed, correct it
            if attention_mask.shape != (src.size(0), src.size(1)):
                attention_mask = attention_mask.transpose(0, 1)

        output = self.transformer_encoder(src, src_key_padding_mask=attention_mask)
        output = self.output_layer(output.mean(dim=1))

        # If labels are provided (e.g., for training), calculate loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output.view(-1, self.num_classes), labels.view(-1))
            return loss

        return output  # Return logits in case of no labels (e.g., for inference)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

# Model parameters
vocab_size = 10000  # Size of your vocabulary
d_model = 512  # Embedding dimension
nhead = 8  # Number of attention heads
num_encoder_layers = 3  # Number of transformer layers
dim_feedforward = 2048  # Hidden layer size in feed forward network
num_classes = 10  # Number of output classes
# Create the model
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes)