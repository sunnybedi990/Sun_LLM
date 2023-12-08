import torch.nn as nn
import math
import torch
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src, attention_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        output = self.transformer_encoder(src, src_key_padding_mask=attention_mask)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        #self.d_model = d_model
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None):
        trg = self.embedding(trg) * math.sqrt(self.d_model)
        trg = self.pos_decoder(trg)
        if trg_mask is not None:
            trg_mask = trg_mask.bool()
        if memory_mask is not None:
            memory_mask = memory_mask.bool()
        output = self.transformer_decoder(trg, memory, tgt_mask=trg_mask, memory_key_padding_mask=memory_mask)
        output = self.output_layer(output)
        return output

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # Encoding
        memory = self.encoder(src, src_mask)
        # Decoding
        output = self.decoder(trg, memory, trg_mask)
        return output