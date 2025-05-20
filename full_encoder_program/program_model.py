import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import tqdm

from Tokenizer import *
from DataLoader import *
from Node import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return x
    
class Model(nn.Module):
    
    def __init__(self, vocab_size, encoder_size=5000, decoder_size=5000, d_model=128, ff_dim= 2048, num_heads= 8, encoder_layers= 6, decoder_layers= 6, dropout= 0.1):
        super(Model, self).__init__()
    
        self._embedding_encoder = nn.Embedding(vocab_size, d_model)
        self._embedding_decoder = nn.Embedding(vocab_size, d_model)

        self._positional_encoder = PositionalEncoding(d_model, encoder_size)
        self._positional_decoder = PositionalEncoding(d_model, decoder_size)

        self._transformer = nn.Transformer(d_model,num_heads, encoder_layers, decoder_layers, ff_dim, batch_first=True, dropout= dropout)
        self._fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, y, mask = None, encoder_mask=None, decoder_mask=None):
        x = self._positional_encoder(self._embedding_encoder(x))
        y = self._positional_decoder(self._embedding_decoder(y))

        output = self._transformer(x, y, tgt_mask=mask,src_key_padding_mask=encoder_mask,tgt_key_padding_mask=decoder_mask)
        
        output = self._fc(output)
        return F.softmax(output, dim=-1)
    






def _train_model(model, optimizer, mask, ds, device, batch_size, PAD_ID):
    losses = []
    ds.shuffle()
    bar = tqdm.tqdm(ds.batch(batch_size),
                f'Training',
                total=len(ds)//batch_size + (1 if len(ds) % batch_size != 0 else 0),
                bar_format='{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}')
    for encoder_input, decoder_input, decoder_output, _, _ in bar:
            
        encoder_input = torch.tensor(encoder_input).to(device)
        decoder_input = torch.tensor(decoder_input).to(device)
        decoder_output = torch.tensor(decoder_output).to(device)

        output = model(encoder_input, decoder_input, mask)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), decoder_output.view(-1), ignore_index=PAD_ID)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if len(losses) % 10 == 0:
            bar.set_description(f'Training, {np.mean(losses[-10:])}')
    return [np.mean(losses[i:i+1000]) for i in range(0, len(losses), 1000)]


def _validate_model(model, mask, ds, device, batch_size, PAD_ID):
    with torch.no_grad():
        ds.shuffle()
        losses = []
        for encoder_input, decoder_input, decoder_output, _, _ in tqdm.tqdm(ds.batch(batch_size),
                                                                            f'Validating',
                                                                            total=len(ds)//batch_size + (1 if len(ds) % batch_size != 0 else 0),
                                                                            bar_format='{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}'):
                
            encoder_input = torch.tensor(encoder_input).to(device)
            decoder_input = torch.tensor(decoder_input).to(device)
            decoder_output = torch.tensor(decoder_output).to(device)

            output = model(encoder_input, decoder_input, mask)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), decoder_output.view(-1), ignore_index=PAD_ID)

            losses.append(loss.item())
    return np.mean(losses)


def _test_model(model, ds, n, device):
    valid, atoms, correct = 0, 0, 0
    if len(ds) < n: n = len(ds)

    with torch.no_grad():
        for encoder_input, encoder_target, labels in tqdm.tqdm(ds.sample(n), f'Testing Model', total=n, bar_format='{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}'):
            for _ in range(5):
                try:
                    seq = [2]
                    encoder_input = torch.tensor([encoder_input]).to(device)

                    for _ in range(ds._decoder_size):
                        decoder_input = torch.tensor([seq]).to(device)
                        output = torch.multinomial(model(encoder_input, decoder_input)[-1][- 1], num_samples=1).item()
                        if output == 3:
                            break
                        else:
                            seq +=[output]

                    target_node = Node.parse(encoder_target)
                    target_node.set_id(labels)
                    output_node = Node.parse(ds._decode(seq)[1:])
                    if output_node:
                        valid += 1
                        try:
                            output_node.set_id(labels)
                            atoms += 1
                            if output_node.key == target_node.key:
                                correct += 1
                            # print(data._decode(seq)[1:], str(target_node), target_node.key, output_node.key)
                        except:
                            pass
                    break
                except:
                    pass
    return valid/n, atoms/n, correct/n

def _run_model(train, test, setx2, setx3, setx4, layers, device, 
               test_size = 1000, batch_size = 128, learning_rate = 1e-4, num_heads = 8, dropout = 0.1, embedding_dim = 128, ff_dim = 2048):

    model = Model(
        vocab_size=len(train._encoded),
        d_model=embedding_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        encoder_layers=layers,
        decoder_layers=layers,
        dropout=dropout).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mask = nn.Transformer.generate_square_subsequent_mask(train._decoder_size, device=device)
    PAD_ID = torch.tensor(0).to(device)
    train_loss = []
    validation_loss = []

    for _ in range(2):
        train_loss += [_train_model(model, optimizer, mask, train, device, batch_size, PAD_ID)]
        validation_loss += [_validate_model(model, mask, test, device, batch_size, PAD_ID)]
    
    generalisation_x2 = _test_model(model, setx2, test_size, device)
    generalisation_x3 = _test_model(model, setx3, test_size, device)
    generalisation_x4 = _test_model(model, setx4, test_size, device)
    return model, train_loss, validation_loss, generalisation_x2, generalisation_x3, generalisation_x4