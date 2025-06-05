import pandas as pd
import random as rd

class DataLoader:
    def __init__(self, df=None, encoded=None, decoded=None, database=None, encoder_size=None, decoder_size=None, dataframe=True):

        self._isDataFrame = dataframe

        if df is not None: self._load(df)

        else:
            self._encoded = encoded

            self._database = database

            self._encoder_size = encoder_size
            self._decoder_size = decoder_size

            self._size = len(self._database['decoder_input'])

            self._decoded = decoded

    def __len__(self):
        return self._size

    def _load(self, df):
        self._encoded = dict()
        self._encoded['<pad>'] = 0
        self._encoded['<unk>'] = 1
        self._encoded['<bos>'] = 2
        self._encoded['<eos>'] = 3


        self._database = dict()
        if self._isDataFrame:
            self._database['encoder_input'] = [[str(n) for n in m.split(' ')] for m in df['encoder_input'].to_list()]
            self._database['decoder_input'] = [['<bos>'] + [str(n) for n in m.split(' ')] for m in df['decoder_input'].to_list()]
            self._database['decoder_output'] = [[str(n) for n in m.split(' ')] + ['<eos>'] for m in df['decoder_input'].to_list()]
            self._database['atom_label'] = [[str(n) for n in m.split(' ')] for m in df['atom_label'].to_list()]
        else:
            self._database['encoder_input'] = [[str(n) for n in m.split(' ')] for m in df['encoder_input']]
            self._database['decoder_input'] = [['<bos>'] + [str(n) for n in m.split(' ')] for m in df['decoder_input']]
            self._database['decoder_output'] = [[str(n) for n in m.split(' ')] + ['<eos>'] for m in df['decoder_input']]
            self._database['atom_label'] = [[str(n) for n in m.split(' ')] for m in df['atom_label']]

        self._encoder_size = 0
        self._decoder_size = 0

        self._create_tokens()
        self._pad()
        self._encode_all()

        self._database['encoder_mask'] = [[n != 0 for n in m] for m in self._database['encoder_input']]
        self._database['decoder_mask'] = [[n != 0 for n in m] for m in self._database['decoder_input']]

        self._size = len(self._database['decoder_input'])

        self._decoded = {int(v) : str(k) for k, v in self._encoded.items()}

    def _create_tokens(self):
        for seq in self._database['encoder_input']:
            if len(seq) > self._encoder_size:
                self._encoder_size = len(seq)

            for token in seq:
                if token not in self._encoded:
                    self._encoded[token] = len(self._encoded)

        for seq in self._database['decoder_input']:
            if len(seq) > self._decoder_size:
                self._decoder_size = len(seq)
                
            for token in seq:
                if token not in self._encoded:
                    self._encoded[token] = len(self._encoded)

    def _pad(self):
        for i in range(len(self._database['encoder_input'])):
            self._database['encoder_input'][i] += ['<pad>'] * (self._encoder_size - len(self._database['encoder_input'][i]))
            self._database['decoder_input'][i] += ['<pad>'] * (self._decoder_size - len(self._database['decoder_input'][i]))
            self._database['decoder_output'][i] += ['<pad>'] * (self._decoder_size - len(self._database['decoder_output'][i]))

    def _encode_all(self):
        for i in range(len(self._database['encoder_input'])):
            self._database['encoder_input'][i] = self._encode(self._database['encoder_input'][i])
            self._database['decoder_input'][i] = self._encode(self._database['decoder_input'][i])
            self._database['decoder_output'][i] = self._encode(self._database['decoder_output'][i])

    def __len__(self):
        return self._size

    def _encode(self, tokens):
        return [self._encoded[token] for token in tokens]
    
    def _decode(self, tokens):
        return [self._decoded[token] for token in tokens]

    def split(self, ratio):

        l_split = DataLoader(encoded=self._encoded, decoded= self._decoded,
                             encoder_size= self._encoder_size, decoder_size= self._decoder_size,
                             database={'encoder_input':self._database['encoder_input'][:int(ratio * self._size)], 
                 'decoder_input':self._database['decoder_input'][:int(ratio * self._size)],
                 'decoder_output':self._database['decoder_output'][:int(ratio * self._size)], 
                 'encoder_mask':self._database['encoder_mask'][:int(ratio * self._size)],
                 'decoder_mask':self._database['decoder_mask'][:int(ratio * self._size)],
                 'atom_label':self._database['atom_label'][:int(ratio * self._size)]})
        
        r_split = DataLoader(encoded=self._encoded, decoded= self._decoded,
                             encoder_size= self._encoder_size, decoder_size= self._decoder_size,
                             database={'encoder_input':self._database['encoder_input'][int(ratio * self._size):], 
                 'decoder_input':self._database['decoder_input'][int(ratio * self._size):],
                 'decoder_output':self._database['decoder_output'][int(ratio * self._size):], 
                 'encoder_mask':self._database['encoder_mask'][int(ratio * self._size):],
                 'decoder_mask':self._database['decoder_mask'][int(ratio * self._size):],
                 'atom_label':self._database['atom_label'][int(ratio * self._size):]})

        return (l_split, r_split)
    
    def sample(self, n = 1):
        for i in rd.sample(list(range(len(self))), n):
            yield (
                self._database['encoder_input'][i][:self._database['encoder_input'][i].index(0)] if 0 in self._database['encoder_input'][i] else \
                    self._database['encoder_input'][i],
                self._decode(self._database['decoder_output'][i][:self._database['decoder_output'][i].index(3)]),
                self._database['atom_label'][i]
            )
    def shuffle(self):
        for i in range(len(self._database['encoder_input'])):
            j = rd.randint(0, len(self._database['encoder_input']) - 1)
            self._database['encoder_input'][i], self._database['encoder_input'][j] = self._database['encoder_input'][j], self._database['encoder_input'][i]
            self._database['decoder_input'][i], self._database['decoder_input'][j] = self._database['decoder_input'][j], self._database['decoder_input'][i]
            self._database['decoder_output'][i], self._database['decoder_output'][j] = self._database['decoder_output'][j], self._database['decoder_output'][i]
            self._database['atom_label'][i], self._database['atom_label'][j] = self._database['atom_label'][j], self._database['atom_label'][i]
    def batch(self, batch_size = 32):
        for i in range(0, self._size, batch_size):
            yield (self._database['encoder_input'][i: min(self._size, i + batch_size)],
                   self._database['decoder_input'][i: min(self._size, i + batch_size)],
                   self._database['decoder_output'][i: min(self._size, i + batch_size)],
                   self._database['encoder_mask'][i: min(self._size, i + batch_size)],
                   self._database['decoder_mask'][i: min(self._size, i + batch_size)])