class Tokenizer:
    def __init__(self, tokens = None, default = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}):
        self._encoded = default | {str(tokens[i]) : i + 4 for i in range(len(tokens))} if default else {str(tokens[i]) : i for i in range(len(tokens))}
        self._decoded = {int(v) : str(k) for k, v in self._encoded.items()}
        self._length = len(self._encoded)

    def __add__(self, value):
        if value not in self._encoded:
            self._encoded[str(value)] = self._length
            self._decoded[self._length] = str(value)
            self._length += 1

    def __len__(self):
        return self._length

    def tokenize(self, value):
        value = value.lower()
        output = []

        for seq in value.split():
            if seq in self._encoded:
                output += [seq]
            else:
                split_seq = self._tokenize(seq)
                if split_seq is None: output += ['<unk>']
                else: output += split_seq

        return output
    
    def encode(self, sequence):
        return [self._encoded[s] if s in self._encoded else -1 for s in sequence]
    
    def decode(self, sequence):
        return [self._decoded[s] if s in self._decoded else '<unk>' for s in sequence]

    def _tokenize(self, value):
        if value in self._encoded: return [value]
        for i in range(1, len(value)):
            if value[:i] in self._encoded:
                right = self._tokenize(value[i:])
                if right:
                    return [value[:i]] + right
        return None
        

    def get_items(self):
        return [(self._decoded[i], i) for i in self._decoded]

    def add(self, value):
        self += value
