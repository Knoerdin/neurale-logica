import tqdm

class Node:
    _atom_keys: list[int] = [21845, 13107, 3855, 255]
    _key_mask: int = 65535

    all_formula_tokens: list[str] = ['p0', 'p1', 'p2', 'p3', '(', ')', '!', '|', '&', '=>', '<=>']

    precedence_parentheses = {'weight': 0, 'bound': 'x'}
    precedence_variable = {'weight': 0, 'bound': 'x'}
    precedence_negation = {'weight': 0, 'bound': 'x'}
    precedence_implication = {'weight': 3, 'bound': 'r'}
    precedence_disjunction = {'weight': 2, 'bound': 'l'}
    precedence_conjuntion = {'weight': 1, 'bound': 'l'}
    precedence_biimplication = {'weight': 4, 'bound': 'r'}

    def __init__(self, key=None, depth=None, length=None, info=None) -> None:
        self._key: int = key
        self._depth: int = depth
        self._length: int = length
        self._info: float = info

        self._formula: str = None

    def __str__(self) -> str: return self.formula

    def __eq__(self, other):
        return isinstance(other, Node) and self.key == other.key
    
    def similar(self, other): pass

    def set_label(self, labels):
        self._left.set_label(labels)
        self._right.set_label(labels)
        self._formula = None

    def set_id(self, labels):
        self._left.set_id(labels)
        self._right.set_id(labels)
        self._key = None

    def is_pure(self):
        for a in self._get_atoms():
            pos, neg = [], []
            for index, i in enumerate(format(self.key,'#018b')[2:]):
                (pos, neg)['0' == format(self._atom_keys[a],'#018b')[index + 2]].append(i)
            if pos == neg: return False
        return True

    @property
    def key(self) -> int:
        if self._key is None: self._key = self._get_key()
        return self._key

    @property
    def depth(self) -> int:
        if self._depth is None: self._depth = self._get_depth()
        return self._depth

    @property
    def length(self) -> int:
        if self._length is None: self._length = self._get_length()
        return self._length
    
    @property
    def info(self) -> int:
        if self._info is None: self._info = self._get_info()
        return self._info

    @property
    def formula(self) -> str:
        if self._formula is None: self._formula = self._get_formula()
        return self._formula
    
    def _get_priority(self): return self.precedence['weight'] / 10 + self._left._get_priority() + self._right._get_priority()

    def _get_atoms(self): return self._left._get_atoms() | self._right._get_atoms()

    def _get_formula(self) -> str:
        return f'{self._left._get_formula()} {self._connective} {self._right._get_formula()}'

    def _get_key(self) -> int: pass

    def _get_depth(self) -> int: return max(self._left.depth, self._right.depth) + 1

    def _get_length(self) -> int: return self._left.length + self._right.length

    def _get_info(self) -> float:
        return 1 - (sum([1 for i in range(16) if (self.key >> i) & 1 == 1]) / 16)
    
    @staticmethod
    def generate(leaves):
        if leaves < 1: return

        for i in range(4):
            yield Variable(id=i)
            yield Negation(Variable(id=i))

        if leaves == 1: return

        for i in range(1, leaves):
            for left in Node.generate(i):
                if left.key == 0: continue
                for right in Node.generate(leaves - i):
                    if right.key == 0: continue
                    if right == left: continue

                    # Implication
                    output = Node._generate_child(Implication(left, right))
                    if output and output.key != 0 and output.key != Node._key_mask: yield output 
                    
                    if left._get_priority() <= right._get_priority():
                        # Disjunction
                        output = Node._generate_child(Disjunction(left, right))
                        if output and output.key != 0 and output.key != Node._key_mask: yield output 
                        
                        # Conjunction
                        output = Node._generate_child(Conjunction(left, right))
                        if output and output.key != 0 and output.key != Node._key_mask: yield output 
                        
                        # Biimplication
                        output = Node._generate_child(Biimplication(left, right))
                        if output and output.key != 0 and output.key != Node._key_mask: yield output

    @staticmethod
    def _generate_child(option):
        if option._left.key != option.key and option._right.key != option.key:

            if not isinstance(option._left,(Negation, Variable, Parentheses)):
                if option.precedence['weight'] < option._left.precedence['weight']:
                    option._left = Parentheses(option._left)

                if option.precedence['weight'] == option._left.precedence['weight'] and option.precedence['bound'] == 'r':
                    option._left = Parentheses(option._left)

            if not isinstance(option._right,(Negation, Variable, Parentheses)):
                if option.precedence['weight'] < option._right.precedence['weight']:
                    option._right = Parentheses(option._right)

                if option.precedence['weight'] == option._left.precedence['weight'] and option.precedence['bound'] == 'l':
                    option._right = Parentheses(option._right)
                
            return option

    @staticmethod
    def split_key(key, keys):
        for i in keys:
            if i < key: continue
            
            for j in keys:
                if j < key: continue
            
                if i != key and j != key and i > j and i & j == key:
                    yield i, j

    @staticmethod
    def info_gain(target, *current) -> float:
        return sum([target.info - c.info for c in current]) / len(current)
    
    @staticmethod
    def info_key(key) -> float:
        return 1 - (sum([1 for i in range(16) if (key >> i) & 1 == 1]) / 16)

    @staticmethod
    def info_gain_key(target, *current) -> float:
        return max([Node.info_key(target)- Node.info_key(c) for c in current]) / len(current)
    
    @staticmethod
    def split_all_keys(keys):
        output = dict()
        for i in tqdm.tqdm(range(len(keys)), f'Matching Keys',bar_format='{desc:<30}{percentage:3.0f}%|{bar:25}{r_bar}'):
            for j in range(i):
                value = keys[i] & keys[j]
                if value in keys and value != keys[i] and value != keys[j]:
                    if value not in output:
                        output[value] = {'split': [], 'info': []}
                    output[value]['split'] += [[keys[i], keys[j]]]
                    output[value]['info'] += [Node.info_gain_key(value, keys[i], keys[j]) ** 2]
        return output
    
    @staticmethod
    def create_dataset(leaves = 4):
        database = dict()
        for f in tqdm.tqdm(Node.generate(leaves), f'Generating expression for {leaves} leave nodes',unit=' expr'):
            if f.is_pure():
                if f.key in database:
                    if f.length < database[f.key]['length']:
                        database[f.key]['length'] = f.length
                        database[f.key]['formulas'] = [{'leader':f, 'notations': [f]}]
                    elif f.length == database[f.key]['length']:
                        grouped = False
                        for group in database[f.key]['formulas']:
                            if group['leader'].similar(f):
                                grouped = True
                                group['notations'] += [f]
                                break
                        if not grouped:
                            database[f.key]['formulas'] += [{'leader':f, 'notations': [f]}]
                else:
                    database[f.key] = dict()
                    database[f.key]['length'] = f.length
                    database[f.key]['formulas'] = [{'leader':f, 'notations': [f]}]

        keys = Node.split_all_keys(list(database.keys()))
        for key in tqdm.tqdm(database, f'Creating information for keys', bar_format='{desc:<30}{percentage:3.0f}%|{bar:25}{r_bar}'):
            database[key]['info'] = Node.info_key(key) ** 2
            if key not in keys:
                database[key]['splits'] = None
            else:
                database[key]['splits'] = keys[key]['split']
                database[key]['info_gain'] = keys[key]['info']
        return database
    
    @staticmethod
    def parse(tokens):
        stack = []
        tokens.reverse()
        while tokens:
            stack.append(tokens.pop())
            reduced, stack = Node._reduce(stack)
            while reduced:
                reduced, stack = Node._reduce(stack)
        return stack[0] if len(stack) == 1 else None

    @staticmethod
    def _reduce(stack):
        for i in range(len(stack)):
            matched = Node._match(stack[- ( 1 + i) :])
            if matched:
                return True, stack[: - ( 1 + i)] + matched
        return False, stack
    
    @staticmethod
    def _match(stack):
        match stack:
            case ['p0']:
                return [Variable(0)]
            case ['p1']:
                return [Variable(1)]
            case ['p2']:
                return [Variable(2)]
            case ['p3']:
                return [Variable(3)]
            case ['!', x] if isinstance(x, Node):
                return [Negation(x)]
            case ['(', x, ')'] if isinstance(x, Node):
                return [Parentheses(x)]
            case [l, '<=>', r] if isinstance(l, Node) and isinstance(r, Node):
                return [Node._combine(l, Biimplication(None, r))]
            case [l, '=>', r] if isinstance(l, Node) and isinstance(r, Node):
                return [Node._combine(l, Implication(None, r))]
            case [l, '&', r] if isinstance(l, Node) and isinstance(r, Node):
                return [Node._combine(l, Conjunction(None, r))]
            case [l, '|', r] if isinstance(l, Node) and isinstance(r, Node):
                return [Node._combine(l, Disjunction(None, r))]
            case [x] if isinstance(x, str) and x.isalpha() and len(x) == 1:
                return [Variable(label=x)]

        return None
    
    @staticmethod
    def _combine(left, right):
        if left.precedence['weight'] < right.precedence['weight']:
            right._left = left
            return right
        if left.precedence['weight'] > right.precedence['weight']:
            left._right = Node._combine(left._right, right)
            return left
        if left.precedence['bound'] != right.precedence['bound']:
            return None
        if left.precedence['bound'] == 'l':
            right._left = left
            return right
        if left.precedence['bound'] == 'r':
            left._right = Node._combine(left._right, right)
            return left
        return None


class Parentheses(Node):
    def __init__(self, expression):
        super().__init__()
        self._expression = expression
        self.precedence = Node.precedence_parentheses

    def similar(self, other): return self._expression.similar(other)

    def set_id(self, labels):
        self._expression.set_id(labels)
        self._key = None

    def set_label(self, labels):
        self._expression.set_label(labels)
        self._formula = None

    def _get_priority(self): return self._expression._get_priority()

    def _get_atoms(self): return self._expression._get_atoms()

    def _get_formula(self) -> str:
        return f'({self._expression._get_formula()})'
    
    def _get_key(self) -> int: return self._expression._get_key()

    def _get_depth(self) -> int: return self._expression._get_depth()

    def _get_length(self) -> int: return self._expression._get_length()


class Variable(Node):
    def __init__(self, id: int = -1, label= None) -> None:
        super().__init__(key= Node._atom_keys[id], depth=0, length=1, info=0.5)
        self._id = id
        self._label = f'p{id}' if label is None else label
        self.precedence = Node.precedence_variable
    
    def similar(self, other): return self.key != other.key or not isinstance(other, Variable)

    def set_id(self, labels):
        if self._label in labels:
            self._id = labels.index(self._label)
        else:
            raise ValueError(f'No Valid Label. Label: {self._label}, Given Labels: {labels}')
        self._key = Node._atom_keys[self._id]

    def set_label(self, labels):
        if self._id < len(labels):
            self._label = labels[self._id]
        else:
            raise IndexError(f'Not enough labels, {len(labels)} where given. Atleast {self._id + 1} are needed')
        self._formula = None

    def _get_priority(self): return self._id

    def _get_atoms(self): return {self._id}

    def _get_formula(self) -> str:
        return self._label
    
    def _get_length(self) -> int: return 1


class Negation(Node):
    def __init__(self, expression) -> None:
        super().__init__()
        self._expression: Variable = expression
        self.precedence = Node.precedence_negation

    def similar(self, other): return self.key != other.key or not isinstance(other, Negation)

    def set_id(self, labels):
        self._expression.set_id(labels)
        self._key = None

    def set_label(self, labels):
        self._expression.set_label(labels)
        self._formula = None

    def _get_priority(self): return self._expression._get_priority()

    def _get_atoms(self): return {self._expression._id}

    def _get_formula(self) -> str:
        return f'!{self._expression._get_formula()}'
    
    def _get_key(self) -> int: return Node._key_mask - self._expression.key

    def _get_depth(self) -> int: return self._expression.depth

    def _get_length(self) -> int: return self._expression._get_length()


class Disjunction(Node):
    def __init__(self, left, right) -> None:
        super().__init__()
        self._left: Node = left
        self._right: Node = right
        self._connective: str = '|'
        self.precedence = Node.precedence_disjunction

    def _get_key(self) -> int: return self._left.key | self._right.key

    def similar(self, other):
        if self.key != other.key or not isinstance(other, Disjunction): return False
        return ((self._left.similar(other._left) and self._right.similar(other._right)) or 
                (self._left.similar(other._right) and self._right.similar(other._left)))


class Conjunction(Node):
    def __init__(self, left, right) -> None:
        super().__init__()
        self._left: Node = left
        self._right: Node = right
        self._connective: str = '&'
        self.precedence = Node.precedence_conjuntion
    
    def _get_key(self) -> int: return self._left.key & self._right.key

    def similar(self, other):
        if self.key != other.key or not isinstance(other, Conjunction): return False
        return ((self._left.similar(other._left) and self._right.similar(other._right)) or 
                (self._left.similar(other._right) and self._right.similar(other._left)))


class Implication(Node):
    def __init__(self, left, right) -> None:
        super().__init__()
        self._left: Node = left
        self._right: Node = right
        self._connective: str = '=>'
        self.precedence = Node.precedence_implication

    def _get_key(self) -> int: return self._left.key | (Node._key_mask - self._right.key)

    def similar(self, other):
        if self.key != other.key or not isinstance(other, Implication): return False
        return self._left.similar(other._left) and self._right.similar(other._right)


class Biimplication(Node):
    def __init__(self, left, right) -> None:
        super().__init__()
        self._left: Node = left
        self._right: Node = right
        self._connective: str = '<=>'
        self.precedence = Node.precedence_biimplication

    def _get_key(self) -> int: return Node._key_mask - (self._left.key ^ self._right.key)

    def similar(self, other):
        if self.key != other.key or not isinstance(other, Biimplication): return False
        return ((self._left.similar(other._left) and self._right.similar(other._right)) or 
                (self._left.similar(other._right) and self._right.similar(other._left)))
    

if __name__ == '__main__':
    x = Node.create_dataset()
    # from Tokenizer import *
    # import random as rd
    # alphabet = list('abcdefghijklmnopqrstuvwxyz')


    # tokenizer = Tokenizer(Node.all_formula_tokens + alphabet)
    # keys = set()
    # labels = rd.sample(alphabet, 4)
    # for f in tqdm.tqdm(Node.generate(4), f'Generating',unit=' expr'):
    #     if f.key not in keys:
    #         keys.add(f.key)
    #     f.set_label(labels)
    #     parsed = Node.parse(tokenizer.tokenize(str(f)))
    #     parsed.set_id(labels)

    #     if f.key != parsed.key:
    #         raise ValueError(str(f), f.key,str(parsed), parsed.key)
        
    # print(len(keys))