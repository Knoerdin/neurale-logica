import tqdm

from Tokenizer import *
from DataLoader import *
from Node import *


alphabet = list('abcdefghijklmnopqrstuvwxyz')
tokenizer = Tokenizer(Node.all_formula_tokens + alphabet)

def _add_split(sequence, allowed_split, dataset, output_keys):
    split = rd.choice(allowed_split)
    if split in output_keys:
        new_split = rd.choices(dataset[split]['splits'],weights=dataset[split]['info_gain'])[0]
        if len(set(new_split) | set(sequence)) == 0:
            sequence.remove(split)
            sequence += new_split
    return sequence


def _create_datapoint(dataset: dict, output_keys, output_info):
    label = rd.choices(output_keys, weights=output_info, k= 1)[0]
    sequence = rd.choices(dataset[label]['splits'],weights=dataset[label]['info_gain'])[0][:]
    allowed_split = [seq for seq in sequence if seq in output_keys]
    if allowed_split and rd.random() < 2/3:
        sequence = _add_split(sequence, allowed_split, dataset, output_keys)
        allowed_split = [seq for seq in sequence if seq in output_keys]

        if allowed_split and rd.random() < 1/2: sequence = _add_split(sequence, allowed_split, dataset, output_keys)
    
    rd.shuffle(sequence)

    return sequence, label


def _create_splits(dataset: dict):
    output_keys = []
    output_info = []
    for key in tqdm.tqdm(dataset, f'Check Keys', bar_format='{desc:<11}{percentage:3.0f}%|{bar:25}{r_bar}'):
        if dataset[key]['splits']:
            output_keys += [key]
            output_info += [dataset[key]['info']]
    return output_keys, output_info


def key_to_form(key, dataset):
    label = rd.choice(dataset[key]['formulas'])
    return rd.choice(label['notations'])


def _create_notations(dataset: dict, labels, inputs: list):
    encoder_input = []
    decoder_input = []
    atom_labels = []

    for i in tqdm.tqdm(range(len(labels)), f'Creating datapoints', bar_format='{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}'):
        atoms = rd.sample(alphabet, 4)
        label = key_to_form(labels[i], dataset)
        label.set_label(atoms)
        decoder_input += [' '.join(tokenizer.tokenize(str(label)))]
        decod = []
        for key in inputs[i]:
            key = key_to_form(key, dataset)
            key.set_label(atoms)
            decod += tokenizer.tokenize(str(key)) + [',']
        decod = decod[:-1]
        encoder_input += [' '.join(decod)]
        atom_labels += [' '.join(atoms)]

    return encoder_input, decoder_input, atom_labels


def _create_dataset(dataset: dict, num_points: int):
    output_keys, output_info = _create_splits(dataset)
    
    labels = []
    sequences = []

    for i in tqdm.tqdm(range(num_points), f'Creating Dataset', bar_format='{desc:<11}{percentage:3.0f}%|{bar:25}{r_bar}'):
        sequence, label = _create_datapoint(dataset, output_keys, output_info)
        labels.append(label)
        sequences.append(sequence)

    encoder_input, decoder_input, atom_labels = _create_notations(dataset, labels, sequences)

    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'atom_label': atom_labels
        }