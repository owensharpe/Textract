"""
File: dataset.py
Author:
Description: Building dataset for modeling
"""

# import libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


# vocabulary for the LaTeX tokens
class LaTeXVocab:
    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.token_to_idx = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.token_freq = Counter()

    # building vocabulary from the latex expressions
    def build_vocab(self, latex_expressions):
        """
        :param latex_expressions: given list of latex expressions
        :return: Null (print size of vocab and most common expressions)
        """

        # count token frequencies
        for expression in latex_expressions:
            tokens = self.tokenize(expression)
            self.token_freq.update(tokens)

        for token, freq in self.token_freq.items():
            if freq >= self.min_freq and token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token

        print(f"Vocabulary Size: {len(self.token_to_idx)}")
        print(f"Most Common Tokens (First 10): {self.token_freq.most_common(10)}")

    # tokenizing given expression
    def tokenize(self, expression):
        """
        :param expression: a given latex expression
        :return: the list of tokens from the latex expression
        """

        # replace latex commands with simpler tokens
        replacements = {
            '\\frac': ' \\frac ',
            '\\sum': ' \\sum ',
            '\\int': ' \\int ',
            '\\sqrt': ' \\sqrt ',
            '\\times': ' \\times ',
            '\\alpha': ' \\alpha ',
            '\\beta': ' \\beta ',
            '\\gamma': ' \\gamma ',
            '\\rightarrow': ' \\rightarrow ',
            '\\leq': ' \\leq ',
            '\\geq': ' \\geq ',
            '\\ldots': ' \\ldots ',
            '\\mbox': ' \\mbox ',
            '\\prime': ' \\prime ',
            '^': ' ^ ',
            '_': ' _ ',
            '{': ' { ',
            '}': ' } ',
            '(': ' ( ',
            ')': ' ) ',
            '+': ' + ',
            '-': ' - ',
            '=': ' = ',
            '*': ' * ',
            '/': ' / ',
        }

        # apply these simpler symbols
        for old_syntax, new_syntax in replacements:
            expression = expression.replace(old_syntax, new_syntax)

        # split tokens into list
        tokens = [t for t in expression.split() if t]

        return tokens

    # encode the latex expressions to token indices
    def encode(self, expression, max_length):
        """
        :param expression: the provided latex expression
        :param max_length: the maximum length we might specify (if provided)
        :return: the encoded indices
        """

        # tokenize the expression
        tokens = ['<SOS>'] + self.tokenize(expression) + ['<EOS>']

        # if we provide a max length
        if max_length:
            tokens = tokens[:max_length]

        indices = []
        for token in tokens:
            if token in self.token_to_idx:  # if it's a known token
                indices.append(self.token_to_idx[token])
            else:
                indices.append(self.token_to_idx['UNK'])  # pass the unknown token if we don't know it
        return indices

    # decode the token indices into a latex expression
    def decode(self, indices, skip_special):
        """
        :param indices: the encoded indices
        :param skip_special: if we want to skip special tokens
        :return: decoded latex expression
        """
        tokens = []
        for idx in indices:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                if skip_special and token in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                    continue
                tokens.append(token)

        # fix original changes made to expression if present
        expression = ' '.join(tokens)
        expression = expression.replace(' ^ ', '^')
        expression = expression.replace(' _ ', '_')
        expression = expression.replace(' { ', '{')
        expression = expression.replace(' } ', '}')
        expression = expression.replace(' ( ', '(')
        expression = expression.replace(' ) ', ')')
        expression = expression.replace(' + ', '+')
        expression = expression.replace(' - ', '-')
        expression = expression.replace(' = ', '=')
        expression = expression.replace(' * ', '*')
        expression = expression.replace(' / ', '/')
        return expression.strip()

    # saving the vocabulary to a file
    def save(self, path):
        """
        :param path: specified file path
        :return: Null (saving vocabulary)
        """

        with open(path, 'w') as file:
            json.dump({
                'token_to_idx': self.token_to_idx,
                'min_freq': self.min_freq
                }, file, indent=2)

    # loading the vocabulary from a file
    def load(self, path):
        """
        :param path: specified file path
        :return: Null (loading vocabulary)
        """

        with open(path, 'r') as file:
            data = json.load(file)
            self.token_to_idx = data['token_to_idx']
            self.idx_to_token = {int(idx): token for token, idx in self.token_to_idx.items()}
            self.min_freq = data['min_freq']


# Pytorch library for CROHME dataset
class CROHMEDataset(Dataset):

    def __init__(self, root_dir, split, vocab=None, transform=None, max_seq_length=256):
        """
        :param root_dir: specified path to crohme_images directory
        :param split: dataset type ('train', 'val', or 'test')
        :param vocab: LaTeXVocabulary object (gets created if not already made)
        :param transform: image transformations
        :param max_seq_length: max sequence length for LaTeX
        """

        self.root_dir = root_dir
        self.split = split
        self.max_seq_length = max_seq_length

        # provide default transform if not provided
        if transform is None:
            self.transform = transform.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.9], std=[0.1])
            ])
        else:
            self.transform = transform

        # load image paths as well as label data
        self.samples = []
        self._load_data()

        # build or use provided vocabulary
        if split == 'train' and vocab is None:
            self.vocab = LaTeXVocab(min_freq=2)
            expressions = [latex for _, latex in self.samples]
            self.vocab.build_vocab(expressions)
        else:
            self.vocab = vocab

    # load image path and label data
    def _load_data(self):
        """
        :return: Null (importing data)
        """

        # try to load the path for the labels
        label_path = os.path.join(self.root_dir, 'labels.txt')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found at {label_path}")

        # read labels
        samples = []
        with open(label_path, 'r', encoding='utf-8') as file:
            for line in file:
                sections = line.strip().split('\t')
                if len(sections) == 2:
                    filename, latex = sections
                    samples.append((filename, latex))

        # filter data by dataset partition
        split_dir = os.path.join(self.root_dir, self.split)
        for filename, latex in samples:
            for root, dirs, files in os.walk(split_dir):  # check if file belongs in data partition
                if filename in files:
                    img_path = os.path.join(root, filename)
                    if os.path.exists(img_path):
                        self.samples.append((img_path, latex))
                    break
        print(f"Loaded {len(self.samples)} samples for {self.split} split")

    # standard dataset length function
    def __len__(self):
        """
        :return: length of dataset
        """
        return len(self.samples)

    # get data sample
    def __getitem__(self, idx):
        """
        :param idx: specified index of dataset
        :return: data point at specific index
        """

        # load and transform image
        img_path, latex = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # encode latex
        encoded_latex = self.vocab.encode(latex, self.max_seq_length)

        # now create a tensor
        latex_tensor = torch.LongTensor(encoded_latex)

        return {
            'image': img,
            'latex': latex_tensor,
            'latex_text': latex,
            'image path': img_path
        }


# handling variable length sequences
def collate(batch):
    """
    :param batch: specified batch from dataset
    :return: grouping of items
    """

    images = torch.stack([item['image'] for item in batch])

    # pad the latex sequences
    latex_seqs = [item['latex'] for item in batch]
    max_len = max([len(seq) for seq in latex_seqs])
    padded_seqs = []
    for seq in latex_seqs:
        padded = torch.cat([
            seq,
            torch.zeros(max_len - len(seq), dtype=torch.long)
        ])
        padded_seqs.append(padded)
    latex_tensors = torch.stack(padded_seqs)

    return {
        'images': images,
        'latex': latex_tensors,
        'latex_texts': [item['latex_text'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }

# creating data loaders for dataset
def create_dataloaders(root_dir, batch_size=32, num_workers=4, transform=None):
    """
    :param root_dir: specified path to crohme_images directory
    :param batch_size: number of batches in dataset
    :param num_workers: number of parallels computations, in essence
    :param transform: image transformations
    :return: data in the form of data loaders
    """

    # make train dataset and vocabulary
    train_dataset = CROHMEDataset(root_dir, 'train', transform=transform)
    vocab = train_dataset.vocab

    # make val and test datasets with same vocabulary
    val_dataset = CROHMEDataset(root_dir, 'val', vocab=vocab, transform=transform)
    test_dataset = CROHMEDataset(root_dir, 'test', vocab=vocab, transform=transform)

    # make data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, vocab
