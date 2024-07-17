
from typing import Union, Iterable, Callable
import random

import torch.nn as nn
import torch


def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os
    
    

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### 1.1 Batching, shuffling, iteration
def build_loader(data_dict: dict, batch_size: int = 64, shuffle: bool = False, seed: int = None) -> Callable[[], Iterable[dict]]:

    def loader():
        keys = list(data_dict.keys())
        data_length = len(data_dict[keys[0]])

        indices = list(range(data_length))
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        for start_idx in range(0, data_length, batch_size):
            end_idx = min(start_idx + batch_size, data_length)
            batch_indices = indices[start_idx:end_idx]

            batch_data = {key: [data_dict[key][i] for i in batch_indices] for key in keys}
            yield batch_data

    return loader


def convert_to_tensors(text_indices: list) -> torch.Tensor:
    # Determine maximum sequence length
    longest_seq = max(map(len, text_indices))

    # Initialize a zero tensor with appropriate size
    padded_tensor = torch.zeros((len(text_indices), longest_seq), dtype=torch.int32)

    # Fill in the tensor with actual sequences
    for idx, sequence in enumerate(text_indices):
        padded_tensor[idx, :len(sequence)] = torch.tensor(sequence, dtype=torch.int32)

    return padded_tensor

### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    # Apply max pooling along the sequence length dimension
    return torch.max(x, dim=1)[0]

class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
        hidden_size = embedding.embedding_dim
        self.layer_pred = nn.Linear(2 * hidden_size, 1)  # Output dimension for binary classification
        self.sigmoid = nn.Sigmoid()

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        premise_embed = self.embedding(premise)
        hypothesis_embed = self.embedding(hypothesis)

        pooled_premise = max_pool(premise_embed)
        pooled_hypothesis = max_pool(hypothesis_embed)

        # Concatenate pooled tensors
        concatenated = torch.cat((pooled_premise, pooled_hypothesis), dim=1)

        logits = self.layer_pred(concatenated)

        output = self.sigmoid(logits)

        # So that output is of correct shape
        output = output.squeeze()

        return output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### 2.2 Choose an optimizer and a loss function

def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    # Default learning rate
    learning_rate = kwargs.pop('lr', 0.001)

    # Using Adam optimizer with the specified or default learning rate
    return optim.Adam(model.parameters(), lr=learning_rate, **kwargs)


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Compute Binary Cross-Entropy loss
    return F.binary_cross_entropy(y_pred, y)


### 2.3 Forward and backward pass

def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    model = model.to(device)

    # Process inputs and transfer to device
    premise_data = convert_to_tensors(batch['premise']).to(device)
    hypothesis_data = convert_to_tensors(batch['hypothesis']).to(device)

    predictions = model(premise_data, hypothesis_data)

    return predictions

def backward_pass(optimizer: torch.optim.Optimizer, labels: torch.Tensor, predictions: torch.Tensor):
    # loss
    loss_value = bce_loss(labels, predictions)  

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    return loss_value


### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    predictions_binary = (y_pred > threshold).float()

    true_positive = torch.sum(predictions_binary * y)
    false_positive = torch.sum(predictions_binary * (1 - y))
    false_negative = torch.sum((1 - predictions_binary) * y)

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1

### 2.5 Train loop
def eval_run(model, loader, device="cpu"):
    model.eval()  
    true_labels, predicted_labels = [], []
    with torch.no_grad():  
        for batch in loader():
            premises, hypotheses = batch['premise'], batch['hypothesis']
            labels = torch.tensor(batch['label']).float().to(device)
            premises, hypotheses = convert_to_tensors(premises).to(device), convert_to_tensors(hypotheses).to(device)

            predictions = model(premises, hypotheses)
            true_labels += labels.tolist()
            predicted_labels += predictions.tolist()

    return torch.tensor(true_labels), torch.tensor(predicted_labels)

def train_loop(model, train_loader, valid_loader, optimizer, n_epochs=3, device="cpu"):
    validation_f1_scores = []

    for epoch in range(n_epochs):
        model.train()  
        for batch in train_loader():
            optimizer.zero_grad()  
            predictions = forward_pass(model, batch, device)
            labels = torch.tensor(batch['label'], dtype=torch.float).to(device)
            loss = bce_loss(labels, predictions)
            backward_pass(optimizer, labels, predictions)

        true_labels, predicted_labels = eval_run(model, valid_loader, device)
        epoch_f1_score = f1_score(true_labels, predicted_labels)
        validation_f1_scores.append(epoch_f1_score.item())
        print(f"Epoch {epoch + 1}, Validation F1 Score: {epoch_f1_score.item()}")

    return validation_f1_scores


### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()
        self.embedding = embedding
        self.ff_layer = nn.Linear(embedding.embedding_dim * 2, hidden_size)
        self.activation = nn.ReLU()
        self.layer_pred = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################


    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        premise_pool = max_pool(self.embedding(premise))
        hypothesis_pool = max_pool(self.embedding(hypothesis))

        ff_out = self.activation(self.ff_layer(torch.cat([premise_pool, hypothesis_pool], dim=1)))
        logits = self.layer_pred(ff_out)

        return self.sigmoid(logits).view(-1)



### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.embedding = embedding
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(embedding.embedding_dim * 2, hidden_size))
        for _ in range(1, num_layers):
            self.ff_layers.append(nn.Linear(hidden_size, hidden_size))
        self.layer_pred = nn.Linear(hidden_size, 1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        premise_embed = self.embedding(premise)
        hypothesis_embed = self.embedding(hypothesis)
        pooled_premise = max_pool(premise_embed)
        pooled_hypothesis = max_pool(hypothesis_embed)
        concatenated = torch.cat((pooled_premise, pooled_hypothesis), dim=1)
        for layer in self.ff_layers:
            concatenated = self.activation(layer(concatenated))
        logits = self.layer_pred(concatenated)
        output = self.sigmoid(logits)
        output = self.sigmoid(logits)
        output = output.squeeze(-1)  # Squeeze the last dimension if it's 1
        return output


if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=10000)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }

    # 1.1
    train_loader = "your code here"
    valid_loader = "your code here"

    # 1.2
    """batch = next(train_loader())"""
    y = "your code here"

    # 2.1
    embedding = "your code here"
    model = "your code here"

    # 2.2
    optimizer = "your code here"

    # 2.3
    y_pred = "your code here"
    loss = "your code here"

    # 2.4
    score = "your code here"

    # 2.5
    n_epochs = 2

    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.1
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.2
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"