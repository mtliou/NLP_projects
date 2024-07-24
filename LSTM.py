'''
implementation of the following:
1. Character RNN 
2. Character LSTM 
3. Shallow BiLSTM for SNLI dataset 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

##### PROVIDED CODE #####

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

# modify build_word_counts for SNLI
# so that it takes into account batch['premise'] and batch['hypothesis']
def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in batch:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader:
    def __init__(self, filepath, seq_len, examples_per_epoch):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = file.read()

        # Create a list of unique characters and ensure it's sorted for consistency
        unique_chars = sorted(list(set(data)))
        self.unique_chars = list(set(data))

        # Generate character mappings
        self.mappings = self.generate_char_mappings(self.unique_chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}

        # Convert the dataset into indices
        self.data_as_indices = self.convert_seq_to_indices(data)

        # Pack dataset into a tensor and move to GPU if available
        self.data_as_indices = torch.tensor(self.data_as_indices).to(device)
        
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch
        self.vocab_size = len(self.unique_chars)


    def __len__(self):
        return self.examples_per_epoch

    def generate_char_mappings(self, uq_chars):
        char_to_idx = {char: idx for idx, char in enumerate(uq_chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        return {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}

    def convert_seq_to_indices(self, seq):
        return [self.mappings['char_to_idx'][char] for char in seq]

    def convert_indices_to_seq(self, indices):
        return [self.mappings['idx_to_char'][idx] for idx in indices]

    def get_example(self):
        for _ in range(self.examples_per_epoch):
            start_index = random.randint(0, len(self.data_as_indices) - self.seq_len - 1)
            end_index = start_index + self.seq_len + 1
            seq_slice = self.data_as_indices[start_index:end_index]
            in_seq = seq_slice[:-1]
            target_seq = seq_slice[1:]
            yield in_seq, target_seq

class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size

        # Embedding layer maps character indices to learned embedding representations
        self.embedding_layer = nn.Embedding(n_chars, embedding_size)
        self.wax = nn.Linear(embedding_size, hidden_size, bias=False)
        self.waa = nn.Linear(hidden_size, hidden_size, bias=True)  
        self.wya = nn.Linear(hidden_size, n_chars, bias=True)  
        
    def rnn_cell(self, x, h_prev):
        h_next = torch.tanh(self.wax(x) + self.waa(h_prev)) 
        # Compute the output from the hidden state, no activation here as we'll use CrossEntropyLoss later
        y = self.wya(h_next)
        return y, h_next

    def forward(self, input_seq, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(device)

        # Pass the input sequence through the embedding layer
        embedded = self.embedding_layer(input_seq)

        out = []
        for i in range(len(input_seq)):  # Iterate over sequence length
            output, hidden = self.rnn_cell(embedded[i], hidden)
            out.append(output)

        # Stack the outputs along a new dimension to match the expected output shape
        out = torch.stack(out, dim=0)

        # Ensure the outputs tensor is of the correct shape, e.g., (batch_size, seq_len, vocab_size)
        return out, hidden

    def get_loss_function(self):
        # CrossEntropyLoss includes softmax, suitable for classification tasks with N classes
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        generated_seq = [starting_char]
        hidden = None
        current_char = torch.tensor([starting_char], dtype=torch.long).to(device)

        for _ in range(seq_len):
            current_char = current_char.unsqueeze(0)
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_size).to(device)

            output, hidden = self.forward(current_char, hidden)
            output = output.squeeze()  # Remove unnecessary dimensions

            logits = output / temp
            # Ensure logits is 2D: [1, vocab_size] before applying softmax
            if logits.dim() > 2:
                logits = logits.view(1, -1)  # Reshape to [1, vocab_size]

            probs = F.softmax(logits, dim=-1)  # Apply softmax along the vocab dimension

            #print(f"Probs shape after softmax: {probs.shape}")  

            next_char_dist = Categorical(probs)
            next_char = next_char_dist.sample()

            generated_seq.append(next_char.item())
            current_char = next_char

        return generated_seq


class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size

        
        # Embedding layer for character indices
        self.embedding_layer = nn.Embedding(n_chars, embedding_size)

        # LSTM layers for gates, initialized to accept the concatenated input of hidden state and input embedding
        self.forget_gate = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.input_gate = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.cell_state_layer = nn.Linear(hidden_size + embedding_size, hidden_size)

        # Layer for the final output from the hidden state
        self.fc_output = nn.Linear(hidden_size, n_chars)

    def lstm_cell(self, i, h, c):
        # Ensure i and h are properly shaped (potentially removing any unnecessary batch dimensions)
        i = i.view(-1)  # Flatten i to 1D
        h = h.view(-1)  # Flatten h to 1D

        # Concatenate along dimension 0, as required by the tests
        combined = torch.cat((i, h), dim=0).unsqueeze(0)  # Add a singleton batch dimension

        # Proceed with LSTM operations
        f = torch.sigmoid(self.forget_gate(combined))
        i_gate = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_state_layer(combined))

        c_new = f * c + i_gate * c_tilde
        h_new = o * torch.tanh(c_new)

        o = self.fc_output(h_new.squeeze(0))  # Remove the singleton batch dimension for the output

        return o, h_new.squeeze(0), c_new.squeeze(0)  # Ensure h_new and c_new are also correctly shaped


    def forward(self, input_seq, hidden=None, cell=None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(device)

        if cell is None:
            cell = torch.zeros(self.hidden_size).to(device)

        # Pass the input sequence through the embedding layer
        embedded = self.embedding_layer(input_seq)

        outputs = []
        for i in range(len(input_seq)):
            # Apply the lstm_cell to get the output and update the hidden and cell states
            output, hidden, cell = self.lstm_cell(embedded[i], hidden, cell)
            outputs.append(output)

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs, dim=0)

        return outputs, hidden, cell

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        generated_seq = [starting_char]
        hidden = None
        cell = None
        current_char = torch.tensor([starting_char], dtype=torch.long).to(device)

        for _ in range(seq_len):
            # Ensure current_char is correctly shaped with batch dimension
            current_char = current_char.unsqueeze(0)

            if hidden is None:
                hidden = torch.zeros(1, self.hidden_size).to(device)
            if cell is None:
                cell = torch.zeros(1, self.hidden_size).to(device)

            output, hidden, cell = self.forward(current_char, hidden, cell)

            # Apply temperature scaling
            logits = output.squeeze(0) / temp  # Remove batch dimension before applying temperature

            # Apply top-k and top-p filtering
            if top_k is not None:
                logits = top_k_filtering(logits, top_k)
            if top_p is not None:
                logits = top_p_filtering(logits, top_p)

            # Sample next character
            probs = F.softmax(logits, dim=-1)
            next_char_dist = Categorical(probs)
            next_char = next_char_dist.sample()

            generated_seq.append(next_char.item())
            current_char = next_char

        return generated_seq

    
def top_k_filtering(logits, top_k=40):
    filtered_logits = logits.clone()
    
    top_k_values, _ = torch.topk(filtered_logits, top_k)
    
    min_value = top_k_values[:, -1].unsqueeze(1)
    
    filtered_logits[filtered_logits < min_value] = float('-inf')
    
    return filtered_logits


def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits


import torch
import torch.nn as nn
import torch.optim as optim
'''
def train(model, dataset, lr, out_seq_len, num_epochs):
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            in_seq, out_seq = in_seq.to(device), out_seq.to(device)
            optimizer.zero_grad()
            output, _ = model(in_seq)
            loss = criterion(output.transpose(1, 2), out_seq)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += 1

            if n % epoch_size == 0:
                print(f"Epoch {epoch}. Running loss so far: {(running_loss / n):.8f}")

        # Sample from the model after the epoch
        print("\n-------------SAMPLE FROM MODEL-------------")
        with torch.no_grad():
            start_char = random.choice(range(len(chars)))  # Randomly choose a start character
            generated_seq = model.sample_sequence(start_char, out_seq_len)
            generated_text = ''.join(idx_to_char[idx] for idx in generated_seq)
            print(generated_text)
            print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    return model  # Optionally return the trained model
'''
def train(model, dataset, lr, out_seq_len, num_epochs):
    model.to(device)
    optimizer = model.get_optimizer(lr)
    loss_function = model.get_loss_function()

    for epoch in range(num_epochs):
        total_loss = 0
        for in_seq, out_seq in dataset.get_example():
            in_seq, out_seq = in_seq.to(device), out_seq.to(device)

            optimizer.zero_grad()

            # Check if the model is an instance of CharLSTM
            if isinstance(model, CharLSTM):
                output, hidden, cell = model(in_seq)  # Expect three outputs: sequence output, last hidden state, and last cell state
                # Detach the hidden and cell states to prevent backpropagating through the entire training history
                hidden = hidden.detach()
                cell = cell.detach()
            else:  # For CharRNN or other models that do not use cell states
                output, hidden = model(in_seq)  # Only expect two outputs: sequence output and last hidden state
                hidden = hidden.detach()  # Detach the hidden state

            loss = loss_function(output.view(-1, model.n_chars), out_seq.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Sample from the model every few epochs
        if (epoch+1) % 10 == 0:
            print("\nSampled Text:")
            start_char = random.choice(list(dataset.char_to_idx.keys()))
            start_idx = dataset.char_to_idx[start_char]
            sampled_seq = model.sample_sequence(start_idx, out_seq_len)
            sampled_text = ''.join([dataset.idx_to_char[idx] for idx in sampled_seq])
            print(sampled_text)
            print("\n---------------------\n")



def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    data_path = "E:\steam\Total war II download\data\shakespeare.txt"

    # Initialize dataset
    dataset = CharSeqDataloader(data_path, seq_len, examples_per_epoch=1000)
    
    # Initialize model
    n_chars = len(dataset.unique_chars)
    model = CharRNN(n_chars, embedding_size, hidden_size)

    train(model, dataset, lr=lr, out_seq_len=200, num_epochs=num_epochs)

    
def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10  # unused?
    out_seq_len = 200
    data_path = "E:\\steam\\Total war II download\\data\\shakespeare.txt" 

    # Initialize dataset
    dataset = CharSeqDataloader(data_path, seq_len, examples_per_epoch=1000)

    # Initialize model
    n_chars = len(dataset.unique_chars)
    model = CharLSTM(n_chars, embedding_size, hidden_size)

    # Train the model
    train(model, dataset, lr=lr, out_seq_len=out_seq_len, num_epochs=num_epochs)


import torch
from torch.nn.utils.rnn import pad_sequence
def fix_padding(batch_premises, batch_hypotheses):
    # Convert lists to tensors and reverse for backward LSTM
    tensor_premises = [torch.tensor(premise) for premise in batch_premises]
    tensor_hypotheses = [torch.tensor(hypothesis) for hypothesis in batch_hypotheses]
    
    reversed_premises = [torch.tensor(premise[::-1]) for premise in batch_premises]
    reversed_hypotheses = [torch.tensor(hypothesis[::-1]) for hypothesis in batch_hypotheses]
    
    # Pad sequences
    padded_premises = pad_sequence(tensor_premises, batch_first=True)
    padded_hypotheses = pad_sequence(tensor_hypotheses, batch_first=True)
    
    padded_reversed_premises = pad_sequence(reversed_premises, batch_first=True)
    padded_reversed_hypotheses = pad_sequence(reversed_hypotheses, batch_first=True)
    
    # Return the padded and reversed tensors
    return padded_premises, padded_hypotheses, padded_reversed_premises, padded_reversed_hypotheses

import numpy as np

def create_embedding_matrix(index_map, emb_dict, emb_dim):
    # Initialize the matrix with zeros
    embedding_matrix = np.zeros((len(index_map), emb_dim))

    # Fill the matrix with embeddings, if the word is found in the emb_dict
    for word, i in index_map.items():
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            # Words not found in emb_dict will remain all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

    return embedding_matrix

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax

import numpy as np

def evaluate(model, dataloader, index_map):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # Disable gradient tracking
        for batch in dataloader:
            premises, hypotheses, labels = batch['premise'], batch['hypothesis'], batch['label']
            
            # Tokenize and convert to indices
            tokenized_premises = tokenize(premises)
            tokenized_hypotheses = tokenize(hypotheses)
            indexed_premises = tokens_to_ix(tokenized_premises, index_map)
            indexed_hypotheses = tokens_to_ix(tokenized_hypotheses, index_map)
            
            # Pad sequences and convert to tensors
            padded_premises = pad_sequence([torch.tensor(p).to(device) for p in indexed_premises], batch_first=True)
            padded_hypotheses = pad_sequence([torch.tensor(h).to(device) for h in indexed_hypotheses], batch_first=True)
            labels = torch.tensor(labels).to(device)

            outputs = model(padded_premises, padded_hypotheses)
            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate accuracy
    accuracy = total_correct / total_samples
    return accuracy

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0) 
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        
        embedded_a = self.embedding_layer(a)
        embedded_b = self.embedding_layer(b)

        _, (final_state_a, _) = self.lstm(embedded_a)
        _, (final_state_b, _) = self.lstm(embedded_b)

       
        final_state_a = final_state_a[-1]  
        final_state_b = final_state_b[-1]  

        combined = torch.cat((final_state_a, final_state_b), dim=0) 

        intermediate_output = F.relu(self.int_layer(combined))  
        output_logits = self.out_layer(intermediate_output)  

        return output_logits


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        self.lstm_forward = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.int_layer = nn.Linear(hidden_dim * 4, hidden_dim)  
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        a_embedded = self.embedding(a)
        b_embedded = self.embedding(b)

        _, (h_n_forward_a, _) = self.lstm_forward(a_embedded)
        _, (h_n_forward_b, _) = self.lstm_forward(b_embedded)

        a_embedded_reversed = torch.flip(a_embedded, dims=[1])
        b_embedded_reversed = torch.flip(b_embedded, dims=[1])

        # Backward LSTM on the reversed input
        _, (h_n_backward_a, _) = self.lstm_backward(a_embedded_reversed)
        _, (h_n_backward_b, _) = self.lstm_backward(b_embedded_reversed)

        # Concatenate the final cell states in the specified order
        combined = torch.cat((h_n_forward_a[-1], h_n_backward_a[-1], 
                              h_n_forward_b[-1], h_n_backward_b[-1]), dim=1)

        # Apply the intermediate fully connected layer with ReLU activation
        intermediate_output = F.relu(self.fc_intermediate(combined))
        
        # Apply the output fully connected layer to get the final class logits
        output_logits = self.fc_output(intermediate_output)

        return output_logits



def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = "" # fill in your code

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    # training code

def run_snli_lstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

def run_snli_bilstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

if __name__ == '__main__':
    pass
    #run_char_rnn()
    run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
