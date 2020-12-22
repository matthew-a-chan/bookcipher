import torch
from torch import nn
import numpy as np




# Create alphabet
alphabet = set('abcdefghijklmnopqrstuvwxyz')

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(alphabet))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}



#create dataset here



# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

n = 9



for i in range(len(text)):
    # Remove last character for input sequence
    # n+1 characters of cipher, n characters of plain, n characters of key (book)
    input_seq.append((text[i][0], text[i][1][:-1] + ['a'], text[i][2][:-1] + ['a']))
    
    # n characters of plain, n characters of key (shifted one right)
    target_seq.append((['a'] + text[i][1][1:], ['a'] + text[i][2][1:]))

    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))




for i in range(len(text)):
    input_seq[i] = [[char2int[character] for character in seq] for seq in input_seq[i]]
    target_seq[i] = [[char2int[character] for character in seq] for seq in target_seq[i]]





dict_size = len(alphabet)
seq_len = n - 1
batch_size = 16


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, 3*dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            for j in range(0, 3):
                features[i, u, j*dict_size + sequence[i][j][u]] = 1
    return features


# Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

input_seq = torch.from_numpy(input_seq)
#target_seq = torch.Tensor(target_seq)
target_seq = torch.from_numpy(target_seq)




















# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")




class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        self.rnn_train = nn.RNN(3*input_ize, hidden_dim, n_layers, batch_first=True)
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, known_data):
        
        batch_size = x.size(0)

        # train hidden on first 3n entries, then use rnn for predictions

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Train hidden on first few known characters
        _, hidden = self.rnn_train(known_data, hidden)

        # alternatively, try taking the output from this sequence and shoving it through
        # a fc layer to intialize parameters of hidden

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    input_seq.to(device)
    known_data.to(device)
    output, hidden = model(input_seq, known_data)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))








# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, character, known_data):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)
    
    out, hidden = model(character, known_data)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden









    # This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start, known_data):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for i in range(size):
        char, h = predict(model, chars, known_data)
        chars.append(char)

    return ''.join(chars)






    sample(model, 15, 'good')




