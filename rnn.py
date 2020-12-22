import torch
from torch import nn
import numpy as np


from cipher import computexorstream
import dataloader



# Create alphabet
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(alphabet))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}





# =========================================== HYPERPARAMS =====================================================

# number of samples
batch_size = 10000

#length of samples
n = 30
known_len = 29
prediction_len = n - known_len

dict_size = len(alphabet)



n_epochs = 500
lr=0.01





text = dataloader.generate_text('inputs/war-peace.txt', 'inputs/war-peace.txt', num_samples=batch_size, length=n, randomoffset=3000000)
print('loaded text')

# Creating lists that will hold our input and target sequences

text = ((a, b, ''.join(computexorstream(a,b))) for (a,b) in text)
print('computed cipher')

input_seq_c = []
target_seq_c = []
known_data_c = []



for sample in text:
    # print(sample)
    # 1 characters of cipher maps to 1 characters of message
    input_seq_c.append(([sample[2][-1]]))
    target_seq_c.append(([sample[0][-1]]))

    # n-1 characters of message, n-1 characters of key, n-1 characters of cipher
    known_data_c.append((sample[0][:-1], sample[1][:-1], sample[2][:-1]))

# print(f'input seq: {input_seq_c}')
# print(f'target seq: {target_seq_c}')
# print(f'known data: {known_data_c}')


print('stripped text')


input_seq = [[char2int[character] for character in input_seq] for input_seq in input_seq_c]
target_seq = [[char2int[character] for character in target_seq] for target_seq in target_seq_c]
known_data = [[[char2int[character] for character in seq] for seq in known_data] for known_data in known_data_c]


# print(f'input seq: {input_seq}')
# print(f'target seq: {target_seq}')
# print(f'known data: {known_data}')



print('Converted text')



def one_hot_encode(sequence, dict_size, seq_len, batch_size=1):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for j in range(seq_len):
            features[i, j, sequence[i][j]] = 1
    return features

def one_hot_encode_known(sequence, dict_size, seq_len, batch_size=1):
    inputs = 3
    features = np.zeros((batch_size, seq_len, inputs*dict_size), dtype=np.float32)

    for i in range(batch_size):
        for k in range(0, inputs):
            for j in range(seq_len):
                features[i, j, k*dict_size + sequence[i][k][j]] = 1
    return features








# Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq, dict_size, prediction_len, batch_size)
# target_seq = one_hot_encode(target_seq, dict_size, prediction_len, batch_size)

known_data = one_hot_encode_known(known_data, dict_size, n-prediction_len, batch_size)

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq[:batch_size])
known_data = torch.from_numpy(known_data)





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
        self.rnn_train = nn.RNN(3*input_size, hidden_dim, n_layers, batch_first=True)
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
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=2)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)



# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    input_seq.to(device)
    known_data.to(device)
    output = model(input_seq, known_data)
    target_seq = target_seq.view(-1).long()
    loss = criterion(output, target_seq)
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))







# This function takes in the model and character as arguments and returns the next character prediction
def predict(model, character, known_data):
    # One-hot encoding our input to fit into the model

    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1])
    character = torch.from_numpy(character)
    character.to(device)


    known_data[0] = ([[char2int[character] for character in seq] for seq in known_data[0]])
    known_data = one_hot_encode_known(known_data, dict_size, n-prediction_len)
    known_data = torch.from_numpy(known_data)
    known_data.to(device)



    out = model(character, known_data)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind]









    # This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, character, known_data):
    model.eval() # eval mode

    char = predict(model, character, known_data)
    return char



incorrect = 0
for i in range(0, batch_size):
    if target_seq_c[i][0] != sample(model, input_seq_c[i], [known_data_c[i]]):
        incorrect += 1

print ("Training validation accuracy:")
print (1 - incorrect / batch_size)



# must do unseen validation as well




