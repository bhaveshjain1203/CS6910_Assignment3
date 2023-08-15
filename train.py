#import necessary libraries
import os
import wandb
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
import torch.nn.functional as Function
import argparse
F=Function

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Set the device type to CUDA if available, otherwise use CPU
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
  
Start_Symbol, End_Symbol, Unknown, Padding = 0, 1, 2, 3

class Vocabulary:
    def __init__(self):
        self.char2count = {}
        self.char2index = {}
        self.n_chars = 4
        self.index2char = {0: "{", 1: "}", 2: "?", 3: "."}


    def addWord(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.char2count[char] = 1
                self.n_chars += 1
            else:
                self.char2count[char] += 1

# Define a function to prepare the data
def prepareDataWithoutAttn(dir):
    # Read the CSV file into a DataFrame with columns "input" and "target"
    data = pd.read_csv(dir, sep=",", names=["input", "target"])

    # Find the maximum length of input and target sequences
    max_input_length = 0
    for txt in data["input"].to_list():
        max_input_length = max(max_input_length, len(txt))
    
    max_target_length = 0
    for txt in data["target"].to_list():
        max_target_length = max(max_target_length, len(txt))
    
    max_len=0
    if max_input_length > max_target_length:
        max_len = max_input_length
    else:
        max_len = max_target_length

    # Create Vocabulary objects for input and output languages
    input_lang = Vocabulary()
    output_lang = Vocabulary()

    # Create pairs of input and target sequences
    pairs = []
    input_list, target_list = data["input"].to_list(), data["target"].to_list()
    for i in range(len(input_list)):
        pairs.append([input_list[i], target_list[i]])

    # Add words to the respective vocabularies
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])

    return input_lang,output_lang,pairs,max_len

# Define a helper function to convert a word to a tensor
def helpTensorWithoutAttn(lang, word, max_length):
    index_list = []
    for char in word:
        if char in lang.char2index.keys():
            index_list.append(lang.char2index[char])
        else:
            index_list.append(Unknown)
    indexes = index_list
    indexes.append(End_Symbol)
    indexes.extend([Padding] * (max_length - len(indexes)))
    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

# Define a function to convert pairs of input and target sequences to tensors
def MakeTensorWithoutAttn(input_lang, output_lang, pairs, reach):
    res = []
    for pair in pairs:
        # Convert input and target sequences to tensors using the helpTensorWithoutAttn function
        input_variable = helpTensorWithoutAttn(input_lang, pair[0], reach)
        target_variable = helpTensorWithoutAttn(output_lang, pair[1], reach)
        res.append((input_variable, target_variable))
    return res

#Encoder Class
class EncoderRNNWithoutAttn(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers_encoder, cell_type, drop_out, bi_directional):
        super(EncoderRNNWithoutAttn, self).__init__()

        # Initialize the EncoderRNNWithoutAttn with the provided parameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers_encoder = num_layers_encoder
        self.cell_type = cell_type
        self.drop_out = drop_out
        self.bi_directional = bi_directional

        # Create an embedding layer
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)

        # Create the specified cell layer (RNN, GRU, or LSTM)
        cell_map = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        self.cell_layer = cell_map[self.cell_type](
            self.embedding_size,
            self.hidden_size,
            num_layers=self.num_layers_encoder,
            dropout=self.drop_out,
            bidirectional=self.bi_directional,
        )

    def forward(self, input, batch_size, hidden):
        # Apply dropout to the embedded input sequence
        embedded = self.dropout(self.embedding(input).view(1, batch_size, -1))

        # Pass the embedded input through the cell layer
        output, hidden = self.cell_layer(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, num_layers_enc):
        # Initialize the hidden state with zeros
        res = torch.zeros(num_layers_enc * 2 if self.bi_directional else num_layers_enc, batch_size, self.hidden_size)

        # Move the hidden state to the GPU if use_cuda is True, else return as is
        return res.cuda() if use_cuda else res

#Decoder class
class DecoderRNNWithoutAttn(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers_decoder, cell_type, drop_out, bi_directional, output_size):
        super(DecoderRNNWithoutAttn, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers_decoder = num_layers_decoder
        self.cell_type = cell_type
        self.drop_out = drop_out
        self.bi_directional = bi_directional

        # Create an embedding layer
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)

        # Create the specified cell layer (RNN, GRU, or LSTM)
        cell_map = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        self.cell_layer = cell_map[self.cell_type](
            self.embedding_size,
            self.hidden_size,
            num_layers=self.num_layers_decoder,
            dropout=self.drop_out,
            bidirectional=self.bi_directional,
        )

        # Linear layer for output
        self.out = nn.Linear(
            self.hidden_size * 2 if self.bi_directional else self.hidden_size,
            output_size,
        )

        # Softmax activation
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        # Apply dropout to the embedded input sequence and pass it through the cell layer
        output = Function.relu(self.dropout(self.embedding(input).view(1, batch_size, -1)))
        output, hidden = self.cell_layer(output, hidden)

        # Apply softmax activation to the output
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Function to calculate loss (if is_training then training loss else validation loss)
def calc_lossWithoutAttn(encoder, decoder, input_tensor, target_tensor, batch_size, encoder_optimizer, decoder_optimizer, criterion, cell_type, num_layers_enc, max_length, is_training, teacher_forcing_ratio=0.5):
    # Initialize the encoder hidden state
    output_hidden = encoder.initHidden(batch_size, num_layers_enc)

    # Check if LSTM and initialize cell state
    if cell_type == "LSTM":
        encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
        output_hidden = (output_hidden, encoder_cell_state)

    # Zero the gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Initialize loss
    loss = 0

    # Encoder forward pass
    for ei in range(input_tensor.size(0)):
        output_hidden = encoder(input_tensor[ei], batch_size, output_hidden)[1]

    # Initialize decoder input
    decoder_input = torch.LongTensor([Start_Symbol] * batch_size)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Determine if using teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Loop over target sequence
    if is_training:
        # Training phase
        for di in range(target_tensor.size(0)):
            decoder_output, output_hidden = decoder(decoder_input, batch_size, output_hidden)
            decoder_input = target_tensor[di] if use_teacher_forcing else decoder_output.argmax(dim=1)
            loss = criterion(decoder_output, target_tensor[di]) + loss
    else:
        # Validation phase
        with torch.no_grad():
            for di in range(target_tensor.size(0)):
                decoder_output, output_hidden = decoder(decoder_input, batch_size, output_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = decoder_output.argmax(dim=1)

    # Backpropagation and optimization in training phase
    if is_training:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    # Return the average loss per target length
    return loss.item() / target_tensor.size(0)


# Calculate the accuracyWithoutAttn of the Seq2SeqWithoutAttn model
def accuracyWithoutAttn(encoder, decoder, loader, batch_size, criterion, cell_type, num_layers_enc, max_length, output_lang):
    with torch.no_grad():
        total = 0
        correct = 0

        for batch_x, batch_y in loader:
            # Initialize encoder hidden state
            encoder_hidden = encoder.initHidden(batch_size, num_layers_enc)

            input_variable = Variable(batch_x.transpose(0, 1))
            target_variable = Variable(batch_y.transpose(0, 1))

            # Check if LSTM and initialize cell state
            if cell_type == "LSTM":
                encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
                encoder_hidden = (encoder_hidden, encoder_cell_state)

            output = torch.LongTensor(target_variable.size()[0], batch_size)

            # Encoder forward pass
            for ei in range(input_variable.size()[0]):
                encoder_hidden = encoder(input_variable[ei], batch_size, encoder_hidden)[1]

            decoder_input = Variable(torch.LongTensor([Start_Symbol] * batch_size))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            # Decoder forward pass
            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
                topi = decoder_output.data.topk(1)[1]
                output[di] = torch.cat(tuple(topi))
                decoder_input = torch.cat(tuple(topi))

            output = output.transpose(0, 1)

            # Calculate accuracyWithoutAttn
            for di in range(output.size()[0]):
                ignore = [Start_Symbol, End_Symbol, Padding]
                sent = [output_lang.index2char[letter.item()] for letter in output[di] if letter not in ignore]
                y = [output_lang.index2char[letter.item()] for letter in batch_y[di] if letter not in ignore]
                if sent == y:
                    correct += 1
                total += 1

    return (correct / total) * 100

# Train and evaluate the Seq2SeqWithoutAttn model
def seq2seqWithoutAttn(encoder, decoder, train_loader, val_loader, test_loader, lr, optimizer, epochs, max_length_word, num_layers_enc, output_lang, wandb_project, wandb_entity):
    max_length = max_length_word - 1
    # Define the optimizer and criterion
    encoder_optimizer = optim.NAdam(encoder.parameters(), lr=lr) if optimizer == "nadam" else optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.NAdam(decoder.parameters(), lr=lr) if optimizer == "nadam" else optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity
    )
	

    run_name = 'hs_'+str(hidden_size)+'_bs_'+str(batch_size)+'_ct_'+cell_type+'_es_'+str(embedding_size)+'_do_'+str(drop_out)+'_nle_'+str(num_layers_enc)+'_nld_'+str(num_layers_decoder)+'_lr_'+str(learning_rate)+'_bd_'+str(bi_directional)

    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0

        # Training phase
        for batch_x, batch_y in train_loader:
            batch_x = Variable(batch_x.transpose(0, 1))
            batch_y = Variable(batch_y.transpose(0, 1))
            # Calculate the training loss
            loss = calc_lossWithoutAttn(encoder, decoder, batch_x, batch_y, batch_size, encoder_optimizer, decoder_optimizer, criterion, cell_type, num_layers_enc, max_length, is_training=True)
            train_loss_total += loss
            
        train_loss_avg = train_loss_total / len(train_loader)
        print(f"Epoch: {epoch} | Train Loss: {train_loss_avg:.4f} |", end="")

        # Validation phase
        for batch_x, batch_y in val_loader:
            batch_x = Variable(batch_x.transpose(0, 1))
            batch_y = Variable(batch_y.transpose(0, 1))
            # Calculate the validation loss
            loss = calc_lossWithoutAttn(encoder, decoder, batch_x, batch_y, batch_size, encoder_optimizer, decoder_optimizer, criterion, cell_type, num_layers_enc, max_length, is_training=False)
            val_loss_total += loss

        val_loss_avg = val_loss_total / len(val_loader)
        print(f"Val Loss: {val_loss_avg:.4f} |", end="")

        # Calculate validation accuracyWithoutAttn
        val_acc = accuracyWithoutAttn(encoder, decoder, val_loader, batch_size, criterion, cell_type, num_layers_enc, max_length, output_lang)
        val_acc /= 100
        print(f"Val Accuracy: {val_acc:.4%}")
        wandb.log({"validation_accuracy": val_acc, "training_loss": train_loss_avg, "Train_loss": train_loss_avg, 'epoch': epoch})
    wandb.run.name = run_name
    wandb.run.save()
    wandb.run.finish()

def prepareData(dir):

    input_lang = Vocabulary()
    output_lang = Vocabulary()
    # Read the CSV file into a DataFrame with columns "input" and "target"
    
    data = pd.read_csv(dir, sep=",", names=["input", "target"])

    input_list = data["input"].to_list()
    target_list = data["target"].to_list()
    # Find the maximum length of input and target sequences
    max_target_length = max([len(txt) for txt in data["target"].to_list()])

    pairs = []
    for i in range(len(target_list)):
        pairs.append([input_list[i], target_list[i]])

    max_input_length = max([len(txt) for txt in data["input"].to_list()])
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])

    prepared_data = {
        "input_lang": input_lang,
        "output_lang": output_lang,
        "pairs": pairs,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
    }

    return prepared_data

def helpindex(lang, word):
    l=[]
    for i in range(len(word)):
        if word[i] not in lang.char2index.keys():
            l.append(Unknown)
        else:
            l.append(lang.char2index[word[i]])
    return l

def helpTensor(lang, word, max_length):
    indexes = helpindex(lang, word)
    indexes.append(End_Symbol)
    indexes.extend([Padding] * (max_length - len(indexes)))
    result = torch.LongTensor(indexes)
    if use_cuda==False:
        return result
    else:
        return result.cuda()

def MakeTensor(input_lang, output_lang, pairs, max_length):
    res = []
    for pair in pairs:
        input_variable = helpTensor(input_lang, pair[0], max_length)
        target_variable = helpTensor(output_lang, pair[1], max_length)
        res.append((input_variable, target_variable))
    return res


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size,num_layers_encoder,cell_type,drop_out,bi_directional):
        super(EncoderRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers_encoder = num_layers_encoder
        self.cell_type = cell_type
        self.drop_out = drop_out
        self.bi_directional = bi_directional

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)

        cell_map = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        self.cell_layer = cell_map[self.cell_type](
            self.embedding_size,
            self.hidden_size,
            num_layers=self.num_layers_encoder,
            dropout=self.drop_out,
            bidirectional=self.bi_directional,
        )

    def forward(self, input, batch_size, hidden):
        embedded = self.dropout(self.embedding(input).view(1, batch_size, -1))
        output, hidden = self.cell_layer(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, num_layers_enc):
        res = torch.zeros(
            num_layers_enc * 2 if self.bi_directional else num_layers_enc,
            batch_size,
            self.hidden_size,
        )
        if use_cuda== False:
            return res
        else:
            return res.cuda()



class DecoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size,
        cell_type,
        num_layers_decoder,
        drop_out,
        max_length_word,
        output_size,
    ):

        super(DecoderAttention, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.num_layers_decoder = num_layers_decoder
        self.drop_out = drop_out
        self.max_length_word = max_length_word

        self.embedding = nn.Embedding(output_size, embedding_dim=self.embedding_size)
        self.attention_layer = nn.Linear(
            self.embedding_size + self.hidden_size, self.max_length_word
        )
        self.attention_combine = nn.Linear(
            self.embedding_size + self.hidden_size, self.embedding_size
        )
        self.dropout = nn.Dropout(self.drop_out)

        self.cell_layer = None
        cell_map = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}

        if self.cell_type in cell_map:
            self.cell_layer = cell_map[self.cell_type](
                self.embedding_size,
                self.hidden_size,
                num_layers=self.num_layers_decoder,
                dropout=self.drop_out,
            )

        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, batch_size, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, batch_size, -1)

        attention_weights = None
        if self.cell_type == "LSTM":
            attention_weights = Function.softmax(
                self.attention_layer(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1
            )

        else:
            attention_weights = Function.softmax(
                self.attention_layer(torch.cat((embedded[0], hidden[0]), 1)), dim=1
            )

        attention_applied = torch.bmm(
            attention_weights.view(batch_size, 1, self.max_length_word),
            encoder_outputs,
        ).view(1, batch_size, -1)
        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = Function.relu(output)
        # if self.cell_type=RNN" :
        output, hidden = self.cell_layer(output, hidden)
        output = Function.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attention_weights



def train_and_val_with_attn(
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    input_tensor,
    target_tensor,
    criterion,
    batch_size,
    cell_type,
    num_layers_enc,
    max_length,is_training,
    teacher_forcing_ratio=0.5,
):

    encoder_hidden = encoder.initHidden(batch_size, num_layers_enc)

    if cell_type == "LSTM":
        encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
        encoder_hidden = (encoder_hidden, encoder_cell_state)

    if is_training:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], batch_size, encoder_hidden
        )
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([Start_Symbol] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    if is_training:
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing == False:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    batch_size,
                    decoder_hidden,
                    encoder_outputs.reshape(batch_size, max_length, encoder.hidden_size),
                )
                #2 for loop ko bhar dal de
                topv, topi = decoder_output.data.topk(1)
                decoder_input = torch.cat(tuple(topi))

                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_tensor[di])
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    batch_size,
                    decoder_hidden,
                    encoder_outputs.reshape(batch_size, max_length, encoder.hidden_size),
                )
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
            

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    else :
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input,
                batch_size,
                decoder_hidden,
                encoder_outputs.reshape(batch_size, max_length, encoder.hidden_size),
            )
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_tensor[di])


    return loss.item() / target_length

def accuracy_with_attention(
    encoder,
    decoder,
    loader,
    batch_size,
    num_layers_enc,
    cell_type,
    output_lang,
    criterion,
    max_length,
):

    with torch.no_grad():

        # batch_size = configuration["batch_size"]
        total = 0
        correct = 0

        for batch_x, batch_y in loader:

            encoder_hidden = encoder.initHidden(batch_size, num_layers_enc)

            input_variable = Variable(batch_x.transpose(0, 1))
            target_variable = Variable(batch_y.transpose(0, 1))

            if cell_type == "LSTM":
                encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
                encoder_hidden = (encoder_hidden, encoder_cell_state)

            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            output = torch.LongTensor(target_length, batch_size)

            encoder_outputs = Variable(
                torch.zeros(max_length, batch_size, encoder.hidden_size)
            )
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_variable[ei], batch_size, encoder_hidden
                )
                encoder_outputs[ei] = encoder_output[0]

            decoder_input = Variable(torch.LongTensor([Start_Symbol] * batch_size))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    batch_size,
                    decoder_hidden,
                    encoder_outputs.reshape(
                        batch_size, max_length, encoder.hidden_size
                    ),
                )
                topv, topi = decoder_output.data.topk(1)
                decoder_input = torch.cat(tuple(topi))
                output[di] = torch.cat(tuple(topi))

            output = output.transpose(0, 1)
            for di in range(output.size()[0]):
                ignore = [Start_Symbol, End_Symbol, Padding]
                sent = [
                    output_lang.index2char[letter.item()]
                    for letter in output[di]
                    if letter not in ignore
                ]
                y = [
                    output_lang.index2char[letter.item()]
                    for letter in batch_y[di]
                    if letter not in ignore
                ]
                if sent == y:
                    correct += 1
                total += 1

    return (correct / total) * 100


def cal_val_loss_with_attn(
    encoder,
    decoder,
    input_tensor,
    target_tensor,
    batch_size,
    criterion,
    cell_type,
    num_layers_enc,
    max_length,
):

    with torch.no_grad():

        encoder_hidden = encoder.initHidden(batch_size, num_layers_enc)

        if cell_type == "LSTM":
            encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
            encoder_hidden = (encoder_hidden, encoder_cell_state)

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        encoder_outputs = Variable(
            torch.zeros(max_length, batch_size, encoder.hidden_size)
        )
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], batch_size, encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([Start_Symbol] * batch_size))
        if use_cuda== True:
            decoder_input = decoder_input.cuda()  
        else :
            decoder_input = decoder_input

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input,
                batch_size,
                decoder_hidden,
                encoder_outputs.reshape(batch_size, max_length, encoder.hidden_size),
            )
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            if use_cuda== True:
                decoder_input = decoder_input.cuda()  
            else :
                decoder_input = decoder_input
            loss += criterion(decoder_output, target_tensor[di])

    return loss.item() / target_length


def Attention_seq2seq(
    encoder,
    decoder,
    train_loader,
    val_loader,
    test_loader,
    learning_rate,
    optimizer,
    epochs,
    max_length_word,
    attention,
    num_layers_enc,
    output_lang,
):
    max_length = max_length_word - 1
    encoder_optimizer = (
        optim.NAdam(encoder.parameters(), lr=learning_rate)
        if optimizer == "nadam"
        else optim.Adam(encoder.parameters(), lr=learning_rate)
    )
    decoder_optimizer = (
        optim.NAdam(decoder.parameters(), lr=learning_rate)
        if optimizer == "nadam"
        else optim.Adam(decoder.parameters(), lr=learning_rate)
    )
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        train_loss_total, val_loss_total  =0, 0
        
        for batchx, batchy in train_loader:
            batchx = Variable(batchx.transpose(0, 1))
            batchy = Variable(batchy.transpose(0, 1))
            loss = train_and_val_with_attn(
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                batchx,
                batchy,
                criterion,
                batch_size,
                cell_type,
                num_layers_enc,
                max_length + 1,
                True, #is_training
            )
            train_loss_total += loss

        train_loss_avg = train_loss_total / len(train_loader)
        print(f"Epoch: {epoch} | Train Loss: {train_loss_avg:.4f} | ", end="")

        for batchx, batchy in val_loader:
            batchx = Variable(batchx.transpose(0, 1))
            batchy = Variable(batchy.transpose(0, 1))
            loss = train_and_val_with_attn(
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                batchx,
                batchy,
                criterion,
                batch_size,
                cell_type,
                num_layers_enc,
                max_length + 1,
                False,#is_training=
            )
            val_loss_total += loss

        val_loss_avg = val_loss_total / len(val_loader)
        print(f"Val Loss: {val_loss_avg:.4f} | ", end="")
        val_acc = accuracy_with_attention(
            encoder,
            decoder,
            val_loader,
            batch_size,
            num_layers_enc,
            cell_type,
            output_lang,
            criterion,
            max_length + 1,
        )
        val_acc = val_acc / 100
        print(f"Val Accuracy: {val_acc:.4%}")
        if epochs-1==epoch:
            test_acc = accuracy_with_attention(
            encoder,
            decoder,
            test_loader,
            batch_size,
            num_layers_enc,
            cell_type,
            output_lang,
            criterion,
            max_length + 1,
        )
            test_acc = test_acc / 100
            print(f"Test Accuracy: {test_acc:.4%}")

def to_dict(input_lang,output_lang,pairs,max_len):
    dictionary = {
        "input_lang": input_lang,
        "output_lang": output_lang,
        "pairs": pairs,
        "max_len": max_len
    }
    return dictionary

def RunModel(attention_flag, wandb_project, wandb_entity):
    teacher_forcing_ratio = 0.5
    optimizer = "Nadam"
    learning_rate = 0.001
    train_path = "aksharantar_sampled/hin/hin_train.csv"
    validation_path = "aksharantar_sampled/hin/hin_valid.csv"
    test_path = "aksharantar_sampled/hin/hin_test.csv"


    if attention_flag:
        train_prepared_data = prepareData(train_path)
        input_langs, output_langs, pairs = (
            train_prepared_data["input_lang"],
            train_prepared_data["output_lang"],
            train_prepared_data["pairs"],
        )
        print("train:sample:", random.choice(pairs))
        print(f"Number of training examples: {len(pairs)}")

        max_input_length, max_target_length = (
            train_prepared_data["max_input_length"],
            train_prepared_data["max_target_length"],
        )

        # validation
        val_prepared_data = prepareData(validation_path)
        val_pairs = val_prepared_data["pairs"]
        print("validation:sample:", random.choice(val_pairs))
        print(f"Number of validation examples: {len(val_pairs)}")
        # Test
        max_input_length_val, max_target_length_val = (
            val_prepared_data["max_input_length"],
            val_prepared_data["max_target_length"],
        )
        test_prepared_data = prepareData(validation_path)
        test_pairs = test_prepared_data["pairs"]
        print("Test:sample:", random.choice(test_pairs))
        print(f"Number of Test examples: {len(test_pairs)}")

        max_input_length_test, max_target_length_test = (
            test_prepared_data["max_input_length"],
            test_prepared_data["max_target_length"],
        )
        max_len_all = (
            max(
                max_input_length,
                max_target_length,
                max_input_length_val,
                max_target_length_val,
                max_input_length_test,
                max_target_length_test,
            )
            + 1
        )

        max_len = max(max_input_length, max_target_length) + 3
        print(max_len)

        pairs = MakeTensor(input_langs, output_langs, pairs, max_len)
        val_pairs = MakeTensor(input_langs, output_langs, val_pairs, max_len)
        test_pairs = MakeTensor(input_langs, output_langs, test_pairs, max_len)

        train_loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_pairs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_pairs, batch_size=batch_size, shuffle=True)

        encoder1 = EncoderRNN(
            input_langs.n_chars,
            embedding_size,
            hidden_size,
            num_layers_encoder,
            cell_type,
            drop_out,
            bi_directional,
        )
        attndecoder1 = DecoderAttention(
            hidden_size,
            embedding_size,
            cell_type,
            num_layers_decoder,
            drop_out,
            max_len,
            output_langs.n_chars,
        )
        if use_cuda== True:
            encoder1 = encoder1.cuda()
            attndecoder1 = attndecoder1.cuda()
        print("with attention")
        attention = True
        Attention_seq2seq(
            encoder1,
            attndecoder1,
            train_loader,
            val_loader,
            test_loader,
            learning_rate,
            optimizer,
            epochs,
            max_len,
            attention,
            num_layers_encoder,
            output_langs,
        )
    else:
        # Prepare training data
        _input_lang,_output_lang,_pairs,_max_len = prepareDataWithoutAttn(train_path)
        train_prepared_data = to_dict(_input_lang,_output_lang,_pairs,_max_len)
        input_langs, output_langs, pairs = train_prepared_data["input_lang"], train_prepared_data["output_lang"], train_prepared_data["pairs"]
        print("train:sample:", random.choice(pairs))
        print(f"Number of training examples: {len(pairs)}")
        max_len = train_prepared_data["max_len"]

        # Prepare validation data
        _input_lang,_output_lang,_pairs,_max_len = prepareDataWithoutAttn(validation_path)
        val_prepared_data = to_dict(_input_lang,_output_lang,_pairs,_max_len)
        val_pairs = val_prepared_data["pairs"]
        print("validation:sample:", random.choice(val_pairs))
        print(f"Number of validation examples: {len(val_pairs)}")
        max_len_val = val_prepared_data["max_len"]

        # Prepare test data
        _input_lang,_output_lang,_pairs,_max_len = prepareDataWithoutAttn(test_path)
        test_prepared_data = to_dict(_input_lang,_output_lang,_pairs,_max_len)
        test_pairs = test_prepared_data["pairs"]
        print("Test:sample:", random.choice(test_pairs))
        print(f"Number of Test examples: {len(test_pairs)}")

        max_len_test = test_prepared_data["max_len"]
        max_len = max(max_len, max_len_val, max_len_test) + 4
        print(max_len)

        # Convert data to tensors and create data loaders
        pairs = MakeTensorWithoutAttn(input_langs, output_langs, pairs, max_len)
        val_pairs = MakeTensorWithoutAttn(input_langs, output_langs, val_pairs, max_len)
        test_pairs = MakeTensorWithoutAttn(input_langs, output_langs, test_pairs, max_len)

        train_loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_pairs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_pairs, batch_size=batch_size, shuffle=True)

        # Create the encoder and decoder models
        encoder1 = EncoderRNNWithoutAttn(input_langs.n_chars, embedding_size, hidden_size, num_layers_encoder, cell_type, drop_out, bi_directional)
        decoder1 = DecoderRNNWithoutAttn(embedding_size, hidden_size, num_layers_encoder, cell_type, drop_out, bi_directional, output_langs.n_chars)

        if use_cuda:
            encoder1, decoder1 = encoder1.cuda(), decoder1.cuda()

        print("vanilla seq2seqWithoutAttn")
        # Train and evaluate the Seq2SeqWithoutAttn model
        seq2seqWithoutAttn(encoder1, decoder1, train_loader, val_loader, test_loader, learning_rate, optimizer, epochs, max_len, num_layers_encoder, output_langs, wandb_project, wandb_entity)

parser = argparse.ArgumentParser(description='Execute the model and calculate the accuracy')
parser.add_argument('-wp', '--wandb_project', type=str, help='wandb project name', default='cs6910_assignment3')
parser.add_argument('-we', '--wandb_entity', type=str, help='wandb entity', default='cs22m029')
parser.add_argument('-es', '--emb_size', type=int, help='embedding size', default=256)
parser.add_argument('-nle', '--num_layers_encoder', type=int, help='number of layers in encoder', default=2)
parser.add_argument('-nld', '--num_layers_decoder', type=int, help='number of layers in decoder', default=2)
parser.add_argument('-hs', '--hidden_size', type=int, help='hidden size', default=256)
parser.add_argument('-bs', '--batch_size', type=int, help='batch size', default=32)
parser.add_argument('-ep', '--epochs', type=int, help='epochs', default=5)
parser.add_argument('-ct', '--cell_type', type=str, help='Cell type', default="LSTM")
parser.add_argument('-bdir', '--bidirectional', type=bool, help='bidirectional', default=False)
parser.add_argument('-drop', '--dropout', type=float, help='dropout', default=0.2)

params = parser.parse_args()
hidden_size = params.hidden_size
input_lang = "eng"
target_lang = "hin"
cell_type = params.cell_type
num_layers_encoder = params.num_layers_encoder
num_layers_decoder = params.num_layers_decoder
drop_out = params.dropout
epochs = params.epochs
embedding_size = params.emb_size
bi_directional = params.bidirectional
batch_size = params.batch_size
attention_flag=False
learning_rate=0.001
if __name__ == '__main__':
    RunModel(attention_flag, params.wandb_project, params.wandb_entity)


