import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features) #(batch_size, embed_size)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #If batch_first = False, default, (seq, batch, feature)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.init_hidden()
        self.init_weight()

    def init_hidden(self):
        """
        Initialize the hidden states of LSTM
        """
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


    def init_weight(self):
        """Initialize weights"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0.0)
    
    def forward(self, features, captions):
        #Do not include special end to the input caption
        embeded_captions = self.embedding(captions[:, :-1])
        embeded = torch.cat((features.unsqueeze(1), embeded_captions), 1)
        output, _ = self.lstm(embeded)
        output = self.linear(output)
        return output


    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, inputs, hidden)
            outputs = self.linear(outputs.squeeze(1))
            target_index = outpus.max(1)[1].item()
            res.append(target_index)
            inputs = self.embedding[target_index]
        return res

        