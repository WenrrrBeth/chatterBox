import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import json
import random
from tqdm import tqdm

# stemmer = LancasterStemmer()
stemmer = PorterStemmer()

class Params():
    def __init__(self):
        self.lr = 0.001
        self.load_weights = 1  # this is for training, 1-no training (chat starts), 0-train model
        self.batch = 8
        self.epoch = 1000
        self.jsfile = '../intents.json'
        self.igrnore_char = ['?', '!', '.', ',']
        self.use_cuda = 0   # this is for GPU usage only, 1-use GPU if available, 0-use CPU

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # print("1", x.shape)
        x = self.relu(self.fc1(x))
        # print("2", x.shape)
        x = self.relu(self.fc2(x))
        # print("3", x.shape)
        x = self.relu(self.fc3(x))
        # print("4", x.shape)
        # activation function?
        return x

class Dataset(object):      # This class is to create data loaders
    def __init__(self, train_tag, train_patterns):
        self.sample_size = len(train_patterns)
        self.data = train_patterns
        self.target = train_tag

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.sample_size

def use_cuda():     # for GPU usage only
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Training on CPU') 


def loadjson(params):
    with open(params.jsfile, 'r') as f:
        js = json.load(f)
    print("loadjson:", js)
    return js

def getWords(params, js):
    """
    nlp process contains:
        1. tokenize
        2. lower all characters (case insensitive)
        3. stem the words
        4. exclude punctuations
    """
    tags = []
    patterns = []
    tag_with_patterns = []

    for intent in js['intents']:
        tag = intent['tag']     # there are one tag for each entry
        tags.append(tag)
        for pattern in intent['patterns']:   # there are mul. patterns in each entry
            # this contains all possible user inputs
            pattern = nltk.word_tokenize(pattern)
            patterns.extend(pattern)
            tag_with_patterns.append((tag, pattern))
    # print("tags contain:", tags)
    # print("tokenized patterns contain:", patterns)
    # print("tag with tokenized patterns tuple contain:", tag_with_patterns)

    # lower case all patterns to make the possible inputs case insensitive
    # then stem the words to get the enssentials of the word
    # I have tried both Porter stemmer and lacaster stemmer, Porter stemmer maintains more of the words
    stem_words = sorted(set([stemmer.stem(word.lower()) for word in patterns if word not in params.igrnore_char]))

    # print("lower case/tokenized/stem patterns contain:", stem_words)

    return tags, patterns, stem_words, tag_with_patterns


def getDataLoader(params, tags, stem_words, tag_with_patterns):
    train_tag = []
    train_patterns = []


    for (tag, patterns) in tag_with_patterns:
        label = [] 
        stem_tuple_patterns = [stemmer.stem(pattern.lower()) for pattern in patterns]
        for idx, word in enumerate(stem_words):
            print("for word", word,"at index", idx)
            if word in stem_tuple_patterns:
                label.append(1)
            else:
                label.append(0)
        print("Labels in getDataLoader:", label)

        # print("TEST TEST TEST", tag)
        train_tag.append(tags.index(tag))
        train_patterns.append(label)    # notice some of labels are ignored due to caps
    
    print("\ntrain_tag:", train_tag)
    print("train_patterns:", train_patterns)


    train_patterns = np.array(train_patterns, dtype=np.float32)

    print("numpy train_patterns:", train_patterns)
    
    return train_tag, train_patterns


def train(params, model, dataloader):
    crossEntropy = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=params.lr)
    for epoch in tqdm(range(params.epoch)):
        for (data, target) in dataloader:
            # data = data.to(device)    # for GPU only
            # target = target.to(device)    # for GPU only
            out = model(data)
            loss = crossEntropy(out, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

    print("Result loss =", loss.item())
    ckpt = {
        'model':model.state_dict()
    }
    torch.save(ckpt, 'testmodel.pth')




def evaluate(params, model, usr_input, tags, stem_words):
    ckpt = torch.load('testmodel.pth')
    model.load_state_dict(ckpt['model'])

    model.eval()
    patterns = nltk.word_tokenize(usr_input)  # tokenized user input

    # labels = []

    
    # stem_input = [stemmer.stem(pattern.lower()) for pattern in patterns]
    # for (tag, patterns) in tag_with_patterns:
        # label = []
    stem_tuple_patterns = [stemmer.stem(pattern.lower()) for pattern in patterns]
    # i=0
    # while(i<10)
    #     for idx, word in enumerate(stem_input):
    #         print("for input word", word,"at index", idx)
    #         if word in stem_tuple_patterns:
    #             label.append(1)
    #             i+=1
    #         else:
    #             label.append(0)
    #             i+=1
    #     label.append(0)
    #     i+=1
    labels = np.zeros(len(stem_words), dtype=np.float32)
    for idx, word in enumerate(stem_words):
        if word in stem_tuple_patterns:
            labels[idx] = 1
    
    # labels.append(label)
    # print("input label:", labels)
    
    labels = np.reshape(labels, (1, np.shape(labels)[0]))
    # print("reshaped label:", labels)
    labels = torch.from_numpy(labels)
    out = model(labels)
    _, pred = torch.max(out, dim=1) 
    tag = tags[pred.item()]
    # print("pred:", pred)
    # print("predicted tag:", tag)

    prob = torch.softmax(out, dim=1)[0][pred.item()]
    # print("prob:", prob)

    if prob > 0.50:
        return tag
    else:
        return -1


def chat(params, model, tags, stem_words, js):
    usr_input = input("You: ")
    rsp = evaluate(params, model, usr_input, tags, stem_words)
    if rsp == -1:
        print("Chatters: Sorry I do not understand.")
    else:
        print("typing ...")
        for intent in js['intents']:
            if rsp == intent['tag']:
                print("Chatter:", random.choice(intent['responses']))



def main():
    params = Params()
    js = loadjson(params)
    tags, patterns, stem_words, tag_with_patterns = getWords(params, js)

    train_tag, train_patterns = getDataLoader(params, tags, stem_words, tag_with_patterns) 
    # these training have turned words to numbers for calculations

    dataset = Dataset(train_tag, train_patterns)
    dataloader = DataLoader(dataset=dataset, batch_size=params.batch) 
    # the dataloader separate data into batches

    if params.use_cuda:     # for GPU usage only
        use_cuda()

    # print(len(train_patterns[0]))
    model = Linear(len(train_patterns[0]), len(tags))   

    if not params.load_weights:
        train(params, model, dataloader)
    else:
        print("You're now connected to bot Chatters.")
        print("Hello, how can I assist you today?")
        print("(you can disconnect any time by typing 'exit')")
        while True:
            chat(params, model, tags, stem_words, js)

main()