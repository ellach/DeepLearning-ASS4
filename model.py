import torch
import torch.nn as nn
import json
import random
import numpy as np
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt

LABELS = { "entailment": 0, "contradiction": 1, "neutral": 2}

HIDDEN_DIM = 300
EMBEDDING_DIM = 300
LABELS_SIZE = 3
BATCH_SIZE = 4 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.F1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.F2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.G1 = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
        self.G2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.H1 = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
        self.H2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.lastLinear = nn.Linear(HIDDEN_DIM, LABELS_SIZE) 

        self.dropout = nn.Dropout(0.2) 
        self.softmax = nn.Softmax(dim=0) 
        self.relu = nn.ReLU(inplace=True)        
 
    def forward(self, sent1, sent2):

        padded_S1 = np.ones((HIDDEN_DIM, HIDDEN_DIM))*0
        padded_S2 = np.ones((HIDDEN_DIM, HIDDEN_DIM))*0
        padded_S1[:len(sent1)-1] = sent1[:len(sent1)-1] 
        padded_S2[:len(sent2)-1] = sent2[:len(sent2)-1]

        sent1 = Variable(torch.Tensor(padded_S1).float())
        sent2 = Variable(torch.Tensor(padded_S2).float())

        # F step  
        fs1 = self.relu(self.dropout(self.F1(sent1)))
        fs2 = self.relu(self.dropout(self.F2(sent2)))
        fs1 = self.relu(self.dropout(fs1))
        fs2 = self.relu(self.dropout(fs2))

        # Attention scoring
        score1 = fs1 * torch.transpose(fs2, 0,1)
        prob1 = self.softmax(score1)
        score2 = torch.transpose(score1, 0,1)
        prob2 = self.softmax(score2)

        # Align pairs using attention
        sent1align = torch.cat([sent1, prob1 * sent2])
        sent2align = torch.cat([sent1, prob1 * sent2])
        
        # G step 
        sent1align = sent1align.view(-1, sent1align.size(0))
        gs1 = self.relu(self.dropout(self.G1(sent1align)))
        sent2align = sent2align.view(-1, sent2align.size(1))
        gs2 = self.relu(self.dropout(self.G2(sent2align)))

        gs1 = self.relu(self.dropout(gs1))
        gs2 = self.relu(self.dropout(gs2))
      
        # Sum
        ss1 = torch.sum(gs1, dim=0) 
        ss2 = torch.sum(gs2, dim=0)

        concat = torch.transpose(torch.cat([ss1, ss2]),-1,0)

        # H step
        hs = self.relu(self.dropout(self.H1(concat))) 
        hs = hs.view(-1,hs.size(0)) 
        hs = self.relu(self.dropout(self.H2(hs)))

        return self.lastLinear(hs)


    def predict(self, data):
        predictions_tensors = [(self.forward(s1, s2).data).numpy() for (s1, s2, label) in data]
        return [np.argmax(arr[0]) for arr in predictions_tensors]

    def  accuracy(self, data):
        good = total = 0.0
        predicted = self.predict(data) 
        golds = [label for (s1, s2, label) in data]
        for pred, gold in zip(predicted, golds):
            total += 1
            if pred == gold:
                good += 1

        return good / total


def read_vectors(file_name):
    words = {}
    vectors = []
    with open(file_name, "r") as lines:
         for line in lines:
             vector = line.split()
             word = vector.pop(0)
             words[word] = np.array([float(v) for v in vector])

    for i, w in enumerate(words):
        if i > 100:       
           break
        vectors.append(words[w])   
    return (words, np.average(vectors, axis=0))


def preprocess_data(type, file_name, words, vectors):
    data = []
    with open(file_name) as lines:
         for line in lines:
             json_line = json.loads(line)
             if json_line["gold_label"] != "-":
                s1 = json_line["sentence1"].lower().rstrip(".").split()
                s2 = json_line["sentence2"].lower().rstrip(".").split()
                sent1 = sent2 = []
                for word in s1:
                    if word in words:
                       sent1.append(words[word])
                    else:
                        sent1.append(vectors) 
                for word in s2:
                    if word in words: 
                       sent2.append(words[word]) 
                    else:
                       sent2.append(vectors) 
                data.append((sent1, sent2, LABELS[json_line["gold_label"]]))
    return data


def get_graphs(file_name):
    accuracy = []
    loss = []
    with open(file_name) as f:
         for line in f.read().split('\n'):
             if "Accuracy" in line: 
                 acc, val = line.split(" :",1)
                 accuracy.append(float(val))
             elif "Loss" in line: 
                  lss, val = line.split(" ",1)
                  loss.append(float(val)) 

    #print("Average accuracy: ", np.average(accuracy), "  Length:  ", len(accuracy))
    plt.figure()
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
   
    y = [i for i in range(len(accuracy))]
    plt.plot(y, accuracy, label = "")
    plt.grid(True)

    plt.ylim(0,1)
    plt.xlim(0,len(accuracy)) 
    plt.legend()
    plt.savefig('plot_acc.png')

                        
 
if __name__ == '__main__':

    '''
    if len(sys.args) < 4:
       print("Error missing arguments")
       return
    else:
    '''

    train = sys.argv[1] 
    dev = sys.argv[2]  
    test = sys.argv[3]
    w2v = sys.argv[4] 
        
    #get_graphs(sys.argv[5])    
    words, vectors = read_vectors(w2v)

    trainData = preprocess_data("train", train, words, vectors)
    devData = preprocess_data("dev", dev, words, vectors)
    testData = preprocess_data("test", test, words, vectors)

    model = Model()
    loss_fc = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)

    for EPOCH in range(5):
        totalLoss = 0 
        random.shuffle(trainData)
        for i, (s1, s2, label) in enumerate(trainData, 1):
   
            if i % (BATCH_SIZE * 25) == 0:
               accuracy = model.accuracy(devData)
               print("Accuracy_dev :  ", accuracy)

            if i % BATCH_SIZE == 0:
                y_hat = model(s1, s2)
                y_hat = y_hat.view(-1, LABELS_SIZE) 

                loss = loss_fc(y_hat, torch.LongTensor([label]))
                totalLoss += loss.item()

                optimizer.zero_grad() 
                loss.backward(retain_graph=True)
                optimizer.step()

        print("Loss:  ", totalLoss/len(trainData))     
    accuracy = model.accuracy(testData)
    print("Accuracy_test :  ", accuracy)


