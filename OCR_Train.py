import numpy as np
import os

class InputLayer:
    def __init__(self, name, batch_size):
        with open(name, 'rb') as f:
            data = np.load(f, None, True)
        self.x = data[0]
        self.y = data[1]
        self.l = len(self.x)
        self.batch_size = batch_size
        self.pos = 0

    def forward(self):
        if self.pos + self.batch_size >= self.l:
            ret = (self.x[self.pos:self.l], self.y[self.pos:self.l])
            self.pos = 0
            index = range(self.l)
            np.random.shuffle(list(index))
            self.x = self.x[index]
            self.y = self.y[index]
        else:
            ret = (self.x[self.pos:self.pos + self.batch_size], self.y[self.pos:self.pos + self.batch_size])
            self.pos += self.batch_size

        return ret, self.pos

    def backward(self, d):
        pass

class FullyConnectedLayer:
    def __init__(self,l_x,l_y):
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)
        self.rate = 0
        
    def load(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        self.x = x
        self.y = np.array([np.dot(self.weights,data) + self.bias for data in x])
        return self.y

    def backward(self, d):
        #calculating gradient by multiplying x with the derivative given by back-propagation
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])
        self.weights -= self.rate * self.dw #calibration of weights
        self.bias -= self.rate * self.db #calibration of biases
        return self.dw, self.db        

class ActivationLayer:
    def __init__(self, actType):
        self.activation = actType
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        i = 0
        out = np.zeros(x.shape)
        for xx in x:
            if np.sum(xx)>0:
                out[i]=xx
            i+=1
        return out
    
    def forward(self, x):
        self.x = x
        if self.activation=="Sigmoid":            
            self.y = self.sigmoid(x)
        elif self.activation=="RELU":            
            self.y = self.relu(x)
        return self.y
    
    def backward(self, d):
        if self.activation=="Sigmoid":
            sig = self.sigmoid(self.x)
            self.dx = d * sig * (1 - sig)
        elif self.activation=="RELU":
            if np.sum(self.y)>0:
                self.dx = 1
            else:
                self.dx = 0
        return self.dx

class QuadraticLoss:
    def __init__(self):
        pass
    
    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2
        return self.loss
    
    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx

class CrossEntropyLoss:
    def __init__(self):
        pass    
        
    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.nan_to_num(-self.label * np.log(x) - ((1 - self.label) * np.log(1 - x)))
        self.loss = np.sum(self.loss) / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x / (1 - self.x)
        return self.dx

class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])
        self.accuracy = 1.0 * self.accuracy / x.shape[0]
        return self.accuracy


def main():
    print("Enter target Accuracy:")
    targetAccuracy = int(input())
    dataLayer1 = InputLayer('train.npy', 1024)
    dataLayer2 = InputLayer('validate.npy', 10000)
    hiddenLayers = []
    hiddenLayers.append(FullyConnectedLayer(17*17, 26))
    if os.path.isfile("OCR_en_US_weights.csv") & os.path.isfile("OCR_en_US_biases.csv"):
        hiddenLayers[0].load(np.loadtxt("OCR_en_US_weights.csv",delimiter=","),np.loadtxt("OCR_en_US_biases.csv",delimiter=","))
    hiddenLayers.append(ActivationLayer("Sigmoid"))
    lossLayer = QuadraticLoss()
    accuracy = Accuracy()
    acc = 0
    #calib = 1.5
    rate = 1100
    for layer in hiddenLayers:
            layer.rate = rate

    #lastLoss = -1
    #epochs = 10
    #for i in range(epochs):
    i = 0
    while acc*100<targetAccuracy:
        print('Epoch ',i)
        i+=1
        lossSum = 0        
        iterations = 0
        while True:
            data, pos = dataLayer1.forward()
            x, label = data
            for layer in hiddenLayers:
                x = layer.forward(x)
            loss = lossLayer.forward(x, label)
            lossSum += loss
            iterations += 1
            d = lossLayer.backward()

            for layer in hiddenLayers[::-1]:
                d = layer.backward(d)

            if pos==0:
                data, _ = dataLayer2.forward()
                x, label = data
                for layer in hiddenLayers:
                    x = layer.forward(x)
                acc = accuracy.forward(x, label)
                averageLoss = lossSum / iterations
                print('Loss:',averageLoss)                
                print('Accuracy:',acc)
                print('Learning Rate:',rate)
                #Changing learning rate
                '''if (lastLoss!=-1):                    
                    if (averageLoss<=lastLoss):
                        rate*=calib
                    else:
                        rate/=calib
                lastLoss = averageLoss
                for layer in hiddenLayers:
                    layer.rate = rate'''
                break
    np.savetxt("OCR_en_US_weights.csv", hiddenLayers[0].weights, delimiter=",")
    np.savetxt("OCR_en_US_biases.csv", hiddenLayers[0].bias, delimiter=",")
        
if __name__ == '__main__':
    main()

