import numpy as np

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
    def __init__(self):
        pass
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y
    
    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
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

class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])
        self.accuracy = 1.0 * self.accuracy / x.shape[0]
        return self.accuracy


def main():
    dataLayer1 = InputLayer('train.npy', 1024)
    dataLayer2 = InputLayer('validate.npy', 10000)
    hiddenLayers = []
    hiddenLayers.append(FullyConnectedLayer(17*17, 26))
    hiddenLayers.append(ActivationLayer())
    lossLayer = QuadraticLoss()
    accuracy = Accuracy()
    rate = 1000
    for layer in hiddenLayers:
            layer.rate = rate
        
    epochs = 30
    for i in range(epochs):
        print('Epoch ',i)
        
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
                print('Loss:',lossSum / iterations)                
                print('Accuracy:',acc)
                print('Learning Rate:',rate)
                #Changing learning rate
                break
    np.savetxt("OCR_en_US.csv", hiddenLayers[0].)
        
if __name__ == '__main__':
    main()

