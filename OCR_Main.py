from scipy import misc
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np
import os

def ConvertToArray(src):    
    img = imread(src)
    #img.resize((289,1))
    #img = Image.open(src).convert('LA')
    #img.show()
    ret = np.array(img)
    ret.resize(289, 1, refcheck = False)
    #ret = ret/255
    #Image.fromarray(ret).show()
    #print(ret)
    #for i in zip(ret):
    #    if (i[0]>0.5):
    #        i = (1)
    #    else:
    #        i = (0)
    return ret      

class FullyConnectedLayer:
    def __init__(self,l_x,l_y):
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)
        
    def load(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        self.x = x
        self.y = np.array([np.dot(self.weights,data) + self.bias for data in x])
        return self.y  

class ActivationLayer:
    def __init__(self, actType):
        self.activation = actType
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        i = 0
        out = np.zeros_like(x)
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

def main():
    print("Loading OCR...")
    hiddenLayers = []
    hiddenLayers.append(FullyConnectedLayer(17*17, 26))
    hiddenLayers[0].load(np.loadtxt("OCR_en_US_weights.csv",delimiter=","),np.loadtxt("OCR_en_US_biases.csv",delimiter=","))
    hiddenLayers.append(ActivationLayer("Sigmoid"))
    #hiddenLayers : FullyConnectedLayer->ActivationLayer
    print("Loading Complete!")
    while True:
        print("Enter path of image:")
        path = input()
        if (path=="bye" or path=="quit" or path=="exit"):
            break
        dat = ConvertToArray(path)
        x = []
        for i in range(1, 26):
            x.append(dat)
    
        for layer in hiddenLayers:
            x = layer.forward(x)
    
        #ASCII A 65 Z 90
        print(chr(np.argmax(hiddenLayers[1].y)+65))
    
if __name__ == "__main__":
    main()
