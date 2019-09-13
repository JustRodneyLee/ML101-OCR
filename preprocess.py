from scipy import misc
from matplotlib.pyplot import imread
import numpy as np

def main(src, dst):
    with open(src, 'r') as f:
        list = f.readlines()
    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')
        print(name + 'processed')
        img = imread(name)
        #img = img/255
        img.resize((img.size, 1))
        data.append(img)
        labels.append(int(label))

    print('write to npy')
    np.save(dst, [data,labels])
    print('completed')

if __name__ == "__main__":
    main("validate.txt","validate.npy")
    main("test.txt", "test.npy")
    main("train.txt", "train.npy")
