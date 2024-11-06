import struct
from array import array
import numpy as np
from torch import nn
import torch

path = 'trainedweights/model_scripted.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  

# Set file paths based on added MNIST Datasets
training_images_filepath = 'samples/train-images.idx3-ubyte'
training_labels_filepath = 'samples/train-labels.idx1-ubyte'
test_images_filepath = 'samples/t10k-images.idx3-ubyte'
test_labels_filepath = 'samples/t10k-labels.idx1-ubyte'

accuracy = []

# Load MINST dataset
print('Loading MNIST dataset...')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print('MNIST dataset loaded.')
image = np.asarray(x_train[2])
# plt.imshow(image)
# plt.show()

# print(len(x_train))
# conv1 = nn.Conv2d(in_channels=1,out_channels=28,kernel_size=5,    
#                       stride=1,        
#                       padding='same',      
#             )
# out = conv1(x_np)
# plt.imshow(out.reshape((28,28)).detach().cpu().numpy())
# plt.show()

class ImageRecognize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,    
                      stride=2,        
                      padding=1,      
            )
        self.pool1 = nn.MaxPool2d(padding=1,kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,    
                      stride=1,        
                      padding=1,      
            )
        self.pool2 = nn.MaxPool2d(padding=1,kernel_size=2)
        
        self.compcon1 = nn.Linear(in_features=1600,out_features=60)
        self.relu = nn.ReLU()
        self.compcon2 = nn.Linear(in_features=60,out_features=10)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
    def Load(self,path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()
        
    def Save(self,path):
        torch.save(self.state_dict(), path)
        
    def Forward(self,x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.compcon1(out)
        out = self.relu(out)
        out = self.compcon2(out)
        out = self.softmax(out)
        return out
    
    def Backward(self,x_train,y_train):
        for i in range(len(x_train)):
            label = torch.tensor([int(y_train[i])])
            np_array = np.array(x_train[i])
            x_np = torch.from_numpy(np_array).reshape((1,1,28,28)).to(torch.float32)
            out = self.Forward(x_np.to(device))
            loss_fun = torch.nn.NLLLoss()
            loss = loss_fun(out,label)
            print(out)
            optim = torch.optim.SGD(self.parameters(), lr=1e-3, momentum = 0.9)  
            optim.zero_grad()
            loss.backward()
            optim.step() #gradient descent
        self.Save(path=path)
        

model = ImageRecognize()

np_array = np.array(x_train[900])
x_np = torch.from_numpy(np_array).reshape((1,1,28,28)).to(torch.float32)


model.Load(path)
out = model.Forward(x_np)
print(out,y_train[900])
_, predicted = torch.max(out, 1)
print(predicted)

