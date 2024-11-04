from os.path  import join
import struct
from array import array
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn
import torch

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


# Load MINST dataset
print('Loading MNIST dataset...')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print('MNIST dataset loaded.')
# image = np.asarray(x_train[2])
# plt.imshow(image)
# plt.show()

np_array = np.array(x_train[2])
x_np = torch.from_numpy(np_array).reshape((1,1,28,28)).to(torch.float32)
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
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=28,kernel_size=5,    
                      stride=1,        
                      padding='same',      
            ),nn.ReLU())
        self.pool1 = nn.MaxPool2d(padding=1,kernel_size=2)
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=28,out_channels=64,kernel_size=5,    
                      stride=1,        
                      padding='same',      
            ),nn.ReLU())
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=0)
        
        self.compcon1 = nn.Linear(in_features=4096,out_features=5000)
        self.compcon2 = nn.Linear(in_features=5000,out_features=10)
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.flatten(out)
        out = self.compcon1(out)
        out = self.compcon2(out)
        out = self.relu(out)
        out = self.softmax(out)
        
        return out

model = ImageRecognize()
print(model)
out = model.forward(x_np)
print(out)