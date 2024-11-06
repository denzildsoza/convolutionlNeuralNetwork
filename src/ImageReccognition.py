import numpy as np
from torch.nn import Conv2d,MaxPool2d,Linear,ReLU,LogSoftmax,Module,NLLLoss
from torch import load,save,tensor,from_numpy,max,float32,no_grad,device,cuda
from torch.optim import SGD
from src.constants import path

device = device('cuda' if cuda.is_available() else 'cpu')

class ImageRecognize(Module):
    def __init__(self,channels) -> None:
        super().__init__()
        
        self.conv1 = Conv2d(in_channels=channels,out_channels=32,kernel_size=3,    
                      stride=2,        
                      padding=1,      
            )
        self.pool1 = MaxPool2d(padding=1,kernel_size=2)
        
        self.conv2 = Conv2d(in_channels=32,out_channels=64,kernel_size=3,    
                      stride=1,        
                      padding=1,      
            )
        self.pool2 = MaxPool2d(padding=1,kernel_size=2)
        
        self.compcon1 = Linear(in_features=1600,out_features=60)
        self.relu = ReLU()
        self.compcon2 = Linear(in_features=60,out_features=10)
        self.softmax = LogSoftmax(dim=1)
        
        
    def Load(self,path):
        self.load_state_dict(load(path, weights_only=True))
        self.eval()
        
    def Save(self,path):
        save(self.state_dict(), path)
        
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
            label = tensor([int(y_train[i])]).to(device)
            np_array = np.array(x_train[i])                                #convert 2d array to np array
            x_np = from_numpy(np_array).reshape((1,1,28,28)).to(float32)   #convert numpy array returned array to a tensor with 1 channel
            out = self.Forward(x_np.to(device))                            #do a forward and get the output
            loss_fun = NLLLoss()                                           #initialize NLLoss function has its a classification function
            loss = loss_fun(out,label)                                     #calculate loss
            optim = SGD(self.parameters(), lr=1e-3, momentum = 0.9)        #initialize SDG optimizer    
            optim.zero_grad()                                              #initialize all the gradients to zero
            loss.backward()                                                #this uses pytorch autograd algo to calculate weights from loss
            optim.step()                                                   #optimize and save weights from grad to parameters of the model
        self.Save(path=path)                                               #save model weights to system
        
    def Testing(self,testImages,testLabels):
        correct,total = 0,0
        with no_grad():
            for i in range(len(testImages)):
                label = tensor([int(testLabels[i])])
                np_array = np.array(testImages[i])
                x_np = from_numpy(np_array).reshape((1,1,28,28)).to(float32)
                # calculate outputs by running images through the network
                outputs = self.Forward(x_np)
                # the class with the highest energy is what we choose as prediction
                _, predicted = max(outputs.data, 1)
                total+=1
                correct += (predicted == label).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')