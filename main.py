from src.ImageReccognition import ImageRecognize
from src.loader import MnistDataloader
from src.constants import path,test_images_filepath,test_labels_filepath,training_images_filepath,training_labels_filepath

# Load MINST dataset
print('Loading MNIST dataset...')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print('MNIST dataset loaded.')


               

model = ImageRecognize(1)                            #initialize ImageRecognize object
model.Load(path)                                     #load the weights
model.Backward(x_train,y_train)                      #do a backward pass and save the weights
model.Testing(testImages=x_test,testLabels=y_test)   #Test and check the accuracy
