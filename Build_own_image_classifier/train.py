# Imports here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch  #pytorch package
import torchvision #for loading the data
import torchvision.transforms as transforms  #transforms the data
import torch.nn as nn #build network
import torch.optim as optim #for optimizing
import torchvision.models as models
from collections import OrderedDict
import os
import argparse
from PIL import Image
import seaborn as sb

def data_transform(data_dir):
#data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    batch_size = 64

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),

            ]) #transforming the training data
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),

                                            ]) # transforming the testing/valid image data

    # TODO: Load the datasets with ImageFolder
    train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
    validation_data = torchvision.datasets.ImageFolder(root=valid_dir, transform=test_transform)

    # TODO: Using the image datasets and the transforms, define the dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    return train_data_loader,test_data_loader,validation_data_loader,train_data

def model(arch):
    model = models.vgg16(pretrained=True) # using the pretrained vgg16 model with 16 layers
    #turn off gradients
    for param in model.parameters():
        param.requires_grad = False

        #need to specify a classifier that is in line with my data

        classifier = nn.Sequential  (OrderedDict ([
                                    ('fc1', nn.Linear (25088, 4096)),
                                    ('relu1', nn.ReLU ()),
                                    ('dropout1', nn.Dropout (p = 0.2)),
                                    ('fc2', nn.Linear (4096, 2048)),
                                    ('relu2', nn.ReLU ()),
                                    ('dropout2', nn.Dropout (p = 0.2)),
                                    ('fc3', nn.Linear (2048, 102)),
                                    ('output', nn.LogSoftmax (dim =1))
                                ]))

        model.classifier = classifier
        return model



def train(model, train_data_loader, validation_data_loader, criterion, optimizer, device,epochs=5, print_every=40):
    
    #epochs = 5
    steps = 0
    running_loss = 0
    model.to(device)
    #print_every = 40
    for epoch in range(epochs):
        for inputs, labels in train_data_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)  #calculates the output
            loss = criterion(logps, labels)  #calculates the loss
            loss.backward()
            optimizer.step()  #

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()  #turns off dropout
                with torch.no_grad():  #turns off gradients
                    for inputs, labels in validation_data_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps2 = model.forward(inputs)
                        batch_loss = criterion(logps2, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps2)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                print("Epoch:{}/{}..".format(epoch+1,epochs),
                       "Train loss:{:.3f}..".format (running_loss/print_every),
                       "Val loss: {:.3f}.." .format(val_loss/len(validation_data_loader)),
                        "Val accuracy: {:.3f}%".format(accuracy/len(validation_data_loader)*100)
                             )

                running_loss = 0
                model.train()  
    print('Model training done!..')
def test(model, test_data_loader, criterion,device):
        val_loss = 0
        accuracy = 0
        model.to(device)
        model.eval()  #turns off dropout
        with torch.no_grad():  #turns off gradients
            for inputs, labels in test_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps2 = model.forward(inputs)
                batch_loss = criterion(logps2, labels)

                val_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps2)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(
              "Model done testing',test accuracy: {:.3f}%".format(accuracy/len(test_data_loader)*100)
             )
def save_model(model,train_data,save_dir):
    model.class_to_idx = train_data.class_to_idx

    model.to('cpu')

    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'mapping':    model.class_to_idx
             } 
    if args.save_dir:
        torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
    else:
        torch.save (checkpoint, 'checkpoint.pth')
    print('Model saved')
    
if __name__ == '__main__':  #used to execute my code since I will running this file directly
    parser = argparse.ArgumentParser (description = "Parser of training script")

    parser.add_argument ('--data_dir', help = 'Provide data directory. Optional argument', default = 'ImageClassifier/flowers', type = str)
    parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', default = 'ImageClassifier',type = str)
    parser.add_argument ('--arch', help = 'Vgg16 can be used if this argument specified', default = 'vgg',type = str)
    #parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

    #setting values data loading
    args = parser.parse_args ()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load data
    train_data_loader,test_data_loader,validation_data_loader,train_data = data_transform(data_dir = args.data_dir)#loads data
    
    #load model
    model = model(arch = args.arch)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = 0.001)
    
    #train
    train(model = model, train_data_loader = train_data_loader, 
          validation_data_loader = validation_data_loader, 
          criterion=criterion, optimizer=optimizer, device = device,epochs=5, print_every=40)
    #test
    test(model = model,test_data_loader = test_data_loader,criterion = criterion,device = device)
    
    #save
    
    save_model(model = model, train_data = train_data,save_dir = args.save_dir)
    
    
    