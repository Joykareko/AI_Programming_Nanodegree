# Imports here
import pandas as pd
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import torch  #pytorch package
import torchvision #for loading the data
import torchvision.transforms as transforms  #transforms the data
import torch.nn as nn #build network
import torch.optim as optim #for optimizing
import torchvision.models as models
from collections import OrderedDict
import os
from PIL import Image
import seaborn as sb
from train import model

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),

                                        ])
    return preprocess(im)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, category_names,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.unsqueeze_(dim = 0) #picks 1 image
    
    model.to(device).float()
    model.eval()#turns off gradients
    
    with torch.no_grad():
        output = model.forward(image.to(device))
    out_prob = torch.exp(output)  #converts to probabilities
    
    #Take top-k classifications
    top_probs, top_classes = out_prob.topk(topk, dim = 1)
        

    
    top_probs = top_probs.tolist () [0] #converting both to list
    top_classes = top_classes.tolist () [0]
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    pred_classes = []
    for value in top_classes:
        pred_classes.append(list(model.class_to_idx)[value]) #Take the class from the index
        
    #Obtain the corresponding class name using cat_to_name json file
    top_names = []
    for pred_class in pred_classes:
        top_names.append(cat_to_name[pred_class])
    
    return top_probs,top_names

    
    
    
if __name__ == '__main__':  #used to execute my code since I will running this file directly
    
    parser = argparse.ArgumentParser (description = "Parser of training script")

    parser.add_argument ('--image_path', help = 'Provide data directory. Optional argument', 
                         default = 'ImageClassifier/flowers/test/102/image_08004.jpg', type = str)
    parser.add_argument ('--load_dir', help = 'Provide saving directory. Optional argument', default = 'ImageClassifier/checkpoint.pth',type = str)
    parser.add_argument ('--category_names', help = "Option to use GPU",default = 'ImageClassifier/cat_to_name.json' ,type = str)

    #setting values data loading
    args = parser.parse_args ()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(file_path = args.load_dir)
    top_p,top_class = predict(image_path = args.image_path,model = model,category_names = args.category_names)
    print(top_p,top_class)
    
    