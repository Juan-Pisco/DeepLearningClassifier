import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import json
from PIL import Image


def get_pargs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_path', action="store")
    parser.add_argument('checkpoint', action="store")
    
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'choose how many probabilities and classes are shown') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'choose th file of category-names') 
    parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='set gpu usage to true')
    in_args = parser.parse_args()
    
    return in_args


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.fc = nn.Sequential(
    nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers']),
    nn.ReLU(),
    nn.Dropout(p = 0.2),
    nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size']),
    nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping_classes']
    return model, checkpoint['mapping_classes']

def process_image(image):
    image = Image.open(image)
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    image = preprocess(image)
    return image


def predict(filepath, image_path, top_k, category_names, gpu_b):
    model, class_to_idx = load_checkpoint(filepath)
    device = 'cuda' if gpu_b == True else 'cpu'
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model.eval()
    model.to(device)
  
    img = process_image(image_path)
    img = img.unsqueeze(0)
    inputs = img.to(device)
    output = model.forward(inputs)
    ps = torch.exp(output) 
    
    prob = torch.topk(ps, top_k)[0].tolist()[0]  
    index = torch.topk(ps, top_k)[1].tolist()[0]     
    item = []
    for i in range(len(model.class_to_idx.items())):
        item.append(list(model.class_to_idx.items())[i][0])
        
    classes = []
    for i in range(top_k):
        classes.append(item[index[i]])
    classes_predict = [cat_to_name[c] for c  in classes]
    
    print('Top ', top_k, ' probabilites predicted from greatest to least:\n', prob, '\n')
    
    print('Top ', top_k, ' classes predicted from greatest to least:\n', classes_predict, '\n')
    
    return prob, classes_predict 

def main():
   in_args = get_pargs()
   print('Predicting your image...')
   predict(in_args.checkpoint, in_args.input_path, in_args.top_k, in_args.category_names, in_args.gpu)

if __name__ == '__main__':
    main()
    
