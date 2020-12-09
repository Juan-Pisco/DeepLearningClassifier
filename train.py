import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

resnet50 = models.resnet50(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet50, 'alexnet': alexnet, 'vgg': vgg16}


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', action="store")
    
    parser.add_argument('--save_dir', type = str, default = '/', 
                    help = 'path to the checkpoints.pth') 
    parser.add_argument('--arch', type = str, default = 'resnet50', 
                    help = 'choose the CNN model you want to use') 
    parser.add_argument('--learning_rate', type = float, default = 0.01, 
                    help = 'learning rate value') 
    parser.add_argument('--hidden_units', type = int, default = 1024, 
                    help = 'hidden units number') 
    parser.add_argument('--epochs', type = int, default = 20, 
                    help = 'number of epochs') 
    parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='set gpu usage to true')
    in_args = parser.parse_args()
    
    return in_args



def training(hidden_units, arch, datadir, epochs, gpu_b, lrv, saving):
    
    train_dir = datadir + '/train'
    
    valid_dir = datadir + '/valid'
    
    
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])



    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    
    model = models[arch]
    
    device = 'cuda' if gpu_b == True else 'cpu'
    
    for param in model.parameters():
        param.requires_grad = False
    
    if models[arch] == resnet50:
        input_units = 2048
        
        model.fc = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p = 0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.fc.parameters(), lr=lrv)
        
    elif models[arch] == alexnet:
        input_units = 9216
        
        model.classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p = 0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.classifier.parameters(), lr=lrv)
        
    else:
        input_units = 25088
        
        model.classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p = 0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.classifier.parameters(), lr=lrv)
    
    criterion = nn.NLLLoss()

    model.to(device)
    
    running_loss = 0

    steps = 0

    for e in range(epochs):
        print('Epoch number: ', e+1)
        for inputs, labels in train_loader:

            #Training Loop

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            steps += 1

            if steps == 5:
                model.eval()
                accuracy = 0
                valid_loss = 0

                with torch.no_grad():

                    for inputs, labels in valid_loader:

                        #Validation Loop

                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        loss_valid = criterion(outputs, labels)
                        valid_loss += loss_valid.item()

                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(
                      f"Train loss: {running_loss/steps:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                running_loss = 0 
                steps = 0        
                model.train()
                
            
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': input_units,
                  'model': models[arch],
                  'output_size': 102,
                  'hidden_layers': hidden_units,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict,
                  'mapping_classes': model.class_to_idx}

    torch.save(checkpoint, saving + '/training.pth')
    print(model)
    print('Training finished!')


def main():
   in_args = get_args()
   print('Training Started!')
   training(hidden_units = in_args.hidden_units, arch = in_args.arch, datadir = in_args.data_directory, epochs = in_args.epochs, gpu_b = in_args.gpu, lrv = in_args.learning_rate, saving = in_args.save_dir)
   
    
if __name__ == '__main__':
    main()