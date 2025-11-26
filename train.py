import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


DEVICE = "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
       
        #3*32*32(one image)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        #6*28*28 -> 6*14*14(after pooling)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        #16*10*10 -> 16*5*5(after pooling)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #16*5*5 = 400
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def get_dataloaders(data_dir,batch_size=16):
    train_dir = os.path.join(data_dir,"train")
    test_dir = os.path.join(data_dir,"test")

    #this function ends up getting applied to all the images
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5,0.5,0.5)
        )
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = ["hotdog", "nothotdog"]
    return train_loader, test_loader, class_names



if __name__=="__main__":
    data_dir = "data"
    train_loader,test_loader,class_names = get_dataloaders(data_dir)
    print(train_loader)
    print(class_names)
    iterator = iter(train_loader)
    images, labels = next(iterator)
    print(images)
    print(labels)
    model = Net()
    print(images[0])
    print(model(images))
