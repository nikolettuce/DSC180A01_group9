# create PyTorch dataset for the training data
import torchvision
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    '''
    CNN architecture taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    def __init__(self):
        super(Net, self).__init__()
        # 3 channel image means first input is 3, then change output to 6, batch size 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # scales tensor
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(51984, 920) # input for matmult, output to decrease dimensionality
        self.fc2 = nn.Linear(920, 170)
        self.fc3 = nn.Linear(170, 80) # output must end in class number aka 80

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features

def run_model(loader, net, epochs, lr, momentum, modelDir, device):
    print()
    print("STARTING MODEL TRAINING.")
    # define loss fn and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # enable cuda processing

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5000 == 4999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training, pog')
    
    print('Saving model to: {}'.format(modelDir))
    torch.save(net.state_dict(), modelDir)

def test_model(net, testloader, device):
    # accuracy test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 2500 test images: %d %%' % (
        100 * correct / total))
    