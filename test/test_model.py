import torch
import os

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_testdata(input_dir, label_dir):
    '''
    load a small set of tensors for object detection cnn
    
    params:
    input_dir, directory with input tensors
    label_dir, directory with label tensors
    
    returns:
    inputs, list of input tensors
    labels, list of label tensors
    '''
    # build directory lists
    input_entries = [i for i in os.listdir(input_dir) if i[0] == 't']
    label_entries = os.listdir(label_dir)
    input_entries.sort()
    label_entries.sort()
    
    # load inputs
    input_tensors = []
    for i in input_entries:
        input_tensors.append(torch.load(input_dir + '/' + i))
        
    # load labels
    label_tensors = []
    for i in label_entries:
        label_tensors.append(torch.load(label_dir + '/' + i))
        
    return input_tensors, label_tensors

def run_model(input_tensors, label_tensors, net, epochs, lr, momentum, modelDir, device):
    print()
    print("STARTING MODEL TRAINING.")
    # define loss fn and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    i = 0
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in zip(input_tensors, label_tensors):
            # get the inputs; data is a list of [inputs, labels]
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
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
            i += 1

    print('Finished Training, pog')
    
    print('Saving model to: {}'.format(modelDir))
    torch.save(net.state_dict(), modelDir)

def test_model(net, test_inputs, test_labels, device):
    # accuracy test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in zip(test_inputs, test_labels):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10 test images: %d %%' % (
        100 * correct / total))
    