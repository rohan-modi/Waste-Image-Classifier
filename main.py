from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import random

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts the images to PyTorch tensor format
    # Normalize the images (optional, if you need to display, you might skip it)
])

dataset = datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/Garbage/GarbageClasses', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

counter = 0

num_classes = len(dataset.classes)
class_counts = [0] * num_classes
for _, label in dataset.samples:
    class_counts[label] += 1

train_size = sum(class_counts) // 3
val_size = sum(class_counts) // 3
test_size = sum(class_counts) - train_size - val_size

indices = [[] for _ in range(num_classes)]
for idx, (_, label) in enumerate(dataset.samples):
    indices[label].append(idx)

train_indices, val_indices, test_indices = [], [], []

for i in range(num_classes):
    np.random.shuffle(indices[i])
    train_indices.extend(indices[i][:train_size // num_classes])
    val_indices.extend(indices[i][train_size // num_classes:(train_size // num_classes + val_size // num_classes)])
    test_indices.extend(indices[i][(train_size // num_classes + val_size // num_classes):])

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

loaders = [train_loader, val_loader, test_loader]
loaderCounter = 0

def train_net(net, num_epochs=30, learningRate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.classifier.parameters(), lr=learningRate, momentum=0.9)

    for epoch in range(num_epochs):
        print("Starting epoch", epoch)
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        net.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # No need to track gradients
            for inputs, labels in val_loader:
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Accumulate the validation loss
                val_loss += loss.item()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')

        epoch_val_loss = val_loss / len(val_loader)
        print("Training loss:", running_loss, "Validation loss:", epoch_val_loss)

mobilenet = models.mobilenet_v2(pretrained=True)

# Freeze all layers except the final layer
for param in mobilenet.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
mobilenet.classifier = nn.Sequential(
    nn.Linear(mobilenet.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
# Train the model
train_net(mobilenet)

classNames = {0: "Battery", 1: "Biological", 2: "Brown glass", 3: "Cardboard", 4: "Clothes", 5: "Green glass", 6: "Metal", 7: "Paper", 8: "Plastic", 9: "Shoes", 10: "Trash", 11: "White glass"}

def predict_single_image(net, dataloader):
    # Assuming the first image in the dataloader is used for prediction
    for images, labels in dataloader:

        label = labels[0].item()
        if classNames[label] == "Clothes":
            if (random.randint(1, 20) < 19):
                continue
        if classNames[label] == "Shoes":
            if (random.randint(1, 4) < 3):
                continue
        # print("Actual class:", classNames[label])

        transform2 = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
        ])

        for image in images:
            image = transform2(image)
            image = image.unsqueeze(0)

        # print("Size", images.size())

        net.eval()
        with torch.no_grad():
            output = net(images)
            # print("The size of output is", output.size())
            maxValue = max(output[0])
            for i in range(len(output[0])):
                if output[0][i] == maxValue:
                    print("Predicted class:", classNames[i])

        # Display the image

        cpuTensor = images[0].cpu()
        drawImage = cpuTensor.squeeze(0).numpy()

        drawImage = np.transpose(drawImage, (1, 2, 0))  # Change dimensions from CxHxW to HxWxC

        plt.imshow(drawImage)
        plt.show()

# Predict the output for a random image from the DataLoader
predicted_class = predict_single_image(mobilenet, test_loader)
