import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import wandb
import torch
import torch.nn as nn
from datetime import timedelta
import wandb
from wandb import AlertLevel

import numpy as np



class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Fully connected layer, 16*5*5 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)         # Fully connected layer, 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 10)          # Fully connected layer, 84 input features, 10 output features (10 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Log in to WandB
wandb.login()

# Initialize a new run
wandb.init(
    project="pytorch_cifar10",
    config=None,
    name='justin',
    dir=None
    )

# Set up the CIFAR-10 dataset and data loader
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor()])

val_transform = transforms.Compose(
    [transforms.ToTensor()])
     
trainset = torchvision.datasets.CIFAR10(root='./remove/data', train=True, download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='./remove/data', train=False, download=True, transform=val_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)

# Define your model, loss function, and optimizer
model = YourModel() # Replace with your model
model.to('cuda:0')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Watch the model
wandb.watch(model)

# Train the model
num_epochs = 10
examples = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader):
        total += inputs.shape[0]
        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = torch.argmax(outputs.detach().cpu(), dim=1)
        running_acc += (pred == labels.cpu()).sum()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
    # Log metrics after each epoch


    image = wandb.Image(np.array(inputs.cpu()[0].permute(1,2,0)), caption=f"random field {epoch}")
    examples.append(image)
    wandb.log({
        "epoch_loss": running_loss / (i+1),
        "epoch_acc" : running_acc / total,
        "example" : examples
    })
    if (running_acc / total) > 0.2:
        wandb.alert(
            title=f'accuracy_{epoch}',
            text=f'Accuracy {running_acc / total} is got to 0.2',
            level=AlertLevel.INFO,
            # wait_duration=timedelta(seconds=5)
        )
    print(f'Epoch loss: {running_loss:0.4f} | Epoch Acc: {running_acc / total:0.4f}')
# Save the trained model and finish the run
torch.save(model.state_dict(), "./remove/cifar10_model.pth")
wandb.save("./remove/cifar10_model.pth")
wandb.finish()
