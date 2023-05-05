import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import wandb
import torch
import torch.nn as nn

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
wandb.init(project="pytorch_cifar10")

# Set up the CIFAR-10 dataset and data loader
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./remove/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

# Define your model, loss function, and optimizer
model = YourModel() # Replace with your model
model.to('cuda:0')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Watch the model
wandb.watch(model)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # Log metrics after each epoch
    wandb.log({"epoch_loss": running_loss / (i+1)})

# Save the trained model and finish the run
torch.save(model.state_dict(), "./remove/cifar10_model.pth")
wandb.save("./remove/cifar10_model.pth")
wandb.finish()
