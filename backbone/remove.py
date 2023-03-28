import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.multiprocessing as mp
import torch.distributed as dist


def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    model = resnet18(num_classes=10).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"GPU {rank}, Epoch {epoch + 1}, Iteration {i + 1}, Model Weights:")
                for name, param in model.named_parameters():
                    print(f"{name}: {param.data[0][0][0]}")
                    break

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss / (i + 1)}")

    if rank == 0:
        print("Training has finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()