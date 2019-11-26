import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(nn.Conv2d(1, 64, 3),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(64),
                                            nn.MaxPool2d(2),
                                            nn.Conv2d(64, 128, 3),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(128),
                                            nn.MaxPool2d(2))

        self.classifier = nn.Sequential(nn.Linear(25 * 128, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 10))

    def forward(self, x):
        x = self.feat_extractor(x)
        n, c, h, w = x.shape
        x = x.view(n, -1)
        x = self.classifier(x)
        return x


class Trainer(object):
    def __init__(self, model, device, train_loader, test_loader, criterion,
                 optimizer):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.softmax = nn.Softmax(dim=1)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epochs=10):
        print('Initial accuracy: {}'.format(self.evaluate()))
        for epoch in range(1, epochs + 1):
            self.model.train()  # set the model to training mode
            for images, labels in tqdm(self.train_loader,
                                       total=len(self.train_loader)):
                self.optimizer.zero_grad()  # don't forget this line!
                images, labels = images.to(self.device), labels.to(self.device)

                output = self.softmax(self.model(images))
                loss = self.criterion(output, labels)
                loss.backward()  # compute the derivatives of the model
                optimizer.step()  # update weights according to the optimizer

            print('\nAccuracy at epoch {}: {}'.format(epoch, self.evaluate()))

    def evaluate(self):
        self.model.eval()  # set the model to eval mode
        total = 0

        for images, labels in tqdm(self.test_loader,
                                   total=len(self.test_loader)):
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.softmax(self.model(images))
            predicted = torch.max(output, dim=1)[1]  # argmax the output
            total += (predicted == labels).sum().item()

        return total / len(self.test_loader.dataset)


if __name__ == '__main__':
    config = {'lr': 1e-4,
              'momentum': 0.9,
              'weight_decay': 0.001,
              'batch_size': 8,
              'epochs': 10,
              'device': 'cuda:0',
              'seed': 314}

    # set the seeds to repeat the experiments
    if 'cuda' in config['device']:
        torch.cuda.manual_seed_all(config['seed'])
    else:
        torch.manual_seed(config['seed'])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    batch_size = 8
    train_loader = DataLoader(
        torchvision.datasets.MNIST('./mnist', download=True, train=True,
                                   transform=transform),
        batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(
        torchvision.datasets.MNIST('./mnist', download=True, train=False,
                                   transform=transform),
        batch_size=config['batch_size'], shuffle=False)

    trainer = Trainer(model, config['device'], train_loader, test_loader,
                      criterion, optimizer)
    trainer.train(epochs=config['epochs'])
