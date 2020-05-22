import torch
import torch.utils.data as data

from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from classifier import simpleClassifier
from data_loader import dataloader


def train(model, train_loader, criterion, optimizer, epoch):
    """
    Train function
    :param model: nn.Module, a Module of a learning model
    :param train_loader: DataLoader, a data loader of the training dataset
    :param criterion: loss function
    :param optimizer: optimizer
    :param epoch: int, current epoch of training process
    :return: None
    """
    model.train()
    train_loss = 0
    for batch_i, (imgs, targets) in enumerate(train_loader):
        imgs, targets = Variable(imgs), Variable(targets)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('\rBatch {}/{}: current loss: {:.6f}'.format(batch_i, len(train_loader), loss.item()), end='', flush=True)
    print('\rEpoch {}: loss: {:.6f}'.format(epoch + 1, train_loss / len(train_loader)),
          end='', flush=True)


def test(model, test_loader, criterion, epoch):
    """
    Test function
    :param model: nn.Module, a trained learning model
    :param test_loader: DataLoader, a data loader of the testing dataset
    :param criterion: loss function
    :param epoch: current epoch of the training process
    :return: None
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = Variable(imgs), Variable(targets)
            outputs = model(imgs)
            test_loss += criterion(outputs, targets)
            outputs = log_softmax(outputs, dim=1)
            pred = torch.max(outputs, dim=1)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('Epoch {}: Test Loss: {:.4f}, Test Accuracy: {}'.format(epoch + 1, test_loss, acc))


if __name__ == '__main__':
    # Set data path
    img_path = './train-images-idx3-ubyte'
    label_path = './train-labels-idx1-ubyte'
    test_img_path = './t10k-images-idx3-ubyte'
    test_label_path = './t10k-labels-idx1-ubyte'
    # Initiate data loader
    train_data = data.DataLoader(dataloader(img_path, label_path), batch_size=256, shuffle=True)
    test_data = data.DataLoader(dataloader(test_img_path, test_label_path), batch_size=256, shuffle=True)
    # Initiate training parameters
    # model = simpleClassifier()
    model = torch.load('./simpleClassifier_20.pth')
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()
    epochs = 20
    # Train
    for epoch in range(epochs):
        train(model, train_data, criterion=criterion, optimizer=optimizer, epoch=epoch)
        # Save the model per 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model, './simpleClassifier_{}.pth'.format(epoch + 21))

        test(model, test_data, criterion, epoch)
