import torch
import time
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net
import json
import os

def test(model, data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss / 10000, correct / 10000  # 10000 samples in fashion mnist test


def train(model, data_loader, optimizer):
    train_loss = 0
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    # 60000 samples in fashion mnist train
    return train_loss / 60000, correct / 60000

if __name__ == '__main__':
    experiments = {'pooling': [torch.nn.AvgPool2d, torch.nn.MaxPool2d], 'batch_size': [32, 64, 128], 'with_bn': [True, False], 'with_dropout': [True, False]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.FashionMNIST(
        'data', train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(
        'data', train=False, transform=transform, download=True)

    results = []
    counter = 0

    for pooling in experiments['pooling']:
        for batch_size in experiments['batch_size']:
            for with_bn in experiments['with_bn']:
                for with_dropout in experiments['with_dropout']:
                    train_load = DataLoader(
                        train_data, shuffle=True, batch_size=batch_size)
                    test_load = DataLoader(
                        test_data, shuffle=False, batch_size=batch_size)

                    model = Net(pooling=pooling, with_bn=with_bn,
                                with_dropout=with_dropout).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    early_stopping_counter = 0
                    early_stopping = 99999
                    current_acc = 0
                    if not os.path.exists('models/{}'.format(counter)):
                        os.mkdir('models/{}'.format(counter))
                    epoch = 1
                    best_epoch = 1
                    start_time = time.time()
                    train_losses = []
                    train_accs = []
                    test_losses = []
                    test_accs = []
                    while True:
                        epoch_time = time.time()
                        train_loss, train_acc = train(model, train_load, optimizer)
                        train_time = time.time() - epoch_time
                        current_time = time.time() - start_time
                        test_loss, test_acc = test(model, test_load)
                        train_losses.append(train_loss)
                        test_losses.append(test_loss)
                        train_accs.append(train_acc)
                        test_accs.append(test_acc)
                        print('Epoch: %d;\t Train Loss: %.4e;\t Test Loss: %.4e;\t Train Acc: %.4f;\t Test Acc: %.4f;\t Time Taken: %.2f;\t Total Time Taken: %.2f' % (
                            epoch, train_loss, test_loss, train_acc, test_acc, train_time, current_time))
                        if test_loss > early_stopping:
                            early_stopping_counter += 1
                        else:
                            early_stopping_counter = 0
                            early_stopping = test_loss
                            best_epoch = epoch
                            torch.save(model, 'models/{}/best.pt'.format(counter))
                        if early_stopping_counter > 3:
                            finish_time = time.time() - start_time
                            results.append({'finish_loss': early_stopping,
                                            'train_losses': train_losses,
                                            'test_losses': test_losses,
                                            'test_accs': test_accs,
                                            'train_accs': train_accs,
                                            'acc': current_acc,
                                            'pooling': 'avg' if pooling is torch.nn.AvgPool2d else 'max',
                                            'with_dropout': with_dropout,
                                            'with_bn': with_bn,
                                            'batch_size': batch_size,
                                            'best_epoch': best_epoch,
                                            'num_epochs': epoch,
                                            'total_time': finish_time})
                            print('Loss exceeded previous loss, early stopping')
                            break
                        current_acc = test_acc
                        epoch += 1
                    counter += 1
    with open('results.txt', 'w') as file:
        file.write(json.dumps(results))
