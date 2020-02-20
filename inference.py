import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.FashionMNIST('data', transform=transforms, train=False, download=True)
    times = []
    try:
        model = torch.load('models/0/best.pt', map_location='cuda0')
    except:
        print('No model found')
        exit()
    data_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for d, _ in data_loader:
            start = time.time()
            d = d.to(device)
            output = model(d)
            times.append(time.time() - start)
    print(sum(times) / len(times))
            

