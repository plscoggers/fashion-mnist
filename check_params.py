'''
Utility file to count the number paramters in a model
'''

import torch

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    try:
        model = torch.load('models/0/best.pt', map_location='cuda0')
        print(model)
        print(count_params(model))
    except:
        print('No model found')