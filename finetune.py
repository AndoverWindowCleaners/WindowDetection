from tools.data_reading import CompressedWindowDataset
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tools import utils
from tools.engine import train_one_epoch, evaluate
from model_zoo.InputInjection import InputInjection
from torch.utils.tensorboard import SummaryWriter

dataset = CompressedWindowDataset()
train = DataLoader(dataset,batch_size = 1, shuffle=True, collate_fn=utils.collate_fn)

model = InputInjection()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'running on {device}')

model.to(device)
writer = SummaryWriter()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params,lr=0.00003)

print('starting...')
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(writer, model, optimizer, train, device, epoch, print_freq=1)
    evaluate(writer, model, train, device, print_freq=20)
    torch.save(model.state_dict(), f'inputinjection-train-1.weights')