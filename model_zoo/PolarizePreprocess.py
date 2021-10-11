import torch
from torch import nn

class PolarizePreprocess(nn.Module):
    def __init__(self):
        # input is of shape (12,9,3*8)
        # we should upsample that to (128,96,2) or (128,96,1)
        super(PolarizePreprocess, self).__init__()
        self.conv1 = nn.Conv2d(24, 72, (3,3), padding='same') # gather info of neighbor cells
        self.upconv1 = nn.ConvTranspose2d(72, 36, (7,9), stride=3, padding=0)
        self.upconv2 = nn.ConvTranspose2d(36, 36, (8,7), stride=3, padding=1) # this should scale up to 128, 96
        # set up depthwise if needed
        self.conv2 = nn.Conv2d(36, 128, (1,1)) # blow up channels for classification
        self.conv3 = nn.Conv2d(128, 2, (1,1)) # final classifier
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def save(self, file_name="PolarizePreprocess.weights"):
        torch.save(self.state_dict(), file_name)

    def forward(self, x):
        '''
        x is expected to be (batch, channel, h, w)
        '''
        x = self.relu(self.conv1(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x)) # try with tanh and relu and sigmoid
        return x	

model = PolarizePreprocess()
model.eval()
x = torch.zeros((1, 24, 9, 12))
y = model(x)
print(y.shape)