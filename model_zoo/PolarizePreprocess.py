import torch
from torch import nn

class PolarizePreprocess(nn.Module):
    def __init__(self):
        # video shape is 360 by 640 (W by H)
        # input is of shape (9,16,3*8) maybe another larger shape (18,32,3*8)
        # we should upsample that to (360,640,2)
        super(PolarizePreprocess, self).__init__()
        self.conv1 = nn.Conv2d(24, 72, (3,3), padding='same') # gather info of neighbor cells
        self.upconv1 = nn.ConvTranspose2d(72, 36, (4,3), stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose2d(36, 36, (6,5), stride=3, padding=0) 
        self.upconv3 = nn.ConvTranspose2d(36, 36, (16,12), stride=6, padding=0) 
        # set up depthwise if needed
        self.conv2 = nn.Conv2d(36, 128, (1,1)) # blow up channels for classification
        self.conv3 = nn.Conv2d(128, 2, (1,1)) # final classifier
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # maybe a good idea to use scale up module and back connections
    
    def save(self, file_name="PolarizePreprocess.weights"):
        torch.save(self.state_dict(), file_name)

    def forward(self, x):
        '''
        x is expected to be (batch, channel, h, w)
        '''
        x = self.relu(self.conv1(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x)) # try with tanh and relu and sigmoid
        return x	

# model = PolarizePreprocess()

# model.eval()

# a = torch.zeros((1,24,16,9))
# p = model(a)
# print(p.shape)