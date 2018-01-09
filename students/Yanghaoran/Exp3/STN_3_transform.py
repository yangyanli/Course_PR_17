

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

plt.ion()  




train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.ReLU(True),  #change
	    nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),  #change 
	    nn.MaxPool2d(2, stride=2)
           
        )
		
        self.localization2=nn.Sequential(
	    nn.Conv2d(10,20,kernel_size=5),
	    nn.ReLU(True),
	    nn.MaxPool2d(2,stride=2)  #4*4*20
	)
		
	

    
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
		
        self.fc_loc2=nn.Sequential(
            
	    nn.Linear(20 * 4 * 4,3*2)
	   # nn.ReLU(True),
	   # nn.Linear(64,10*3*2)
	)
		
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
		
       # self.fc_loc2[0].weight.data.fill_(0)
       # self.fc_loc2[0].weight.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
	
    def stn2(self,x):
        xs=self.localization2(x)
       # print(xs.size())
        xs=xs.view(-1,20*4*4)
       # print(xs.size())
        theta=self.fc_loc2(xs)

       # print(theta.size())
        theta=theta.view(-1,2,3)
       # print("theta:",theta.size())
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
	
    def forward(self, x):
        # transform the input
		# 28*28
        x = self.stn(x)  #28*28

        # Perform the usual forward pass
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 12*12*10
       # print(x.size())		
        x=self.stn2(x) #12*12*10
		
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()


optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn():
   
    data, _ = next(iter(test_loader))
    data = Variable(data, volatile=True)

    if use_cuda:
        data = data.cuda()

    input_tensor = data.cpu().data
    transformed_input_tensor = model.stn(data).cpu().data
    print(data.size())
	
    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))

    out_grid = convert_image_np(
        torchvision.utils.make_grid(transformed_input_tensor))

    
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(in_grid)
    axarr[0].set_title('Dataset Images')

    axarr[1].imshow(out_grid)
    axarr[1].set_title('Transformed Images')
    plt.savefig("resultmain3.png")
    x=model.stn(data)
    x= F.max_pool2d(F.relu(model.conv1(x)), 2)
    y=x[0]
    print("y",y.size())
    y=y.view(10,1,12,12)
    print("y",y.size())	
    x=model.stn2(x) 
    print("x[0]",x[0].size())
    y2=x[0].view(10,1,12,12)
    input_tensor2=y.cpu().data
    trans_tensor2=y2.cpu().data
    
    in_grid2 = convert_image_np(
        torchvision.utils.make_grid(input_tensor2))

    out_grid2 = convert_image_np(
        torchvision.utils.make_grid(trans_tensor2))
    
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(in_grid2)
    axarr[0].set_title('Dataset Images')

    axarr[1].imshow(out_grid2)
    axarr[1].set_title('Transformed Images')
    plt.savefig("result2main3.png")		
	
for epoch in range(1, 20+1):
    train(epoch)
    test()
torch.save(model, "model.pkl")
model=torch.load("model.pkl")

visualize_stn()

