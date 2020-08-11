from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch import save

from model_pytorch import LeNet5
from .dataloader import get_mnist


# check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
net = LeNet5()
# if gpu is available, load model to gpu
net.to(device)

# define criterion
criterion = CrossEntropyLoss()
# define optimizer
optimizer = optim.Adam(net.parameters(), lr=2e-3)
# define epochs
epochs = 10

# get data
train_loader, test_loader = get_mnist()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print(">>> Finished Training")

# PATH to save our model
PATH = "./lenet5.pth"

# save model
save(net.state_dict(), PATH)
