from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch
import argparse
import sys

sys.path.append("..")

from model_pytorch import LeNet5
from dataloader import get_mnist

# check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(pool, activation):
    # load model
    net = LeNet5(pool=pool, activation=activation)
    # if gpu is available, load model to gpu
    net.to(device)

    return net


def load_optimizer(parameters):
    # define criterion
    criterion = CrossEntropyLoss()
    # define optimizer
    optimizer = optim.Adam(parameters, lr=2e-3)

    return criterion, optimizer


def train(net, train_loader, epochs, optimizer, criterion):
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
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print(">>> Finished Training")


def save_model(net, save_path):
    # save model
    torch.save(net.state_dict(), save_path)


def evaluate(net, test_loader):
    # evaluate our model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
            100 * correct / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeNet5 architecture in PyTorch.")
    parser.add_argument("-p", "--pool", default="max", help="avg pool or max pool?")
    parser.add_argument("-a", "--activation", default="relu", help="choose from relu, tanh, and leaky")
    parser.add_argument("-b", "--batchsize", type=int, default=256, help="Batch size for training/testing data.")
    parser.add_argument("-d", "--data", default="./data/mnist", help="Path to data.")
    parser.add_argument("-s", "--shuffle", default="True", help="Shuffle data or not?")
    parser.add_argument("-n", "--numworker", type=int, default=8,
                        help="Number of workers to divide the batching process.")
    parser.add_argument("-m", "--modelsavepath", default="./lenet5.pth", help="Model save path.")
    parser.add_argument("-l", "--learningrate", type=float, default=2e-3, help="Set learning rate.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Set number of epochs.")
    parser = vars(parser.parse_args())

    model = load_model(parser["pool"], parser["activation"])
    criterion, optimizer = load_optimizer(model.parameters())
    epochs = parser["epochs"]

    train_loader, test_loader = get_mnist(path=parser["data"], batch_size=parser["batchsize"], shuffle=parser["shuffle"]
                                          , num_workers=parser["numworker"])

    train(model, train_loader, epochs, optimizer, criterion)

    save_model(model, parser["modelsavepath"])

    evaluate(model, test_loader)
