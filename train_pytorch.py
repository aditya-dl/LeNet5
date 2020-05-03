from model_pytorch import Net
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

IF_DOWNLOAD = False
ROOT_DIR = r"D:\MachineLearning\Datasets\MNIST"
BATCH_SIZE = 64

# ToTensor() converts the image into numbers i.e, RGB values are normalised between 0 and 255.
# Normalize() normalizes the tensor with a mean and standard deviation which goes as the two parameters respectively.
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
dataset = MNIST(root=ROOT_DIR, download=IF_DOWNLOAD, transform=transform)
train, test = dataset.train_data, dataset.test_data
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)


net = Net()
