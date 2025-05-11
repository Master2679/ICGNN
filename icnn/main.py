import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from mlp import MLP
#from cnn_lenet import LeNet_MNIST
from icnn import ICNN


def train_loop(dataloader, model, loss_fn, optimizer):
    """ Fetches batches from dataloader and runs backpropagation on them

    Args:
        dataloader: torch.utils.data.Dataloader() instance
        model: Neural network model
        loss_fn: particular loss function to use
        optimizer: particular optimizer to use

    Prints:
        Loss
    """
    size = len(dataloader.dataset)
    device = next(model.parameters()).device
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #compute prediciton and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Key Code guaranteeing nonegative Weight Matrix Below
        # Note that biases of the below associated linear transformations
        # as well as weights and biases and skip connections are not
        # constrained
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "prim" in name: 
                    param.copy_(param.clamp(min=0, max=None))
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    """ Fetches batches from test data dataloader and fowards them through net

    Args:
        dataloader: torch.utils.data.Dataloader() instance
        model: Neural network model
        loss_fn: particular loss function to use

    Prints:
        Test Error and Average Loss
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def call_train_loop(epochs, train_loader, test_loader, model, loss_fn, optimizer):
    """ Trains for epochs number of Epochs

    Args:
        epochs: number of Epochs
        train_loader: dataloader for training set
        test_loader: dataloader for test set
        model: particular nn model to train
        loss_fn: loss function to be used in training and testing
        optimizer: optimizer to be used during training
    
    Prints:
        Epochs and Done!
    """

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")

    return model


#def train_icnn(learning_rate, epochs):
    """Trains models (in particular model of ICNN())

    Configures the torch.device() to cuda if available
    Creates training dataset dataloader
    Creates testing dataset dataloader
    Creates model instance from appropriate class
    Creates loss_fn instance
    Creates optimizer instance

    calls call_train_loop() using the above

    Args:
        learning_rate: the learning rate for the optimizer
        epochs: the number of epochs to train for    
    """
    """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root = ".", train = True, download = True, transform = 
            transforms.ToTensor()),
        batch_size = 64, shuffle = True, num_workers = 4)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform = 
            transforms.ToTensor()),
        batch_size=64, shuffle=True, num_workers=4)
    
    
    #MODEL
    model = ICNN()

    # loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # trained_model = call_train_loop(epochs, train_loader, test_loader, model, loss_fn, optimizer)

    return model"""


"""def print_trained_models_weights(model):
    for name, parameters in model.named_parameters():
        print(f"{name}: {parameters}")"""


def main():
    # Set device to MPS if available, otherwise fallback to CPU
    #if torch.backends.mps.is_available():
     #   device = torch.device("mps")
    #    print("Using MPS device")
   # else:
    device = torch.device("cpu")
    print("Using CPU device")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = ICNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    epochs = 10
    call_train_loop(epochs, train_loader, test_loader, model, loss_fn, optimizer)

if __name__ == "__main__":
    main()