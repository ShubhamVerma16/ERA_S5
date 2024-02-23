# Imports
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm

# The utility class with the required utilitty functions
class Utils():
  # Initialised variables
  def __init__(self):
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    self.criterion = F.nll_loss
    self.num_epochs = 20
    self.batch_size = 512
    self.num_workers = 2
    self.shuffle = True
    self.pin_memory = True
    self.train_losses = []
    self.test_losses = []
    self.train_acc = []
    self.test_acc = []
    self.num_rows = 5
    self.test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

  # Function to get the train and test transformations
  def get_transform(self):
    """
    Gets the train and test transformation.
 
    Args:
        None
 
    Returns:
        list: list of train transforms to compose.
        list: list of test transforms to compose
    """

    # Train data transformations
    train_transforms = transforms.Compose([ # clubs all the transforms provided to it. So, all the transforms in the transforms.Compose are applied to the input one by one.
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1), # Ramdomapply applies list of tranforms to the images with prob=p.
                                            # centercrop Crops the center part of the image of shape (22, 22)
        transforms.Resize((28, 28)), # input image is resized to be of size (18, 28)
        transforms.RandomRotation((-15., 15.), fill=0), # Rotate the image randomly between min and max degree of angle and pixel fill = 0
        transforms.ToTensor(), # This just converts input image to PyTorch tensor.
        transforms.Normalize((0.1307,), (0.3081,)), # input data scaling and these values (mean and std) are precomputed for dataset
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(), # This just converts input image to PyTorch tensor.
        transforms.Normalize((0.1307,), (0.3081,)) # input data scaling and these values (mean and std) are precomputed for dataset
        ])

    return train_transforms, test_transforms

  # Function to get the MNIST train and test data 
  def get_data(self, train_transforms, test_transforms): 
    """
    Gets the MNIST train and test data.
 
    Args:
        list: list of train transforms to compose.
        list: list of test transforms to compose
 
    Returns:
        list: list of train data
        list: list of test data
    """
    # download the train data to data folder and apply train trnasforms to the data
    train_data = datasets.MNIST('../data', 
                                train=True, download=True, 
                                transform=train_transforms)
    # download the test data to data folder and apply train trnasforms to the data
    test_data = datasets.MNIST('../data', 
                              train=False, download=True, 
                              transform=test_transforms)

    return train_data, test_data

  def get_data_loader(self, train_data, test_data):
    """
    Gets the MNIST train and test dataloader.
 
    Args:
        list: list of train data
        list: list of test data
 
    Returns:
        object: train data dataloader object
        object: test data dataloader object
    """

    kwargs = {'batch_size': self.batch_size, 
              'shuffle': self.shuffle, 
              'num_workers': self.num_workers, 
              'pin_memory': self.pin_memory}
    # Dataloader  represents a Python iterable over a dataset
    # calls individual fetched data samples into batches via arguments batch_size
    # num_workers sets multi-process data loading with the specified number of loader worker processes
    # Host to GPU copies are much faster when they originate from pinned (page-locked) memory
    # Hence we use automatic memory pinning which enables fast data transfer to CUDA-enabled GPUs
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

    return train_loader, test_loader

  def get_model_summary(self, model):
    """
    Gets the MNIST train and test data.
 
    Args:
        object: Model class object
 
    Returns:
        Model summary
    """
    return summary(model, input_size=(1, 28, 28)) # get model summary

  # Function to get count of correct predictions
  def GetCorrectPredCount(self, pPrediction, pLabels):
    """
    Gets the correct prediction count for each epoch.
 
    Args:
        tensor: Modle predictions
        tensor: Correct Output labels
 
    Returns:
        int: count of corret model outputs 
    """

    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

  # function to train the model
  def train(self, model, device, train_loader, optimizer, criterion):
    """
    Trains the model.
 
    Args:
        object: Model class object
        string: device available (CPU / GPU)
        object: data loader object for train data
        object: training otpimizer object
        object: loss function criterion
 
    Returns:
        None
    """

    model.train() # set model to train state
    pbar = tqdm(train_loader) # load the dataset, tqdm shows progress bar

    # initialise train_loss, correct and processed data
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar): # enmerate data to get index and data, target pair
      data, target = data.to(device), target.to(device) # move data to device
      optimizer.zero_grad() # initialize the optimiers

      # Predict
      pred = model(data)

      # Calculate loss
      loss = criterion(pred, target) # calculate loss based on the creterion=loss function
      train_loss+=loss.item() # add loss value to the list

      # Backpropagation
      loss.backward() # perofrom backpropagation
      optimizer.step() # update the parameters

      correct += self.GetCorrectPredCount(pred, target) # update the correct data
      processed += len(data) # update the process value

      pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    self.train_acc.append(100*correct/processed) # update the train accouracy for each epoch
    self.train_losses.append(train_loss/len(train_loader)) # update the train loss for each epoch

  # Function to Test the model
  def test(self, model, device, test_loader, criterion):
    """
    Tests the model.
 
    Args:
        object: Model class object
        string: device available (CPU / GPU)
        object: data loader object for test data
        object: loss function criterion
 
    Returns:
        None
    """
    model.eval() # set model in eval mode

    # initialize test_loss and correct value
    test_loss = 0
    correct = 0

    with torch.no_grad(): # without maitaining gradinet for the parameters, this prevents training or paramatere updated during test phase
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += self.GetCorrectPredCount(output, target) # update correct value


    test_loss /= len(test_loader.dataset) # calculate test loss
    self.test_acc.append(100. * correct / len(test_loader.dataset)) # update test accruracy values per epoch
    self.test_losses.append(test_loss) # update test loss values per epoch

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

  # Function to get the training optimizer
  def get_optimizer(self, model):
    """
      retunrn the optimizer object for training.
  
      Args:
          Object: Model
  
      Returns:
          Object: optimizer object
    """
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # set SDG optimizer with model paramtters, learning rate and momentum

  # Function to get the training scheduler
  def get_scheduler(self, optimizer):
    """
      return the scheduler object for training.
  
      Args:
          Object: training Optimiser/
  
      Returns:
          Object: Scheduler object
    """
    return optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True) # set learning rate scheduler with optimizer, step size, gamma and verbose give logs

  # Funtion to plot the input images in tabluar format
  def plot_input_data(self, images, num_images):
    """
      Plot the images in the tabular format.
  
      Args:
          None
  
      Returns:
          None
      """
    import matplotlib.pyplot as plt
    # train loader is a dataloader class which returns an iterable, so we use net(iter([iterale]))
    batch_data, batch_label = next(iter(images))

    fig = plt.figure()

    for i in range(num_images):
      plt.subplot(self.num_rows, int(num_images / self.num_rows),i+1) # create a 3, 4 matrix and 3rd argument is the index of the current plot
      plt.tight_layout() # ight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
      # it will also adjust spacing between subplots to minimize the overlaps
      plt.imshow(batch_data[i].squeeze(0), cmap='gray') # show the data as an image; cmap is colormap instance or registered colormap name, Returns a tensor with all the dimensions of input of size 1 removed.
      plt.title(batch_label[i].item()) # plot title as label
      plt.xticks([]) # empty xticks removes xticks from the plot
      plt.yticks([]) # empty yticks removes xticks from the plot

  # Plot the learning curves
  def plot_learning_curve(self):
    """
    Plot the training and test loss, training and test accuracy.
 
    Args:
        None
 
    Returns:
        None
    """
    
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(self.train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(self.train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(self.test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(self.test_acc)
    axs[1, 1].set_title("Test Accuracy")
