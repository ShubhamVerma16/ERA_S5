# ERA_S5: Implementation of CNN model on MNIST Dataset

## Get Started
- S5.ipynb python notebook includs the implmentation as the main code. The file imports the other 2 modules and calls the required class and its methods.
- utils.py includes Utils class with the utility methods required for the model to train and to load the data.
- model.py incldes the model class with the CNN architecture and forward pass defined.

For the data set we are using the MNIST dataset with Mean=0.1307 and STD=0.3081.
Train data is applied with the below transforms.
* CenterCrop
* Image resize to (28,28)
* random Rotation
* The data is converted to tensors
* Data is normalised to Mean and STD of the dataset.

Test data is applied with the below transforms.
* The data is converted to tensors
* Data is normalised to Mean and STD of the dataset.

The repositior follows the below folder strucutre.
<pre>ERA_S5
  |── S5.ipynb
  |── model.py
  |── utils.py
  |── README.md

Run the S5.ipynb file to go through the entire process of loading hte dataset and training on the CNN model defined.


