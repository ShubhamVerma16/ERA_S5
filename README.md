# ERA_S5: Implementation of CNN model on MNIST Dataset

## Get Started
- the notebook has been mounted to get access to the model and utils files.
  <pre>drive.mount('/content/drive/', force_remount=True)</pre>  
- Once mount is complete, the directory is changed to where the files are present.
  <pre>%cd /content/drive/[required_dir]</pre>
- ERA_S5.ipynb python notebook includs the implmentation as the main code. The file imports the other 2 modules and calls the required class and its methods.
- utils.py includes Utils class with the utility methods required for the model to train and to load the data.
- model.py incldes the model class with the CNN architecture and forward pass defined.

For the data set we are using the MNIST dataset with Mean=0.1307 and STD=0.3081.

![image](https://github.com/ShubhamVerma16/ERA_S5/assets/46774613/2aaa00a0-daa0-4a58-a038-d245d93664d4)


Train data is applied with the below transforms.
* CenterCrop
* Image resize to (28,28)
* random Rotation
* The data is converted to tensors
* Data is normalised to Mean and STD of the dataset.

Test data is applied with the below transforms.
* The data is converted to tensors
* Data is normalised to Mean and STD of the dataset.

The model is a CNN based model with 2D convolutions and fully connected layers.

![image](https://github.com/ShubhamVerma16/ERA_S5/assets/46774613/9c488a5f-e581-4306-a422-99e2cf9b1a79)

The repositior follows the below folder strucutre.
<pre>ERA_S5
  |── ERA_S5.ipynb
  |── model.py
  |── utils.py
  |── README.md
  |── .gitignore
</pre>


Run the ERA_S5.ipynb file to go through the entire process of loading hte dataset and training on the CNN model defined.


