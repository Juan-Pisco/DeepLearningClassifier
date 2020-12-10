# AI Programming with Python Project - DeepLearning Classifier

Project code for Udacity's AI Programming with Python Nanodegree program. This code is used for classifying images in the `predict.py` file with the model 
you trained with resnet, alexnet or vgg previously on the `train.py` file. In addition, there's a trainning, testing and plotting exercise developed into a
notebook at the `Image_Classifier_Proyect.ipynb`.

# Packages Needed

For downloading and testing out this proyect, you'll have to make sure you've installed the following python packages:
*Pytorch (at least 1.6.0)
* argparse
* json
* PIL
* matplotlib
* seaborn

# Usage examples

* Open the `train.py` file in the terminal using the following sintax: (Just 1 line of code)
`python train.py --save_dir (path to the checkpoints.pth) --arch (choose the CNN model you want to use between "resnet)
--learning_rate (learning rate value) --hidden_units (hidden units number) --epochs (number of epochs) 
--gpu (if --gpu is written, enables gpu usage) `

* Open the `predict.py` file in the terminal using the following sintax: (Just 1 line of code)
`python predict.py input_image_pathfile checkpoint.pth_filepath --top_k (int of how many probability results are shown)
--category_names (json filepath to convert from categories index to labels/names) --gpu (if --gpu is written, enables gpu usage)`

* When openning the `Image_Classifier_Proyect.ipynb` file, run every cell one-by-one.

# Results

* For `Image_Classifier_Proyect.ipynb` you should get this kind of results:

  * Training using resnet50:
  
    * Epochs: 5
    * LearningRate: 0.003
    * Hidden-units: 1024
    * Dropout: 0.2
    
    * Validation accuracy result: 83.8%
    * Validation loss result: 0.574
    * Trainning loss result: 1.157
    
    * Testing accuracy result: 84.1%
    * Testing loss result: 0.586

  * Training using vgg16:
  
    * Epochs: 5
    * LearningRate: 0.003
    * Hidden-units: 1024
    * Dropout: 0.2
    
    * Validation accuracy result: 83.8%
    * Validation loss result: 0.574
    * Trainning loss result: 1.157
    
    * Testing accuracy result: 84.1%
    * Testing loss result: 0.586
    
  * Training using alexnet:
  
    * Epochs: 5
    * LearningRate: 0.003
    * Hidden-units: 1024
    * Dropout: 0.2
    
    * Validation accuracy result: 83.8%
    * Validation loss result: 0.574
    * Trainning loss result: 1.157
    
    * Testing accuracy result: 84.1%
    * Testing loss result: 0.586

# Architectures Modifications and Criterion Explanation:

* Architecture modification to the downloaded model (resnet50, alexnet, vgg16):

![alt text](https://github.com/JuanDavidPiscoJaimes/DeepLearningClassifier/blob/master/nnsquentialexplained.png?raw=true)

* Criterion usage:
For this model I selected the `nn.NLLLoss()` (Negative Log Likelihood Loss) criterion due to the fact that it is one of the best functions
to calculate the error of a model with a lot of classes to classify. This criterion is the reason why the last layer of the network
is a LogSoftmax() function.
