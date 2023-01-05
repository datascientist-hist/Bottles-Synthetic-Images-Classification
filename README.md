# Bottles Synthetic Images Classification
![](/image/dataset-cover.jpg)

# Data Exploration
 [**Link to the Dataset**](https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset)
 
 [**Link to complete report**](/pdf/Report_Giuseppe_Pulino.pdf)
 
The dataset contains synthetically generated images of bottles scattered around random backgrounds.
The main folder  contains 25000 Images divided in 5 categories containing each one 5000 images with a resolution of 512 X 512 RGB and JPG as file extension.
The categories are the following:
- Soda Bottles
- Wine Bottles
- Water Bottles
- Plastic Bottles
- Beer Bottles

<p align="center">
  <img width="600"src="/image/collage.png">
</p>


As we can see in the example, images can contain more than one  bottles but always of the same type as the creator of the dataset reported.

# Problem Designing

The problem that we are designing is a **classification** problem, the environment that i will use is  MatLab.The  goal is to classify an image containing a bottle  among the following classes of bottles:

- Soda 
- Beer 
- Wine 
- Plastic
- Water

To perform this task I could  train one of the most famous nets from scratch but this would require too much time especially with a huge network  and a weak computational power like in my case, furthermore a huge network with million parameters requires millions of images to avoid overfitting, so even if that method remains the best because training from scratch a net allow us to extract  better features I can't use that method for the following reasons:

- weak computational power
- a few number of images compared with size of the net

Hence in this case, i will use transfer learning technique starting from a  pre-trained-model among those are available in MatLab and retraining the net  with new output starting from the weights of the pre-trained model.

<p align="center">
  <img width="800"src="/image/pretrained_image.png">
</p>

# Choosing the Neural Network
**From the Figure above we can see that we have a lot of available nets, so which is the best net for our task?**

Since the task is not so hard, different nets can be good to solve this task, so I decided to have a look to the winning nets of ImageNet Large Scale Visual Recognition Challenge (ILSVRC)(Figure 3).ILSVRC evaluates algorithms for object detection and image classification at large scale,it has been held from 2010 to 2017 and then has been stopped since it has been reached an high level of accuracy very close to 98-99%.Hence, looking at the performance of the nets in the challenge I decided to choose **ResNet** architecture but since the net has  152 layers with more then 50 millions parameters i chose the lighter version called ResNet-18 with only 18 layers and 11.7 millions parameters

<p align="center">
  <img width="800"src="/image/Original-ResNet-18-Architecture.png">
  <img width="800"src="/image/resnet.png">
</p>


# Data Processing

Since I had a problem with my computational power during the training of the net, I decided to reduce the number of images for each categories during the train phase using the following partition:

- Train with respectively 500 images for categories 
- Validation  with respectively 200 images for categories 
- Test with respectively 4300 images for categories 

```
[train,validation,test] = splitEachLabel(image_datastore,0.1, 0.04, 0.86, 'randomized');
```

Then i proceeded resizing the image to give in input to net from 512-by-512 X 3 to 224-by-224 X 3

```
resized_images_train=augmentedImageDatastore([224 224 3],train);
resized_images_validation = augmentedImageDatastore([224 224 3],validation);
resized_images_test = augmentedImageDatastore([224 224 3],test);
```

I could have used data augmentation procedure to allow to the model generalize more, but since there is already randomness in the images for example bottles rotated,bottles shifted, I preferred don't use that procedure due to my limited computational power.

# Training ResNet-18

After data preprocessing step i setted the hyperparameter

```
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",10,...
    "MiniBatchSize",64,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",70,...
    "Plots","training-progress",...
    "ValidationData",resized_images_validation,...
    "Momentum",0.9);
```
 
 and I trained the net
 
```
[net, traininfo] = trainNetwork(resized_images_train,resnet_18,opts);
```
As we can see from Figure below the net has been trained for more than 2 hours,it has been able to reach a good accuracy after the second epoch, furthermore from the training chart it doesn't seem to be overfitting since that train accuracy and validation accuracy have almost the same value.Since I left 4300 images out for each categories from training phase, I will use them to test the real performance of the model.

<p align="center">
    <img width="800"src="/image/training chart.JPG">
</p>

# Model performance evaluation on Test set

Looking at the confusion matrix computed on test set we can note that only 101 images has been misclassified leading to an accuracy of 99.53%, so I  can consider that the  model is a very good model even without tuning the hyperparameters
Below I report the association between name class and label assigned to the class:

- 1 Beer
- 2 Plastic
- 3 Soda
- 4 Water
- 5 Wine

```
true_test_labels = test.Labels;
pred_test_labels = classify(net, rsz_test);
accuracy_test = mean(true_test_labels == pred_test_labels);

C = confusionmat(true_test_labels, pred_test_labels);
confusionchart(C)
```
<p align="center">
    <img width="800"src="/image/conftest.jpg">
</p>

As we can from the confusion matrix most of errors occured in the beer predicted class, instead the predicted class with less errors has been plastic class with only 1 misclassification error.

Maybe training the model with all the images that are available and fine-tuning the hyperparameters I could obtain better result even if I think that do better than this becomes very hard indeed we could obtain the opposite effect making the model worse
