# Huge Bottles Synthetic Images Classification
![](/image/dataset-cover.jpg)

# Data Exploration
 [**Link to the Dataset**](https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset)

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




