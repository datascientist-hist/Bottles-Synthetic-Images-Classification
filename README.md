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

<img src="/image/pretrained_image.png" align="center" width="800" />


