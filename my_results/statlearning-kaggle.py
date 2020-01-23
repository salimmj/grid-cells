# -*- coding: utf-8 -*-
"""statlearning-kaggle.py"""

!unzip ./statlearning-sjtu-2019.zip
!ls ./quickdraw-data/train

"""### Import Libraries

For this project, I use multiple Python libraries such PIL for image processing, matplotlib for plotting the images, numpy for numerical operations, and scikit-learn for Machine Learning models.
"""


from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import skimage
from skimage.io import imread
from skimage.transform import resize
from skimage import color
import shutil
import csv

"""### Process Data
We inspect the shape of the data
"""

im = Image.open("quickdraw-data/train/airplane/"+os.listdir("quickdraw-data/train/airplane")[0])
print('the image size:',im.size)

categories = set(os.listdir("extra_training_data/data"))
assert categories == set(os.listdir("quickdraw-data/train"))
print('the image categories are:', categories)

"""#### Converting images in extra_training_data from JPG to PNG and from RGB to Grayscale

In order to process the data, we do the following. For each category, we open each image in the appropriate folder, make sure it is converted to grayscale, convert it into the PNG format, then save it back. We then delete all the JPG files left.
"""

for cat in categories:
  for file in os.listdir("extra_training_data/data/" + cat): 
    if file.endswith(".jpg"):
      im1 = Image.open("extra_training_data/data/" + cat + '/' + file)
      im1.convert('L')
      pngfile = file.split('.')
      pngfile[-1] = 'png'
      pngfile = '.'.join(pngfile)
      im1.save("extra_training_data/data/" + cat + '/' + pngfile)
      os.remove("extra_training_data/data/" + cat + '/' + file)
    elif not file.endswith(".png"):
      os.remove("extra_training_data/data/" + cat + '/' + file)

# cleaning out png files
for cat in categories:
  for file in os.listdir("quickdraw-data/train/" + cat): 
    if not file.endswith(".png"):
      os.remove("quickdraw-data/train/" + cat + '/' + file)

"""#### Merging data from the extra_training_data folder into quickdraw-data

After all of this is done, we copy the extra_training data over to the primary training data folder quickdraw-data.
"""

for cat in categories:
  source = os.listdir("extra_training_data/data/" + cat)
  destination = "quickdraw-data/train/" + cat
  for files in source:
      shutil.copy("extra_training_data/data/" + cat +"/"+files,destination)

"""#### Processing data into a Bunch() class (flat, labeled, and clean)

We are given a mapping from text labels to numerical representation which we initialize below. 

Now, we load the data from image files into a python object containing the matric representations of those images. We use the sklearn class Bunch().

We then split the data into training (two thirds) and testing (one third).
"""

# map text labels to their numerical representation
target_dict = {'airplane': 0, 'ant': 1, 'bear': 2, 'bird': 3, 'bridge': 4,
     'bus'     : 5, 'calendar': 6, 'car': 7, 'chair': 8, 'dog': 9,
     'dolphin' : 10, 'door': 11, 'flower': 12, 'fork': 13, 'truck': 14}


def load_image_files(container_path, dimension=(28, 28)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "An image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            # taking care of irregularities
            if len(img.shape) != 2:
              if len(img.shape) == 3:
                img = color.rgb2gray(img)
              else:
                continue
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(target_dict[direc.name])
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
    
image_dataset = load_image_files("quickdraw-data/train")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

print('The training data matrix contains', X_train.shape[0], 'image vectors (rows) of size', X_train.shape[1])
print('The test data matrix contains', X_test.shape[0], 'image vectors (rows) of size', X_test.shape[1])

"""### SVM Model

The first model we try and the one which seems to perform the best out of all was a Support Vector Machine model. The first times we ran this model we actually got a locally-tested accuracy of 77%. We were not able to reproduce these results unfortunately.

One thing to note, however, is that we used to run a long stretching 15-hour Grid Search procedure to do cross validation and fine tuning of the models hyperparameter which might have contributed to the greater score.

#### Fitting

We start by initializing the model without any fancy hyperparameter settings. We then call the fit() function which trains the model over X_train with labels y_train.
"""

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

"""#### Predicting on test and printing out the accuracy report

We test the model's accuracy on X_test and get a score of around 60%.
"""

score = svm_model.score(X_test, y_test)
print(score)

"""### Logistic Regression

The second model we try out is a Logistic Regression model which is popularly for classification tasks.

#### Fitting

Similarly, we initialize then train the model.
"""

logr_model = LogisticRegression(
    C=50. / X_train.shape[0], penalty='l2', solver='liblinear', tol=0.1
)
logr_model.fit(X_train, y_train)

"""#### Predicting on test and printing out the accuracy report

We get an accuracy of around 47%.
"""

score = logr_model.score(X_test, y_test)
print(score)

"""### XGBOOST

XGBoost is one the most popular models for both regression and classification. It is a clever implementation of boosted tree models and is widely used to win Datathons.

#### Downloading XGB

I had to reinstall XGB as it is not available on Colab.
"""

!wget https://s3-us-west-2.amazonaws.com/xgboost-wheels/xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl
!pip uninstall xgboost
!pip install xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl

"""#### Fitting

We fit the model in a similar fashion.
"""

import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective = "multi:softmax")
xgb_model.fit(X_train, y_train)

"""#### Predicting on test and printing out the accuracy report"""

score = xgb_model.score(X_test, y_test)
print(score)

"""### Predicting on the released_test data

We use the trained models to predict over the released testing dataset.

#### Preparing the released_test data
"""

image_dir = Path("quickdraw-data/released_test")

descr = "An image classification dataset"
images = []
flat_data = []
for file in image_dir.iterdir():
    img = skimage.io.imread(file)
    if len(img.shape) != 2:
      if len(img.shape) == 3:
        img = color.rgb2gray(img_resized)
      else:
        continue
    img_resized = resize(img, (28,28), anti_aliasing=True, mode='reflect')
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
X_released_test = np.array(flat_data)
print('Released test is of shape', X_released_test.shape)

"""#### Generating prediction for each model and generating a submission file (csv)"""

released_test_svm_preds = svm_model.predict(X_released_test)
released_test_logr_preds = logr_model.predict(X_released_test)
released_test_xgb_preds = xgb_model.predict(X_released_test)

image_dir = Path("quickdraw-data/released_test")
file_names = []
for file in image_dir.iterdir():
  file_names.append(file.name.split('.')[0])

def write_submission_file(name, preds, file_names):
  with open('submission_'+name+'.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["id", "categories"])
      print('writing submission_'+name+'.csv ...')
      for i in range(45000):
        writer.writerow([int(file_names[i]), preds[i]])
      print('done!')

write_submission_file('svm', released_test_svm_preds, file_names)
write_submission_file('logr', released_test_logr_preds, file_names)
write_submission_file('xgb', released_test_xgb_preds, file_names)