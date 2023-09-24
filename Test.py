import os
import cv2
import numpy as np
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

path  = 'real-vs-fake/'

real_images_test =[r for r in os.listdir(f'{path}test/real/')]
fake_images_test =[r for r in os.listdir(f'{path}test/fake/')]

model = load_model("image_model.h5")
model.summary()

data_test_real = path + 'test/real/'
data_test_fake = path + 'test/fake/'

classes=[]
threshold = 0.5
for i in fake_images_test[0:60]:
    data = img_to_array(cv2.resize(cv2.imread(data_test_fake+i), (224, 224))).flatten() / 255.0
    data = data.reshape(-1, 224, 224, 3)
    predictions = model.predict(data)
    predicted_class = int(predictions >= threshold)
    classes.append(predicted_class)

image_gen = ImageDataGenerator(rescale=1./255.)
test_generator = image_gen.flow_from_directory(
    path + 'test/',
    target_size=(224, 224),
    batch_size=64,
    shuffle = False,
    class_mode='binary',
    classes=['fake','real']

)

y_pred = model.predict(test_generator)
y_test = test_generator.classes



plt.figure(figsize = (8,5))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred.round()), annot = True,fmt="d",cmap = "Blues")
plt.show()

print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred))
print("AP Score:", metrics.average_precision_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred > 0.5))