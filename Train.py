import os
import glob
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


path  = 'real-vs-fake/'

dataset = {"image_path":[],"img_status":[],"where":[]}
for where in os.listdir(path):
    for status in os.listdir(path+"/"+where):
        for image in glob.glob(path+where+"/"+status+"/"+"*.jpg"):
            dataset["image_path"].append(image)
            dataset["img_status"].append(status)
            dataset["where"].append(where)
dataset = pd.DataFrame(dataset)
real = dataset.value_counts("img_status")[1]
fake = dataset.value_counts("img_status")[0]

print(f"Real: {real},\nFake: {fake}\n")

image_gen = ImageDataGenerator(rescale=1./255.)

train_generator = image_gen.flow_from_directory(
    path + 'train/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    classes=['fake','real']
)

valid_generator = image_gen.flow_from_directory(
    path + 'valid/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    classes=['fake','real']
)

test_generator = image_gen.flow_from_directory(
    path + 'test/',
    target_size=(224, 224),
    batch_size=64,
    shuffle = False,
    class_mode='binary',
    classes=['fake','real']

)

densenet = DenseNet121( weights='imagenet', include_top=False, input_shape=(224,224,3) )

model = Sequential([ 
        densenet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())
history = model.fit(
    train_generator,
    steps_per_epoch = (100000//1000),
    validation_data = valid_generator,
    validation_steps = (10000//100),
    epochs = 3
)

model.save('image_model.h5')
print("Model Saved")