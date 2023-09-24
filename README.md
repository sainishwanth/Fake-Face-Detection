# Fake Face Detection

Basic Discriminator Built using CNN in order to detect face faces. It can be coupled with a Generator to create a Generative Adversial Network

## Dataset

Fake and Real Image dataset retreived from kaggle

![Link](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## Requirements

- Python3.9+
- tensorflow
- numpy
- pandas
- opencv
- seaborn
- tkinter
- scikit-learn

## Steps to run

#### Installing Dependencies

```
pip install -r requirements.txt
```

### Training

To train the model from the beginning
```
python3 Train.py
```

### Testing

To Tests the models accuracy
```
python3 Test.py
```

### Running

To run the model on sample data
```
python3 main.py
```
Load Image -> Load Model -> Run

## Results

### Working Example
![Working](images/readme/testing.png)

### Confusion Matrix
![Confusion Matrix](images/readme/confusion.png)

### Classiciation Report
![Classification Report](images/readme/classification.png)