# Emotion Detection using CNN

## Project Overview

This project demonstrates a **simple image-based emotion detection system** using **Deep Learning (Convolutional Neural Networks)**.

The model is trained on facial emotion images and predicts the emotion from a new image.

The system can classify emotions such as:

* Happy
* Sad
* Angry

This project is designed to be **simple for educational purposes**, making it suitable for beginners learning **Machine Learning and Computer Vision**.

---

# Technologies Used

| Technology         | Purpose                             |
| ------------------ | ----------------------------------- |
| Python             | Programming language                |
| TensorFlow / Keras | Building and training the CNN model |
| OpenCV             | Image loading and preprocessing     |
| NumPy              | Numerical operations                |
| Matplotlib         | Displaying images                   |

---

# Dataset

Dataset used: **FER2013 Facial Emotion Dataset**

Download link:
https://www.kaggle.com/datasets/msambare/fer2013

Dataset structure:

```
dataset/
   train/
       happy/
       sad/
       angry/

   test/
       happy/
       sad/
       angry/
```

Each folder contains images representing that emotion.

---

# Code Explanation

## 1. Import Required Libraries

```python
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```

### Explanation

* **TensorFlow / Keras** → Used to build and train the CNN model
* **NumPy** → Used for numerical operations
* **OpenCV** → Used for reading and preprocessing images
* **Matplotlib** → Used to display results

---

## 2. Define Dataset Paths

```python
train_path = "dataset/train"
test_path = "dataset/test"
```

### Explanation

This defines the dataset directory paths.

* `train_path` → training images
* `test_path` → validation/testing images

---

## 3. Image Preprocessing

```python
datagen = ImageDataGenerator(rescale=1./255)
```

### Explanation

Images normally have pixel values between **0–255**.

Rescaling converts them to **0–1**, which helps the neural network train more efficiently.

---

## 4. Load Training Dataset

```python
train_data = datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)
```

### Explanation

This loads images from the training directory.

Parameters:

* `target_size` → resize images to **128×128**
* `batch_size` → process **32 images at once**
* `class_mode` → multi-class classification

Labels are automatically assigned based on folder names.

---

## 5. Load Testing Dataset

```python
test_data = datagen.flow_from_directory(
    test_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)
```

### Explanation

This loads testing images used for **model validation**.

The model does not train on these images.

---

## 6. Create the CNN Model

```python
model = Sequential()
```

### Explanation

Initializes a **Sequential neural network**, where layers are stacked one after another.

---

## 7. First Convolution Layer

```python
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
```

### Explanation

* **Conv2D layer** extracts image features like edges and shapes.
* **MaxPooling** reduces image dimensions and computation.

---

## 8. Second Convolution Layer

```python
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
```

### Explanation

This layer extracts **more complex features**, such as facial structures and expressions.

Increasing filters from **32 → 64** improves feature detection.

---

## 9. Flatten Layer

```python
model.add(Flatten())
```

### Explanation

Converts the 2D feature maps into a **1D vector** so it can be used by dense layers.

---

## 10. Dense Layers

```python
model.add(Dense(128,activation='relu'))
model.add(Dense(train_data.num_classes,activation='softmax'))
```

### Explanation

* First dense layer learns complex patterns.
* Output layer predicts emotion probabilities using **Softmax**.

Example output:

```
Happy → 0.80
Sad → 0.10
Angry → 0.10
```

---

## 11. Compile the Model

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Explanation

* **Optimizer (Adam)** adjusts model weights during training
* **Loss Function** measures prediction error
* **Accuracy** tracks performance

---

## 12. Train the Model

```python
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)
```

### Explanation

This trains the CNN model.

* `train_data` → used for learning
* `validation_data` → used for performance evaluation
* `epochs` → number of training iterations

---

## 13. Save the Model

```python
model.save("emotion_model.h5")
```

### Explanation

The trained model is saved so it can be reused without retraining.

---

## 14. Load and Preprocess Test Image

```python
img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img / 255.0
img = np.reshape(img,(1,128,128,3))
```

### Explanation

Steps performed:

1. Load image
2. Resize to model input size
3. Normalize pixel values
4. Convert to model input format

---

## 15. Predict Emotion

```python
prediction = model.predict(img)
```

### Explanation

The model predicts probabilities for each emotion class.

---

## 16. Convert Prediction to Label

```python
labels = list(train_data.class_indices.keys())
predicted_class = labels[np.argmax(prediction)]
```

### Explanation

`np.argmax()` returns the index with the highest probability.

Example:

```
[0.80, 0.10, 0.10]
```

Prediction = **Happy**

---

## 17. Display Prediction

```python
plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.title("Prediction: " + predicted_class)
plt.axis("off")
plt.show()
```

### Explanation

Displays the image with the predicted emotion label.

---

# Example Output

```
Predicted Emotion: Happy
```

The model successfully detects emotion from the image.

---

# Future Improvements

* Increase dataset size
* Train for more epochs
* Add more emotion classes
* Implement **real-time webcam emotion detection**
