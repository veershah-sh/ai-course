import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ===============================
# DATASET PATH
# ===============================

train_path = "face_dataset/train"
test_path = "face_dataset/test"

# ===============================
# DATA PREPROCESSING
# ===============================

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    test_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# ===============================
# BUILD CNN MODEL
# ===============================

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(train_data.num_classes,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ===============================
# TRAIN MODEL
# ===============================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# ===============================
# SAVE MODEL
# ===============================

model.save("emotion_model.h5")

# ===============================
# TEST WITH NEW IMAGE
# ===============================

img_path = "test_image.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img / 255.0

img = np.reshape(img,(1,128,128,3))

prediction = model.predict(img)

labels = list(train_data.class_indices.keys())

predicted_class = labels[np.argmax(prediction)]

print("Predicted Emotion:", predicted_class)

# ===============================
# SHOW IMAGE
# ===============================

plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.title("Prediction: " + predicted_class)
plt.axis("off")
plt.show()