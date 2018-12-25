# from sklearn.metrics import classification_report
import os
import plaidml
from plaidml import keras
plaidml.keras.install_backend()

# import keras
import keras
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import Flatten, Dropout, Dense, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator # when using notebook

import matplotlib
matplotlib.use("Agg")
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import matplotlib.pyplot as plt
from  scipy import ndimage
from imutils import paths
import config
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

NUM_EPOCHS = 80
INIT_LR = 1e-1
BS = 64

# def poly_decay(epoch):
    
#     maxEpochs = NUM_EPOCHS
#     baseLR = INIT_LR
#     power = 1.0

#     alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

#     return alpha

totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
print("[INFO] total training images {}".format(totalTrain))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
print("[INFO] total validation images {}".format(totalVal))
totalTest = len(list(paths.list_images(config.TEST_PATH)))
print("[INFO] total testing images {}".format(totalTest))

trainAug = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=30,
    shear_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

valAug = ImageDataGenerator(rescale=1/255.0)

trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS
)

valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS
)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS
)
# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('model loaded!')
# else:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy']
)

print(model.summary())
# callbacks = [LearningRateScheduler(poly_decay)]
history = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps = totalVal // BS,
    epochs=NUM_EPOCHS,
    verbose=1
    # callbacks=callbacks
)

print("[INFO] evaluating model...")
testGen.reset()
# predIdxs = model.predict_generator(testGen,
#     steps=(totalTest//BS) + 1)

# predIdxs = np.argmax(predIdxs, axis=1)

score = model.evaluate_generator(testGen, verbose=1)

model.save("saved_model.model")

# Loss Curves
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Curves_Plot.png")
plt.show()

print("Test loss: {}".format(score[0]))
print("Test accuracy: {}".format(score[1]))