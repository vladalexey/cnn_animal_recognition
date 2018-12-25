from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

from imutils import paths, build_montages
import numpy as np 
import h5py
import random
import cv2

image_paths = list(paths.list_images("all/testing_old"))
random.shuffle(image_paths)
images = image_paths[:25]
results = []

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('saved_model-218-0.2131.hdf5')

for image in images:
    original = cv2.imread(image)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float') / 255.0

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    pred = model.predict(image)
    pred = pred.argmax(axis=1)[0]

    label = "Cat" if pred == 0 else "Dog"
    original = cv2.resize(original, (128, 128))
    cv2.putText(original, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    results.append(original)

montage = build_montages(results, (150, 150), (5, 5))[0]
cv2.imshow('Image: ',montage)
cv2.waitKey(0)
cv2.destroyAllWindows()