from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

model = VGG16(weights='imagenet')
model.summary()

img = image.load_img('elephant.jpg', target_size=(224, 224))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict(preprocess_input(x))
results = decode_predictions(preds)[0]

for result in results:
    print(result)
