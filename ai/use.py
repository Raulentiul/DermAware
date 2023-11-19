import tensorflow as tf
from keras.models import load_model
from keras.layers import Layer
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sys

# Define a dummy custom layer with the same name
class DummyCastToFloat32Layer(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Load the model with the custom_objects parameter
loaded_model = load_model('cifar_model.h5', custom_objects={'Custom>CastToFloat32': DummyCastToFloat32Layer})

def use_AI(path):
    SIZE = 32
    # Load and preprocess the image
    #image_path = "C:\\Users\\maros\\Desktop\\Unihack\\data\\HAM10000_images_part_1\\ISIC_0025963.jpg"  # Replace with the path to your image
    image_path = path  # Replace with the path to your image
    image = Image.open(image_path).resize((SIZE, SIZE))
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = loaded_model.predict(image)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Print the result
    skin_df = pd.read_csv('data/HAM10000_metadata.csv')
    le = LabelEncoder()
    le.fit(skin_df['dx'])
    LabelEncoder()

    print("Classes:\n[1] Healthy\n[2] Actinic Keratoses\n[3] Basal Cell Carcinoma\n[4] Dermatofibroma"
          "\n[5] Melanoma\n[6] Melanocytic Nevi\n[7] Vascular Lesions")

    if("0" in str(predicted_class)):
        print(f'There is a chance of you to have Actinic Keratoses.')
        #The model predicts class Actinic keratoses (akiec) for the given image.
    elif("1" in str(predicted_class)):
        print(f'There is a chance of you to have Basal Cell Carcinoma.')
        #The model predicts class Basal cell carcinoma (bcc)  for the given image.
    elif ("2" in str(predicted_class)):
        print(f'Your skin looks healthy!')
    elif ("3" in str(predicted_class)):
        print(f'There is a chance of you to have Dermatofibroma.')
        #The model predicts class Dermatofibroma (df) for the given image.
    elif ("4" in str(predicted_class)):
        print(f'There is a chance of you to have Melanoma.')
        #The model predicts class Melanoma (mel) for the given image.
    elif ("5" in str(predicted_class)):
        print(f'There is a chance of you to have Melanocytic Nevi.')
        #The model predicts class Melanocytic nevi (nv) for the given image.
    elif ("6" in str(predicted_class)):
        print(f'There is a chance of you to have Vascular Lesions.')
        #The model predicts class Vascular Lesions (vasc) for the given image.


if __name__ == "__main__":
    script_name = sys.argv[0]
    use_AI(sys.argv[1])
