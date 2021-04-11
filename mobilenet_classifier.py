import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class MobileNetClassifier:

    def __init__(self, directory):

        self.directory = directory
        self.supported_formats = set(['.jpg', '.png'])
        self.angles = set([0, 90, 180, 270])
        self.model = self.LoadModel()

    def LoadModel(self):

        '''
        load the model which was trained using "transfer_learning.ipynb"
        '''

        model_path = os.path.join(os.getcwd(), 'data\\cats_vs_dogs.h5')
        model = load_model(model_path)

        return model

    def LoadImage(self, path, rotation=0):

        '''
        resize the image to size required by the model - 224x224 pixels
        optionally image can be rotated either 90, 180 or 270 degrees
        use "preprocess_input" for normalizing image pixel values to range [-1, 1]
        also we need to expand the array dimensions to be (1, 224, 224, 3)
        https://keras.io/api/applications/#usage-examples-for-image-classification-models
        '''

        if rotation not in self.angles:
            raise ValueError('Invalid rotation value, must be one of 0, 90, 180, 270')

        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        if rotation > 0:
            image = np.rot90(image, rotation / 90)

        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        return image

    def Classify(self, image):

        '''
        the activation function of last layer was softmax, predictions[0] will be array of size 3 (labels: 0, 1, 2)
        the predicted label will be array index where probability value is the highest
        '''

        predictions = self.model.predict(image)
        label = predictions[0].argmax()
        return label

    def GetClassName(self, image_path):

        '''
        load and preprocess image
        predict label
        labels: 0 - cat, 1 - dog, 2 - unknown_class
        '''

        image_format = os.path.splitext(image_path)[1]
        if image_format not in self.supported_formats:
            return 'unsupported_file'

        class_names = ['cat', 'dog', 'unknown_class']
        image = self.LoadImage(image_path)
        label = self.Classify(image)

        return class_names[label]

    def Run(self):

        for file_name in os.listdir(self.directory):
            absolute_path = os.path.join(self.directory, file_name)
            if os.path.isfile(absolute_path):
                class_name = self.GetClassName(absolute_path)
                print('%s | %s' % (file_name, class_name))