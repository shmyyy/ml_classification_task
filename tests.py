import os
import io
import unittest
import numpy as np
from contextlib import redirect_stdout
from tensorflow.python.keras.engine.functional import Functional
from mobilenet_classifier import MobileNetClassifier

unittest.TestLoader.testMethodPrefix = 'Test'

class TestMobileNetClassifier(unittest.TestCase):

    def setUp(self):

        self.cwd = os.getcwd()
        test_directory = os.path.join(self.cwd, 'data\\test_directory')
        self.classifier = MobileNetClassifier(test_directory)
        np.random.seed()

    def TestLoadModel(self):

        '''
        test if loaded model has correct type
        '''
             
        model = self.classifier.LoadModel()
        self.assertIsInstance(model, Functional)

    def TestLoadImage(self):

        '''
        test if loaded image has correct type, shape, and is normalized
        test "rotation" input argument: invalid angles should raise ValueError
        '''

        images_directory = os.path.join(self.cwd, 'data\\samples\\cat')
        image_path = os.path.join(images_directory, os.listdir(images_directory)[0])
        image = self.classifier.LoadImage(image_path)

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (1, 224, 224, 3))
        
        image = image.ravel()
        is_normalized = ((image >= -1) & (image <= 1)).all()
        self.assertTrue(is_normalized)

        valid_angles = [0, 90, 180, 270]
        for angle in valid_angles:
            try:
                image = self.classifier.LoadImage(image_path, rotation=angle)
            except ValueError as e:
                self.fail(e)

        invalid_angles = [1, 45, 120, 245]
        for angle in invalid_angles:
            with self.assertRaises(ValueError):
                image = self.classifier.LoadImage(image_path, rotation=angle)

    def TestClassify(self):

        '''
        tests that for each class most of the predicted labels are correct
        at the same time we test that predicted are labels are in [0, 1, 2]
        '''

        class_names = ['cat', 'dog', 'unknown_class']

        for label, class_name in enumerate(class_names):
            predicted_labels = [] 
            directory = os.path.join(self.cwd, 'data\\samples\\%s' % class_name)

            for file in os.listdir(directory)[:10]:
                image_path = os.path.join(directory, file)
                image = self.classifier.LoadImage(image_path)
                predicted_label = self.classifier.Classify(image)
                predicted_labels.append(predicted_label)

            most_common_label = max(predicted_labels, key=predicted_labels.count)
            self.assertEqual(most_common_label, label)  
            
    def TestGetClassName(self):

        '''
        test that all returned class names are correct for "unsupported_file"
        test that most of the returned class names are correct for others  
        '''

        class_names = ['unsupported_file', 'cat', 'dog', 'unknown_class']
        
        for class_name in class_names:
            ret_class_names = []
            directory = os.path.join(self.cwd, 'data\\samples\\%s' % class_name)

            for file in os.listdir(directory)[:10]:
                file_path = os.path.join(directory, file)
                ret_class_name = self.classifier.GetClassName(file_path)
                ret_class_names.append(ret_class_name)

            if class_name == 'unsupported_file':
                all_correct = all(ret_class_name == class_name for ret_class_name in ret_class_names)
                self.assertTrue(all_correct)
            else:
                most_common_class_name = max(ret_class_names, key=ret_class_names.count)
                self.assertEqual(most_common_class_name, class_name) 

    def TestRun(self):

        '''
        test that are exactly the same count of lines printed to stdout as there number of files in test directory
        test that all file names from directory are output to stdout
        test that there are at least one line with the name of each class
        '''

        class_names = ['unsupported_file', 'cat', 'dog', 'unknown_class']
        file_names = os.listdir(self.classifier.directory)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.classifier.Run()

        stdout = buffer.getvalue()
        lines = stdout.strip().split('\n')
        self.assertEqual(len(lines), len(file_names))

        output_file_names, output_class_names = set(), set()
        for line in lines:
            file_name, class_name = line.split('|')
            file_name, class_name = file_name.strip(), class_name.strip()
            output_file_names.add(file_name)
            output_class_names.add(class_name)

        for file_name in file_names:
            self.assertIn(file_name, output_file_names)

        for class_name in class_names:
            self.assertIn(class_name, output_class_names)

    def TestModelAccuracy(self):

        '''
        we trained our model using "tansfer_learning.ipynb", final validation accuracy of our model was 98%
        lets take a more conservative threshold of 95% and test if our model accuracy exceeds this threshold
        '''

        def Samples():

            class_names = ['cat', 'dog']
            angles = [0, 90, 180, 270]

            for label, class_name in enumerate(class_names):
                directory = os.path.join(self.cwd, 'data\\samples\\%s' % class_name)
                label = np.array([label], dtype=np.int32)

                for file in os.listdir(directory):
                    image_path = os.path.join(directory, file)
                    angle = np.random.choice(angles)
                    image = self.classifier.LoadImage(image_path, rotation=angle)
                    yield image, label

        metrics = self.classifier.model.evaluate(Samples())
        self.assertGreater(metrics[1], 0.95)
        

if __name__ == '__main__':
    unittest.main()
