# ML Classification Task

Image classification using machine learning

## Requirements

```bash
python 3.6
tensorflow==2.4.1
numpy==1.19.5
```

## Usage

```bash
python ml_classification_task.py --directory data\test_directory
```

## Tests

```bash
python tests.py
```

## Description

Classification model was created using MobileNetV2 as the base model. 
Subsequent neural network layers were added on top of it.
We used a transfer learning technique as described in: https://keras.io/guides/transfer_learning/
15000 images were used for training and 3000 for validation.
Labels: 0 - cat, 1 - dog, 2 - unknown_class.
Final accuracy on validation dataset was 98%.
Detailed description about the code functionality is provided in the comments.

