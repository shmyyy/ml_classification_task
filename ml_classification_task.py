import argparse
from mobilenet_classifier import MobileNetClassifier

parser = argparse.ArgumentParser(description='A program for cat and dog images classification')
parser.add_argument('--directory', help='Path to a directory with cat and dog images')
args = parser.parse_args()

classifier = MobileNetClassifier(args.directory)
classifier.Run()
