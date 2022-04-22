import argparse
import sys
import os
from pathlib import Path

# resolve the package import mess
current_path = os.getcwd()
module_path = Path(__file__).parent.parent
sys.path.append(str(module_path.resolve()))
os.chdir(module_path)

from model_wrapper import PersonalityTransfer

os.chdir(current_path)


def eval_dataset(classifier, texts, labels):
    correct_count = 0
    for text, label in zip(texts, labels):
        infered = classifier.classify(text.strip())
        if infered == int(label):
            correct_count += 1
    return correct_count / len(texts)


def main():
    parser = argparse.ArgumentParser(description='Model for text style transfer')
    parser.add_argument('--base-path',
                        default='style_transfer_paraphrase/utils/data',
                        help='The base path where to find the datasets.')
    parser.add_argument('--checkpoint',
                        default='final_model.pth',
                        help='The name of the checkpoint to use')
    args = parser.parse_args()
    personality_classifier = PersonalityTransfer('utils.config',
                                                 datasets_default_path=args.base_path,
                                                 checkpoint_name=args.checkpoint)
    datasets = ['personage-data', 'personality-detection', 'essays_test', 'essays', 'essays_unexpanded', 'mbti', 'mbti_unexpanded']
    print(f'Evaluating the model: {args.checkpoint}')
    for dataset in datasets:
        with open(os.path.join(args.base_path, dataset, 'text'), 'r') as texts,\
             open(os.path.join(args.base_path, dataset, 'labels_EXT'), 'r') as labels:
            labels = labels.readlines()
            texts = texts.readlines()
        accuracy = eval_dataset(personality_classifier, texts, labels)
        print(f'Accuracy for dataset {dataset} is {accuracy*100}%')


if __name__ == "__main__":
    main()
