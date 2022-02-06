import random
import os
import argparse
from cv2 import cv2
from model import Classifier
from matplotlib import pyplot as plt


def parse_arguments():
    """
    Object for parsing command line strings into Python objects.
    """
    arg = argparse.ArgumentParser()
    arg.add_argument('--source', '-s', type=str, default='data/EuroSAT/2750',
                     help="give main source directory")
    arg.add_argument('--device', '-d', default='cuda',
                     type=str, choices=['cuda', 'cpu'])
    arg.add_argument('--model_path', '-m', type=str, default='saved_models/model_best.pth',
                     help="give saved model path")
    arg.add_argument('--display', action='store_true')
    arg.add_argument('--colab', action='store_true')

    return vars(arg.parse_args())


def display(img, gt, pred, is_colab):
    """
    Display the image and the prediction
    """
    if gt == pred:
        text = f"Correct. Pred: {pred}"

    else:
        text = f"Incorrect. GT: {gt}, Pred: {pred}"

    if is_colab:
        plt.imshow(img)
        plt.title(text)
        plt.savefig(f'{gt}.png')
    else:
        cv2.imshow(f'predict/{text}', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    kwargs = parse_arguments()
    device = kwargs.pop('device')
    source = kwargs.pop('source')
    model_path = kwargs.pop('model_path')
    is_display = kwargs.pop('display')
    is_colab = kwargs.pop('colab')
    random.seed(42)

    model = Classifier()
    model = model.from_pretrained(model_path).to(device)

    category_list = os.listdir(source)
    for category in category_list:
        category_path = os.path.join(source, category)
        category_img_list = os.listdir(category_path)
        random_selected = random.choice(category_img_list)
        print(random_selected)
        img = cv2.imread(os.path.join(
            category_path, random_selected))

        result = model.predict(img)
        max_proba_result = max(result[0], key=result[0].get)

        print("--"*20)
        print(f"Ground truth: {category}")
        print(f"Predicted: {max_proba_result}")
        print(
            f"Result: {'Correct' if category == max_proba_result else 'Incorrect'}")

        if is_display:
            display(img, category, max_proba_result, is_colab)
