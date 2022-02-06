from typing import Dict, List
import numpy as np
import torch
from torch import nn
import torchsummary


class Classifier(nn.Module):
    """
    Image Classifier Pytorch Model
    """
    # pylint: disable=no-member
    # pylint: disable=not-callable

    __default_config__ = {
        "input_size": 64,
        "labels": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'],
        "loss_function": "crossentropy",
        'dropout': 0.2,

    }

    def __init__(self, config: Dict = None):
        """
        Image Classifier

        Args:
            config (Dict): Configurations that contains the following keys: input_size, labels
        """
        super(Classifier, self).__init__()

        self.config = self.__default_config__ if config is None else config

        self.labels = list(self.config["labels"])
        self.num_classes = len(self.labels)

        if self.config['loss_function'] == 'crossentropy':
            self.loss_fcn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown criterion')

        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(
                int(((self.config["input_size"] - 5 + 1) // 2)) ** 2 * 32, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes output Tensors from input Tensors.

        Args:
            inputs (torch.Tensor): Input Tensors.

        Returns:
            torch.Tensor: Output Tensors.
        """

        return self.backbone(inputs)

    def predict(self, inputs: np.ndarray) -> List[str]:
        """
        Converts logits to predictions.

        Args:
            inputs (np.ndarray): Input Data.

        Returns:
            List[str]: Returns a list of predicted labels.
        """
        data = self.preprocess_input(inputs)
        preds = torch.softmax(self.backbone(data), axis=-1)

        outputs = []
        for scores in preds.detach().cpu().numpy().tolist():
            output = {label: round(score, 3) for score,
                      label in zip(scores, self.labels)}
            outputs.append(output)
        return outputs

    @classmethod
    def from_pretrained(cls, model_path: str, *args, **kwargs) -> nn.Module:
        """
        Load model from a pre-trained model.

        Args:
            model_path (str): Pretrained model path.

        Returns:
            nn.Module: Pretrained model.
        """
        state_dict = torch.load(model_path)
        pretrained_model = cls(config=state_dict['config'], *args, **kwargs)
        pretrained_model.load_state_dict(state_dict['state_dict'])
        return pretrained_model

    @classmethod
    def summarize(cls, input_size=(3, 64, 64)):
        """
        Summarizes torch model by showing trainable parameters and weights.

        Args:
            input_size (tuple, optional): Input size of the model. Defaults to (3, 64, 64).

        Returns:
            Summary of the model.
        """

        return torchsummary.summary(cls().to(device), input_size=input_size)

    def to(self, device: str, *args, **kwargs):
        """
        Performs Tensor device conversion.

        Args:
            device (str): Device to convert to.

        """
        self.device = device
        return super().to(device, *args, **kwargs)

    def preprocess_input(self, input_array: np.ndarray) -> torch.Tensor:
        """
        Preprocesses input array to be compatible with the model.

        Args:
            input_array (np.ndarray): Input array.

        Raises:
            AssertionError: If input_array is not a shape of (h,w,c) or (bs,h,w,c).

        Returns:
            torch.Tensor: Converted input array as torch.Tensor.
        """
        data = input_array.copy()

        if len(data.shape) == 3:  # h,w,c
            new_inputs = torch.from_numpy(
                np.expand_dims(data, axis=0)).float().permute(0, 3, 1, 2).contiguous()
        elif len(data.shape) == 4:  # assumed bs,h,w,c
            new_inputs = torch.from_numpy(
                data).float().permute(0, 3, 1, 2).contiguous()
        else:
            raise AssertionError(
                f"input shape not supported: {data.shape}")
        return new_inputs.to(self.device)

    def loss(self, y_pred, y_true):
        """
        [summary]

        Args:
            y_pred ([type]): [description]
            y_true ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.loss_fcn(y_pred, y_true)

    def predict_proba(self, x):
        """
        [summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return nn.functional.softmax(self(x), dim=-1)

    def predict_classes(self, x):
        """
        [summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return torch.argmax(self(x), dim=-1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier().to(device)

    input_array = np.random.rand(64, 64, 3)
    result = model.predict(input_array)
    print(result)

    # input_tensor = torch.randn(1, 3, 64, 64).to(device)
    # forward = model(input_tensor)  # pylint: disable=not-callable
    # print(forward)

    # from cv2 import cv2
    # input_img = cv2.imread('data/EuroSAT/2750/AnnualCrop/AnnualCrop_12.jpg')
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # result = model.predict(input_img)
    # print(result)

    # print(model.summarize())
