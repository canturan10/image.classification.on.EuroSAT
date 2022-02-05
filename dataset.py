import os

import imageio
import numpy as np
import torch
from skimage import img_as_float32, img_as_ubyte
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.datasets.utils import download_and_extract_archive
from colorama import Fore


class EurosatDataset(torch.utils.data.Dataset):
    """
    EuroSAT: Land Use and Land Cover Classification with Sentinel-2
    Eurosat is a dataset and deep learning benchmark for land use and land cover classification. The dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.
    """

    def __init__(self, is_train, root_dir="data/EuroSAT/2750/", transform=None, seed=42, download=False):
        """
        EurosatDataset

        Args:
            is_train (bool): If true returns training set, else test set.
            root_dir (str, optional): Root directory of dataset. Defaults to "data/EuroSAT/2750/".
            transform ([type], optional): Optional transform to be applied on a sample. Defaults to None.
            seed (int, optional): Seed used for train/test split. Defaults to 42.
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded it is not downloaded again. Defaults to False.
        """

        self.seed = seed
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.download = download

        self.size = [64, 64]
        self.num_channels = 3
        self.num_classes = 10
        self.test_ratio = 0.2
        self.N = 27000
        self._load_data()

    def _load_data(self):
        """
        Loads the data from the passed root directory. Splits in test/train based on seed.

        Raises:
            RuntimeError: It will raise when folder not exists.
        """

        images = np.zeros(
            [self.N, self.size[0], self.size[1], 3], dtype="uint8")
        labels = []
        filenames = []

        if self.download:
            self.download_dataset()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        i = 0

        with tqdm(os.listdir(self.root_dir), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)) as dir_bar:
            for item in dir_bar:
                f = os.path.join(self.root_dir, item)
                if os.path.isfile(f):
                    continue
                for subitem in os.listdir(f):
                    sub_f = os.path.join(f, subitem)
                    filenames.append(sub_f)

                    # a few images are a few pixels off, we will resize them
                    image = imageio.imread(sub_f)
                    if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
                        # print("Resizing image...")
                        image = img_as_ubyte(
                            resize(
                                image, (self.size[0], self.size[1]), anti_aliasing=True)
                        )
                    images[i] = img_as_ubyte(image)
                    i += 1
                    labels.append(item)

                dir_bar.set_description(
                    f"{'Train' if self.is_train else 'Test'} images are reading..")
                dir_bar.set_postfix(category=item)

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # sort by filenames
        images = images[filenames.argsort()]
        labels = labels[filenames.argsort()]

        # convert to integer labels
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(np.sort(np.unique(labels)))
        labels = label_encoder.transform(labels)
        labels = np.asarray(labels)
        # remember label encoding
        self.label_encoding = list(label_encoder.classes_)

        # split into a is_train and test set as provided data is not presplit
        x_train, x_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=labels,
        )

        if self.is_train:
            self.data = x_train
            self.targets = y_train
        else:
            self.data = x_test
            self.targets = y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        image = np.asarray(img / 255, dtype="float32")

        return image.transpose(2, 0, 1), self.targets[idx]

    def _check_exists(self) -> bool:
        """
        Check the Root directory is exists
        """
        return os.path.exists(self.root_dir)

    def download_dataset(self) -> None:
        """
        Download the dataset from the internet
        """

        if self._check_exists():
            return

        os.makedirs(self.root_dir, exist_ok=True)
        download_and_extract_archive(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
            download_root=self.root_dir,
            md5="c8fa014336c82ac7804f0398fcb19387",
        )


if __name__ == '__main__':
    dset = EurosatDataset(is_train=True, seed=42, download=True)
    print(len(dset))
    print(dset.label_encoding)
