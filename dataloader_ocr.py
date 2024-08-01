from matplotlib import pyplot as plt
import tensorflow as tf
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
import json


class DataLoader(tf.keras.utils.Sequence):

    def __init__(
        self,
        source_files,
        im_size=[200, 50, 3],
        batch_size=64,
        max_text_size=9,
        split=80,
        shuffle=True,
        augmentation=False,
        vocabulary=list(
            "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ:. _()л"
        ),
        work_mode="train",
    ):
        self.source_files = source_files
        self.im_size = im_size
        self.batch_size = batch_size
        self.max_text_size = max_text_size
        self.split = split
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.vocab = vocabulary
        self.work_mode = work_mode
        self.dataset = []
        self.load_datasets()
        self.init_dataset()

    def load_datasets(self):
        all_data = []
        for json_file in self.source_files:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.extend(data["ocr_files"])
        self.dataset = []
        for entry in all_data:
            img_path = entry["file"]
            label = entry["text"]
            if len(label) < self.max_text_size:
                label += "-" * (self.max_text_size - len(label))
            if self.work_mode == "train":
                label = [self.vocab.index(char) + 1 for char in label]
            self.dataset.append({"path_img": img_path, "label": label})

    def init_dataset(self):
        self.count_imgs = len(self.dataset)
        split_index = int(abs(self.split) * self.count_imgs / 100)
        if self.split > 0:
            self.dataset = self.dataset[:split_index]
        else:
            self.dataset = self.dataset[-split_index:]
        self.indexes = np.arange(len(self.dataset))
        self.batches = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size:
            self.batches += 1
        self.on_epoch_end()
        if self.augmentation:
            self.aug_gaussian_noise = self._init_augmentation()

    def _init_augmentation(self):
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=50.0, per_channel=True, p=0.8),
                        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0, 3), p=0.8),
                        A.ISONoise(
                            color_shift=(0.05, 0.5), intensity=(0.1, 0.5), p=0.8
                        ),
                    ]
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, 0.4),
                    contrast_limit=(-0.4, 0.4),
                    brightness_by_max=True,
                    always_apply=False,
                    p=0.6,
                ),
                A.Rotate(limit=(-5, 5)),
            ],
            p=1,
        )

    def pad_image(self, image, target_size=(200, 50), pad_color=(181, 181, 181)):
        h, w = image.shape[:2]

        # Проверка, если текущая высота меньше целевой высоты, добавляем паддинги сверху и снизу
        if h < target_size[1]:
            top = (target_size[1] - h) // 2
            bottom = target_size[1] - h - top
        else:
            top = 0
            bottom = 0

        # Проверка, если текущая ширина меньше целевой ширины, добавляем паддинг справа
        if w < target_size[0]:
            left = 0
            right = target_size[0] - w
        else:
            left = 0
            right = 0

        new_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )

        # Обрезка до целевого размера, если изображение превышает его
        new_image = new_image[: target_size[1], : target_size[0]]

        # plt.imsave("original_image.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.imsave("new_image.png", cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        return new_image

    def __getsample__(self, idx):
        example = self.dataset[idx]
        img = cv2.imread(str(example["path_img"]))
        img = self.pad_image(img, (self.im_size[0], self.im_size[1]))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (self.im_size[1], self.im_size[0]))

        # plt.imsave("train_image.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.augmentation:
            img = self.aug_gaussian_noise(image=img)["image"]
        return img, example["label"]

    def __getitem__(self, idx):
        start_ind = idx * self.batch_size
        end_ind = (idx + 1) * self.batch_size
        if end_ind >= len(self.indexes):
            indexes = self.indexes[start_ind:]
        else:
            indexes = self.indexes[start_ind:end_ind]

        imgs = np.zeros(
            (len(indexes), self.im_size[0], self.im_size[1], self.im_size[2]),
            dtype=np.uint8,
        )

        if self.work_mode == "train":
            labels = np.zeros((len(indexes), self.max_text_size), dtype=np.int64)
            for sample_ind, ind in enumerate(indexes):
                imgs[sample_ind, :, :, :], labels[sample_ind, :] = self.__getsample__(
                    ind
                )
            imgs = imgs.astype(np.float32) / 255.0
            labels = labels.astype(np.int64)
            return imgs, labels
        else:
            labels = [None] * len(indexes)
            for sample_ind, ind in enumerate(indexes):
                imgs[sample_ind, :, :, :], label = self.__getsample__(ind)
                labels[sample_ind] = label
            imgs = imgs.astype(np.float32) / 255.0
            return imgs, labels

    def __len__(self):
        return self.batches

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def main():
    source_files = [
        Path("/content/synt/ocr.json"),
    ]

    print()
    vocabulary = "-0127436LPN.СГОВПХТE58ROXл:B _C9ASTIVР"
    data_loader_train = DataLoader(
        source_files, vocabulary=vocabulary, split=80, work_mode="train"
    )

    data_loader_val = DataLoader(
        source_files, vocabulary=vocabulary, split=-20, work_mode="test"
    )

    print(f"Train batches: {len(data_loader_train)}")
    print(f"Validation batches: {len(data_loader_val)}")


if __name__ == "__main__":
    main()
