import os
import cv2


def draw_bounding_boxes(source_folder, target_folder):
    images_folder = os.path.join(source_folder, "images")
    labels_folder = os.path.join(source_folder, "labels")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            # Загрузка изображения
            image_path = os.path.join(images_folder, image_file)
            image = cv2.imread(image_path)

            # Поиск соответствующего файла меток
            label_file = image_file.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(labels_folder, label_file)

            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])

                        img_height, img_width = image.shape[:2]
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height

                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)

                        # Отрисовка bounding box
                        color = (0, 255, 0)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            image,
                            str(class_id),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2,
                        )

            # Сохранение изображения с отрисованными bounding boxes
            output_path = os.path.join(target_folder, image_file)
            cv2.imwrite(output_path, image)


# Указание путей
source_folder = "g:/My Drive/AIProplex/testdata2/dataset/train"
source_folder = "g:/My Drive/AIProplex/datasets/001/Dataset/train"
target_folder = "c:/proplex/test"

# Запуск функции
draw_bounding_boxes(source_folder, target_folder)
