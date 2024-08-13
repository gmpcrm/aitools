import cv2
import numpy as np
import tensorflow as tf
import json
import time
from tqdm import tqdm  # Импорт tqdm для прогресс-бара

vocab = "*02137O64PX.TBC5L8:ERN-Г9Пл_SAIV "


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results_indx = tf.sparse.to_dense(
        tf.nn.ctc_greedy_decoder(
            tf.transpose(pred, perm=[1, 0, 2]), input_len, blank_index=0
        )[0][0]
    ).numpy()
    results_char = []
    for result in results_indx:
        result_char = []
        for indx in result:
            result_char.append(vocab[indx])
        results_char.append("".join(result_char))
    return results_char


def get_plate(plate, ocr_model):
    plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
    plate = cv2.resize(plate, (50, 200))
    imgs = np.zeros((1, 200, 50, 3), dtype=np.uint8)
    imgs[0, :, :, :] = plate

    start_time = time.time()  # Начало измерения времени
    pred_logits = ocr_model.predict_on_batch(imgs.astype(np.float32) / 255.0)
    end_time = time.time()  # Конец измерения времени

    pred_labels_chars = decode_batch_predictions(pred_logits)
    elapsed_time = end_time - start_time  # Время выполнения OCR модели
    return pred_labels_chars[0], elapsed_time


def validate_ocr(json_file, model_path):
    # Загружаем модель
    custom_objects = {
        "LeakyReLU": tf.keras.layers.LeakyReLU,
        "LSTM": tf.keras.layers.LSTM,
        "Bidirectional": tf.keras.layers.Bidirectional,
        "EfficientNetV2L": tf.keras.applications.EfficientNetV2L,
    }

    ocr_model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    ocr_model.compile()

    # Загружаем JSON файл
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Инициализация счетчиков
    correct_count = 0
    incorrect_count = 0
    total_time = 0
    processed_count = 0

    # Используем tqdm для отображения прогресса
    with tqdm(total=len(data["ocr_files"]), desc="Processing images") as pbar:
        for entry in data["ocr_files"]:
            file_path = entry["file"]
            expected_text = entry["text"]

            plate = cv2.imread(file_path)
            result, elapsed_time = get_plate(plate, ocr_model)

            if expected_text == result:
                correct_count += 1
            else:
                incorrect_count += 1

            total_time += elapsed_time
            processed_count += 1

            # Расчет процентов
            correct_pct = (correct_count / processed_count) * 100
            incorrect_pct = (incorrect_count / processed_count) * 100
            avg_time = total_time / processed_count

            # Обновляем tqdm с текущими значениями
            pbar.set_postfix(
                {
                    "C": f"{correct_pct:.2f}%",
                    "I": f"{incorrect_pct:.2f}%",
                    "A": f"{avg_time:.4f}",
                }
            )
            pbar.update(1)

    # Вывод окончательных результатов
    print(f"Total images processed: {processed_count}")
    print(f"Correctly recognized: {correct_count} ({correct_pct:.2f}%)")
    print(f"Incorrectly recognized: {incorrect_count} ({incorrect_pct:.2f}%)")
    print(f"Average OCR time per image: {total_time / processed_count:.4f} seconds")


if __name__ == "__main__":
    # Укажите путь к JSON файлу и модели
    json_file = "/content/ocr.json"
    model_path = "/projects/models/best.keras"

    validate_ocr(json_file, model_path)
