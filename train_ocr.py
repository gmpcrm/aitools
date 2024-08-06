from pathlib import Path
from datetime import datetime
import numpy as np
import argparse

import tensorflow as tf
from .dataloader_ocr import DataLoader

print(f"TensorFlow version: {tf.__version__}")
# tf.config.run_functions_eagerly(True)


class Config:

    def __init__(
        self,
        source_files,
        log_dir="logs",
        vocabulary="-012.LN436785ВПХPл9-СГОТEO:RXB_ CTASIVР",
        cosine_decay=True,
        cosine_decay_warmup_target=1e-4,
        cosine_decay_alpha=1e-12,
        cosine_decay_warmup_epochs=0,
        cosine_decay_initial_learning_rate=0.0,
        cosine_decay_restarts_initial_learning_rate=1e-4,
        cosine_decay_restarts_first_decay_epochs=10,
        cosine_decay_restarts_t_mul=1.5,
        cosine_decay_restarts_m_mul=0.7,
        cosine_decay_restarts_alpha=1e-12,
        save_best_only_check_point=True,
        epochs=300,
        batch_size=128,
        max_text_size=9,
        device="0",
        shape="200,50,3",
        dropout=0.5,
        augmentation=False,
        weights=None,
    ):
        self.source_files = source_files
        self.log_dir = log_dir
        self.vocabulary = vocabulary
        self.cosine_decay = cosine_decay
        self.cosine_decay_warmup_target = cosine_decay_warmup_target
        self.cosine_decay_alpha = cosine_decay_alpha
        self.cosine_decay_warmup_epochs = cosine_decay_warmup_epochs
        self.cosine_decay_initial_learning_rate = cosine_decay_initial_learning_rate
        self.cosine_decay_restarts_initial_learning_rate = (
            cosine_decay_restarts_initial_learning_rate
        )
        self.cosine_decay_restarts_first_decay_epochs = (
            cosine_decay_restarts_first_decay_epochs
        )
        self.cosine_decay_restarts_t_mul = cosine_decay_restarts_t_mul
        self.cosine_decay_restarts_m_mul = cosine_decay_restarts_m_mul
        self.cosine_decay_restarts_alpha = cosine_decay_restarts_alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.max_text_size = max_text_size
        self.shape = shape
        self.save_best_only_check_point = save_best_only_check_point
        self.dropout = dropout
        self.augmentation = augmentation
        self.weights = weights


class TrainModel:
    def __init__(self, config):
        self.model_name = "EfficientNetV2L_ocr_CosineDecay"
        self.config = config
        self.vocab = list(config.vocabulary)
        self.shape_inp_img = tuple([np.int32(i) for i in (config.shape.split(","))])
        self.train_dl = DataLoader(
            source_files=config.source_files,
            vocabulary=config.vocabulary,
            im_size=self.shape_inp_img,
            max_text_size=config.max_text_size,
            batch_size=config.batch_size,
            split=80,
            shuffle=True,
            augmentation=self.config.augmentation,
        )
        self.val_dl = DataLoader(
            source_files=config.source_files,
            vocabulary=config.vocabulary,
            im_size=self.shape_inp_img,
            max_text_size=config.max_text_size,
            batch_size=config.batch_size,
            split=-20,
            shuffle=False,
            augmentation=False,
        )

        self.model = self.build_model()
        if config.weights:
            self.model.load_weights(config.weights)

        self.lr_scheduler = self.init_lr_scheduler()
        self.opt = tf.keras.optimizers.Adam(self.lr_scheduler)

        self.model.compile(
            optimizer=self.opt,
            loss=self.Loss_CTC,
            metrics=[Symbols_recognized(), Imgs_recognized()],
        )
        self.logdir = Path(config.log_dir) / datetime.now().strftime(
            f"{self.model_name}__%d_%m_%Y__%H_%M_%S"
        )
        self.check_point_dir = self.logdir / "checkpoints"
        Path.mkdir(self.check_point_dir, parents=True)

    def build_model(self):
        shape = (self.shape_inp_img[0], self.shape_inp_img[1], self.shape_inp_img[2])
        input_img = tf.keras.layers.Input(
            shape=shape,
            name="image",
            dtype="float32",
        )
        base_model = tf.keras.applications.EfficientNetV2L(
            include_top=False,
            weights=None,
            input_shape=(200, 50, 3),
            input_tensor=input_img,
            include_preprocessing=True,
        )
        x = []
        for layer in base_model.layers:
            if layer.name == "block6a_expand_activation":
                print(layer)
                x = tf.keras.layers.Reshape(
                    target_shape=(
                        (
                            layer.output.shape[1],  # 13 символов максимум
                            int(layer.output.shape[2] * layer.output.shape[3]),
                        )
                    )
                )(layer.output)
                break

        x = tf.keras.layers.Dropout(self.config.dropout)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256,
                return_sequences=True,
                dropout=self.config.dropout,
            ),
            merge_mode="ave",
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        output = tf.keras.layers.Dense(
            len(self.vocab) + 2,
            activation=tf.keras.layers.LeakyReLU(alpha=0.3),
            kernel_initializer="lecun_normal",
            # kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name="dense1",
        )(x)

        model = tf.keras.models.Model(
            inputs=input_img, outputs=output, name="EfficientNetV2L_ocr_v1"
        )

        return model

    def Loss_CTC(self, y_true, y_prediction):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_prediction)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len,), dtype="int64")

        y_true = tf.cast(y_true, dtype="int64")

        loss = tf.nn.ctc_loss(
            logits_time_major=False,
            labels=y_true,
            logits=y_prediction,
            label_length=label_length,
            logit_length=input_length,
            blank_index=0,
        )

        return tf.math.reduce_mean(loss)

    def init_lr_scheduler(self):
        if self.config.cosine_decay:
            if self.config.cosine_decay_warmup_epochs == 0:
                return tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.config.cosine_decay_warmup_target,
                    decay_steps=len(self.train_dl) * (self.config.epochs),
                    alpha=self.config.cosine_decay_alpha,
                )
            else:
                return tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.config.cosine_decay_initial_learning_rate,
                    decay_steps=len(self.train_dl)
                    * (self.config.epochs - self.config.cosine_decay_warmup_epochs),
                    alpha=self.config.cosine_decay_alpha,
                    warmup_target=self.config.cosine_decay_warmup_target,
                    warmup_steps=len(self.train_dl)
                    * self.config.cosine_decay_warmup_epochs,
                )
        else:
            return tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.config.cosine_decay_restarts_initial_learning_rate,
                first_decay_steps=self.config.cosine_decay_restarts_first_decay_epochs
                * len(self.train_dl),
                t_mul=self.config.cosine_decay_restarts_t_mul,
                m_mul=self.config.cosine_decay_restarts_m_mul,
                alpha=self.config.cosine_decay_restarts_alpha,
            )

    def train(self):
        # self.model.summary()

        tensorboard_cbk = tf.keras.callbacks.TensorBoard(
            log_dir=self.logdir,
            histogram_freq=0,
            update_freq="epoch",
            write_graph=True,
        )

        if self.config.save_best_only_check_point:
            filepath_check_point = str(self.check_point_dir / "best.keras")
            save_best = True
        else:
            filepath_check_point = str(
                self.check_point_dir
                / "{epoch:03d}--val_imgs_recognized-{val_imgs_recognized:.4f}.keras"
            )
            save_best = False

        check_point_cbk = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath_check_point,
            monitor="val_imgs_recognized",
            mode="max",
            save_best_only=save_best,
            save_weights_only=False,
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_imgs_recognized",
            mode="max",
            patience=20,
            restore_best_weights=True,
        )

        self.model.fit(
            self.train_dl,
            validation_data=self.val_dl,
            validation_freq=1,
            verbose=2,
            epochs=self.config.epochs,
            callbacks=[tensorboard_cbk, check_point_cbk, early_stopping],
            # max_queue_size=512,
            # workers=8,
            # use_multiprocessing=True,
        )


class Imgs_recognized(tf.keras.metrics.Metric):
    def __init__(self, name="imgs_recognized", **kwargs):
        super(Imgs_recognized, self).__init__(name=name, **kwargs)
        self.false_rec_imgs = self.add_weight(
            name="false_rec_imgs", initializer="zeros"
        )
        self.all_imgs = self.add_weight(name="all_imgs", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        seq_length = tf.cast(
            input_len * tf.ones(shape=(batch_len,), dtype="int64"), dtype="int32"
        )
        y_pred_decode_st = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=seq_length,
            merge_repeated=True,
            blank_index=0,
        )[0][0]
        st = tf.SparseTensor(
            y_pred_decode_st.indices, y_pred_decode_st.values, (batch_len, input_len)
        )
        y_pred_decode = tf.sparse.to_dense(
            sp_input=st, default_value=tf.cast(tf.shape(y_pred)[2] - 1, dtype=tf.int64)
        )

        res_in_batch_bool = tf.logical_not(
            tf.equal(y_true, y_pred_decode[:, :9])
        )  # Здесь True это правильно распознанные символы
        res_in_batch_num = tf.cast(
            res_in_batch_bool, tf.int32
        )  # А здесь False это правильно распознанные символы

        # Кол-во плохо распознанных объектов (минимум 1 ошибка)
        self.false_rec_imgs.assign_add(
            tf.reduce_sum(tf.reduce_max(res_in_batch_num, axis=1))
        )
        self.all_imgs.assign_add(tf.cast(batch_len, dtype=tf.int32))

    def result(self):
        # Доля правильно распознанных изображений в батче
        return 1.0 - tf.divide(self.false_rec_imgs, self.all_imgs)

    def reset_state(self):
        self.false_rec_imgs.assign(0)
        self.all_imgs.assign(0)


class Symbols_recognized(tf.keras.metrics.Metric):
    def __init__(self, name="symbols_recognized", **kwargs):
        super(Symbols_recognized, self).__init__(name=name, **kwargs)
        self.false_rec_symbols = self.add_weight(
            name="false_rec_symbols", initializer="zeros"
        )
        self.all_symbols = self.add_weight(name="all_symbols", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        seq_length = tf.cast(
            input_len * tf.ones(shape=(batch_len,), dtype="int64"), dtype="int32"
        )
        y_pred_decode_st = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=seq_length,
            merge_repeated=True,
            blank_index=0,
        )[0][0]
        st = tf.SparseTensor(
            y_pred_decode_st.indices, y_pred_decode_st.values, (batch_len, input_len)
        )
        y_pred_decode = tf.sparse.to_dense(
            sp_input=st, default_value=tf.cast(tf.shape(y_pred)[2] - 1, dtype=tf.int64)
        )

        res_in_batch_bool = tf.logical_not(
            tf.equal(y_true, y_pred_decode[:, :9])
        )  # True - правильно распознанные символы
        res_in_batch_num = tf.cast(
            res_in_batch_bool, tf.int32
        )  # False - правильно распознанные символы

        # Кол-во плохо распознанных символов
        self.false_rec_symbols.assign_add(tf.reduce_sum(res_in_batch_num))
        self.all_symbols.assign_add(
            tf.shape(y_true)[0] * tf.shape(res_in_batch_bool)[1]
        )

    def result(self):
        # Доля правильно распознанных символов во всём батче
        return 1.0 - tf.divide(self.false_rec_symbols, self.all_symbols)

    def reset_state(self):
        self.false_rec_symbols.assign(0)
        self.all_symbols.assign(0)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    trainer = TrainModel(config)
    trainer.train()
    return trainer


def main():
    base = "/proplex"
    source_files = [
        f"{base}/label/ocr.json",
        f"{base}/label1/ocr.json",
        f"{base}/label2/ocr.json",
        f"{base}/label3/ocr.json",
        f"{base}/label4/ocr.json",
    ]
    log_dir = f"{base}/logs"

    weights = f"g:/My Drive/AIProplex/datasets/010/logs/EfficientNetV2L_ocr_CosineDecay__05_08_2024__21_25_59/checkpoints/best.keras"
    config = Config(
        source_files=source_files,
        log_dir=log_dir,
        weights=weights,
        batch_size=64,
        epochs=300,
        cosine_decay=False,
        cosine_decay_restarts_initial_learning_rate=1e-3,
        cosine_decay_restarts_first_decay_epochs=10,
    )
    parser = argparse.ArgumentParser(description="Утилита для обучения модели OCR")

    parser.add_argument(
        "--log_dir",
        type=str,
        default=config.log_dir,
        help="Путь к папке для сохранения логов",
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        default=config.vocabulary,
        help="Словарь символов",
    )
    parser.add_argument(
        "--cosine_decay",
        type=bool,
        default=config.cosine_decay,
        help="Использовать CosineDecay для изменения шага обучения",
    )
    parser.add_argument(
        "--cosine_decay_warmup_target",
        type=float,
        default=config.cosine_decay_warmup_target,
        help="Целевая скорость обучения для CosineDecay во время разогрева",
    )
    parser.add_argument(
        "--cosine_decay_alpha",
        type=float,
        default=config.cosine_decay_alpha,
        help="Минимальная скорость обучения для CosineDecay",
    )
    parser.add_argument(
        "--cosine_decay_warmup_epochs",
        type=int,
        default=config.cosine_decay_warmup_epochs,
        help="Количество эпох для разогрева CosineDecay",
    )
    parser.add_argument(
        "--cosine_decay_initial_learning_rate",
        type=float,
        default=config.cosine_decay_initial_learning_rate,
        help="Начальная скорость обучения для CosineDecay",
    )
    parser.add_argument(
        "--cosine_decay_restarts_initial_learning_rate",
        type=float,
        default=config.cosine_decay_restarts_initial_learning_rate,
        help="Начальная скорость обучения для CosineDecayRestarts",
    )
    parser.add_argument(
        "--cosine_decay_restarts_first_decay_epochs",
        type=int,
        default=config.cosine_decay_restarts_first_decay_epochs,
        help="Количество эпох для первого спада CosineDecayRestarts",
    )
    parser.add_argument(
        "--cosine_decay_restarts_t_mul",
        type=float,
        default=config.cosine_decay_restarts_t_mul,
        help="Фактор умножения для CosineDecayRestarts",
    )
    parser.add_argument(
        "--cosine_decay_restarts_m_mul",
        type=float,
        default=config.cosine_decay_restarts_m_mul,
        help="Фактор умножения для начальной скорости обучения CosineDecayRestarts",
    )
    parser.add_argument(
        "--cosine_decay_restarts_alpha",
        type=float,
        default=config.cosine_decay_restarts_alpha,
        help="Минимальная скорость обучения для CosineDecayRestarts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.epochs,
        help="Количество эпох обучения",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.batch_size,
        help="Размер батча",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        help="CUDA устройство для обучения",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=config.shape,
        help="Форма входного изображения",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
