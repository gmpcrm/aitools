from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Параметры
initial_learning_rate = 1e-3
final_learning_rate = 1e-4
warmup_epochs = 5
total_epochs = 50
steps_per_epoch = 181
decay_power = 0.8  # Параметр для регулирования уровня затухания амплитуды

# Вычисление шагов
warmup_steps = warmup_epochs * steps_per_epoch
total_steps = total_epochs * steps_per_epoch


# Создание расписания скорости обучения с разогревом и затуханием
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        final_learning_rate,
        warmup_steps,
        steps_per_epoch,
        total_epochs,
        decay_power,
    ):
        super(WarmUpCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.warmup_steps = warmup_steps
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.decay_power = decay_power

    def __call__(self, step):
        def warmup_phase():
            return self.initial_learning_rate * (
                tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
            )

        def decay_phase():
            epoch = (step - self.warmup_steps) // self.steps_per_epoch
            epoch_step = (step - self.warmup_steps) % self.steps_per_epoch
            decay_steps = self.steps_per_epoch
            cosine_decay = 0.5 * (
                1
                + tf.cos(
                    np.pi
                    * tf.cast(epoch_step, tf.float32)
                    / tf.cast(decay_steps, tf.float32)
                )
            )
            decayed = (
                1 - self.final_learning_rate / self.initial_learning_rate
            ) * cosine_decay + (self.final_learning_rate / self.initial_learning_rate)
            return (
                self.initial_learning_rate
                * decayed
                * tf.pow(
                    (
                        1
                        - (
                            tf.cast(epoch, tf.float32)
                            / tf.cast(self.total_epochs, tf.float32)
                        )
                    ),
                    self.decay_power,
                )
            )

        return tf.cond(step < self.warmup_steps, warmup_phase, decay_phase)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "warmup_steps": self.warmup_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "total_epochs": self.total_epochs,
            "decay_power": self.decay_power,
        }


learning_rate_schedule = WarmUpCosineDecay(
    initial_learning_rate=initial_learning_rate,
    final_learning_rate=final_learning_rate,
    warmup_steps=warmup_steps,
    steps_per_epoch=steps_per_epoch,
    total_epochs=total_epochs,
    decay_power=decay_power,
)

# Пример использования в оптимизаторе
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

# Компиляция и обучение модели
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Загрузка данных и обучение модели
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

model.fit(
    x_train,
    y_train,
    epochs=total_epochs,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback],
)

# Генерация значений скорости обучения для каждого шага для построения графика
steps = np.arange(total_steps)
lr_values = [learning_rate_schedule(step).numpy() for step in steps]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(steps, lr_values, label="Learning Rate")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.title("WarmUp Cosine Decay Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.show()
