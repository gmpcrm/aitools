import matplotlib.pyplot as plt
import numpy as np

# Параметры
initial_learning_rate = 1e-3
final_learning_rate = 1e-4
warmup_epochs = 5
total_epochs = 300
steps_per_epoch = 181
decay_power = 0.8  # Новый параметр для регулирования уровня затухания

# Вычисление шагов
warmup_steps = warmup_epochs * steps_per_epoch
total_steps = total_epochs * steps_per_epoch


# Создание расписания скорости обучения с разогревом и затуханием
class WarmUpCosineDecay:
    def __init__(
        self,
        initial_learning_rate,
        final_learning_rate,
        warmup_steps,
        steps_per_epoch,
        total_epochs,
        decay_power,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.warmup_steps = warmup_steps
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.decay_power = decay_power

    def __call__(self, step):
        # Фаза разогрева
        if step < self.warmup_steps:
            return self.initial_learning_rate * (step / self.warmup_steps)
        # Фаза затухания с перезапусками каждую эпоху
        epoch = (step - self.warmup_steps) // self.steps_per_epoch
        epoch_step = (step - self.warmup_steps) % self.steps_per_epoch
        decay_steps = self.steps_per_epoch
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_step / decay_steps))
        decayed = (
            1 - self.final_learning_rate / self.initial_learning_rate
        ) * cosine_decay + (self.final_learning_rate / self.initial_learning_rate)
        return (
            self.initial_learning_rate
            * decayed
            * ((self.total_epochs - epoch) / self.total_epochs) ** self.decay_power
        )


# Создание экземпляра расписания скорости обучения
learning_rate_schedule = WarmUpCosineDecay(
    initial_learning_rate=initial_learning_rate,
    final_learning_rate=final_learning_rate,
    warmup_steps=warmup_steps,
    steps_per_epoch=steps_per_epoch,
    total_epochs=total_epochs,
    decay_power=decay_power,  # Передаем новый параметр
)

# Генерация значений скорости обучения для каждого шага
steps = np.arange(total_steps)
lr_values = [learning_rate_schedule(step) for step in steps]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(steps, lr_values, label="Learning Rate")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.title("WarmUp Cosine Decay Learning Rate Schedule with Adjustable Decay")
plt.legend()
plt.grid(True)
plt.show()
