import matplotlib.pyplot as plt
import numpy as np

# Параметры
initial_learning_rate = 1e-4
first_decay_steps = 10 * 181  # 10 эпох по 181 шагу в каждой
t_mul = 0.97  # 1.5
m_mul = 0.92  # 0.7
alpha = 1e-12
total_steps = 181 * 300  # Общее количество шагов для визуализации


# Функция для вычисления значения cosine decay restarts
def cosine_decay_restarts(
    step, initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha
):
    completed_fraction = step / first_decay_steps
    i_restart = np.floor(np.log(1 - completed_fraction * (1 - t_mul)) / np.log(t_mul))
    sum_r = (1 - t_mul**i_restart) / (1 - t_mul)
    completed_fraction = (step - first_decay_steps * sum_r) / (
        first_decay_steps * t_mul**i_restart
    )
    cosine_decayed = 0.5 * (1 + np.cos(np.pi * completed_fraction))
    decayed = (1 - alpha) * cosine_decayed + alpha
    decayed_learning_rate = initial_learning_rate * m_mul**i_restart * decayed
    return decayed_learning_rate


# Генерация значений скорости обучения для каждого шага
learning_rates = [
    cosine_decay_restarts(
        step, initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha
    )
    for step in range(total_steps)
]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(learning_rates)
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.title("Cosine Decay Restarts Learning Rate Schedule")
plt.grid(True)
plt.show()
