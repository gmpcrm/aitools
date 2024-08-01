import numpy as np
import matplotlib.pyplot as plt

# Параметры
epochs = 300
total_images = 10000
batch_size = 300
steps_per_epoch = total_images // batch_size
total_steps = epochs * steps_per_epoch

initial_learning_rate = 1e-3
t_mul = 1.5
m_mul = 0.7
alpha = 1e-12
first_decay_epochs = 3
first_decay_steps = steps_per_epoch * first_decay_epochs

global_step = np.arange(total_steps)
learning_rate = np.zeros_like(global_step, dtype=np.float32)


def cosine_decay_restarts(
    global_step, initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha
):
    completed_fraction = global_step / first_decay_steps
    i_restart = 0
    while completed_fraction >= (t_mul**i_restart):
        i_restart += 1
    completed_fraction = (
        completed_fraction / (t_mul ** (i_restart - 1))
        if i_restart > 0
        else completed_fraction
    )

    cosine_decay = 0.5 * (1 + np.cos(np.pi * completed_fraction))
    decayed_learning_rate = (
        initial_learning_rate * (m_mul**i_restart) - alpha
    ) * cosine_decay + alpha

    return decayed_learning_rate


for step in range(total_steps):
    learning_rate[step] = cosine_decay_restarts(
        step, initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha
    )

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(global_step, learning_rate, label="Learning Rate")
plt.xlabel("Global Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule with Cosine Decay Restarts")
plt.legend()
plt.grid(True)
plt.show()
