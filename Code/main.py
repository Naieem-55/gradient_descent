import os
import imageio
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 7*x + 10

grad_f = grad(f)

def gradient_descent(initial_x, learning_rate, iterations, save_folder):
    x = initial_x
    image_paths = []

    for i in range(iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        print("Iteration {}: x = {}, f(x) = {}".format(i+1, x, f(x)))

        if (i + 1) % 10 == 0:
            plt.figure()
            plt.plot(np.linspace(-10, 10, 100), f(np.linspace(-10, 10, 100)), label='f(x)')
            plt.scatter(x, f(x), color='red', label='Current Point')
            plt.title("Iteration {}".format(i + 1))
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()

            image_path = os.path.join(save_folder, 'iteration_{}.png'.format(i + 1))
            plt.savefig(image_path)
            plt.close()
            image_paths.append(image_path)

    return image_paths

initial_x = 0.0
learning_rate = 0.01
iterations = 1000
save_folder = r'C:\Users\Hp\OneDrive\Desktop\ML Lab\Images'
animation_folder = r'C:\Users\Hp\OneDrive\Desktop\ML Lab\Animation' 

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_paths = gradient_descent(initial_x, learning_rate, iterations, save_folder)
print("Images saved. Creating GIF.......")

with imageio.get_writer(os.path.join(animation_folder, 'Gradient_descent.gif'), mode='I') as writer:
    for image_path in image_paths:
        image = imageio.imread(image_path)
        writer.append_data(image)

print("GIF created successfully.")
