# Andrew Enrique Oliveira
# Ciência da Computação - Universidade Federal de Itajubá (2017 - )
# 02/03/2021

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Model():
    def __init__(self, x_samples, y_samples, thetas):
        '''
        Constructor takes
        x_samples as a list of features,
        y_samples as a list of correspondent values to X and
        thetas as a list of all thetas generated in the calculation
        '''
        self.thetas = thetas
        self.x_samples = x_samples
        self.y_samples = y_samples

    def animate(self, fileName):
            print('plot_model.animate(): Generating gif...')
            fig, ax = plt.subplots()

            x_data = np.array(self.x_samples)
            y_data = np.array(self.y_samples)
            plt.scatter(x_data, y_data)

            max_number = max(np.amax(x_data), np.amax(y_data))
            min_number = min(np.amin(x_data), np.amin(y_data))

            # Initial plot
            x = np.arange(0 if min_number > 0 else min_number, max_number, 0.2)
            y = np.arange(0 if min_number > 0 else min_number, max_number, 0.2)
            line, = ax.plot(x, y, '-r')

            def animate(i):
                if i >= len(self.thetas):
                    theta_index = len(self.thetas) - 1
                else:
                    theta_index = i

                t0 = self.thetas[theta_index][0]
                t1 = self.thetas[theta_index][1]
                line.set_xdata(x)
                line.set_ydata(t0 + t1 * x)
                
                line.set_label(r'$\theta_{0}$: ' + str(t0) + '\n' + r'$\theta_{1}$: ' + str(t1))
                ax.legend(loc='upper left')
                
                return line,

            surplus_frames = 20
            anima = FuncAnimation(fig, animate, frames=np.arange(0, len(self.thetas) + surplus_frames, 1), interval=100)
            anima.save(str(fileName)+'.gif', fps=15)

            print('plot_model.animate(): Success! ' + str(fileName)+'.gif ' + 'generated')

if __name__ == '__main__':
    exit()