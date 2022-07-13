import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_gif(full, joint):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=50)

    data = [full, joint]
    ax_0 = axs[0].imshow(data[0][0])
    ax_1 = axs[1].imshow(data[1][0])
    ax_s = [ax_0, ax_1]

    def animate(i):
        ax_s[0].set_data(data[0][i])
        ax_s[1].set_data(data[1][i])

        return ax_s

    anim = animation.FuncAnimation(fig, animate, frames=999, interval=0.05)
    anim.save('test_anim.gif', fps=25)



