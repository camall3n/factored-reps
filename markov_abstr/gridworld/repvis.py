import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RepVisualization:
    def __init__(self, env, obs, batch_size, n_dims, colors=None, cmap=None):
        self.env = env
        self.fig = plt.figure(figsize=(10, 8))
        self.cmap = cmap
        self.colors = colors
        self.n_dims = n_dims

        self.text_ax = self.fig.add_subplot(4, 4, 7)
        self.text_ax.set_xticks([])
        self.text_ax.set_yticks([])
        self.text_ax.axis('off')
        self.text_ax.set_ylim([0, 1])
        self.text_ax.set_xlim([0, 1])
        self.text = self.text_ax.text(0.05, 0.12, '')

        self.env_ax = self.fig.add_subplot(4, 4, 4)
        env.plot(self.env_ax)
        self.env_ax.set_title('Environment')

        self.obs_ax = self.fig.add_subplot(4, 4, 8)
        self.obs_ax.imshow(obs)
        self.obs_ax.set_xticks([])
        self.obs_ax.set_yticks([])
        self.obs_ax.set_title('Sampled observation (x)')

        self.effects = self._setup_effects(subplot=(4, 4, 3))

        z = np.zeros((batch_size, n_dims))

        self.scats = []
        for row in range(4):
            self.scats.append([])
            for col in range(row + 1):
                if (row + 1) < self.n_dims:
                    subplot_idx = 4 * row + col + 1
                    x_idx = col
                    y_idx = row + 1
                    _, phi_scat = self._plot_rep(z,
                                                 subplot=(4, 4, subplot_idx),
                                                 title=r'$\phi(x)$',
                                                 labels=(x_idx, y_idx))
                    self.scats[row].append(phi_scat)

        self.fig.tight_layout(pad=5.0, h_pad=1.1, w_pad=2.5)
        self.fig.show()

    def _plot_states(self, x, subplot=(1, 1, 1), title=''):
        ax = self.fig.add_subplot(*subplot)
        ax.scatter(x[:, 1], -x[:, 0], c=self.colors, cmap=self.cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def _plot_rep(self, z, subplot=(1, 1, 1), title='', labels=('x', 'y')):
        ax = self.fig.add_subplot(*subplot)
        x = z[:, 0]
        x_label = labels[0]
        y = z[:, 1]
        y_label = labels[1]
        scat = ax.scatter(x, y, c=self.colors, cmap=self.cmap)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlabel(r'$z_{}$'.format(x_label))
        ax.set_ylabel(r'$z_{}$'.format(y_label))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return ax, scat

    def _setup_effects(self, subplot=(1, 1, 1), title=''):
        ax = self.fig.add_subplot(*subplot)
        # ax.set_xlabel('action')
        ax.set_ylabel(r'$\Delta\ z$')
        ax.set_ylim([-2, 2])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _plot_effects(self, z0, z1, a, ax, title='', noise=False):
        ax.clear()
        ax.set_xlabel('action')
        ax.set_ylabel(r'$\Delta\ z$')
        ax.set_ylim([-2, 2])
        ax.set_title(title)
        n_dims = z0.shape[-1]
        dz_flat = (z1 - z0).flatten()
        if noise:
            dz_flat += noise * np.random.randn(len(dz_flat))
        a_flat = np.repeat(a, n_dims)
        var_flat = np.tile(np.arange(n_dims), len(a))
        sns.violinplot(x=a_flat,
                       y=dz_flat,
                       hue=var_flat,
                       inner=None,
                       dodge=False,
                       bw='silverman',
                       legend=False,
                       ax=ax)
        ax.get_legend().remove()

        ax.axhline(y=0, ls=":", c=".5")

        # Re-label legend entries
        # for i, t in enumerate(ax.legend_.texts):
        #     t.set_text(r'$z_{(' + str(i) + ')}$')
        plt.setp(ax.collections, alpha=.7)
        return ax

    def update_plots(self, z0, a, z1, text):
        for row in range(4):
            for col in range(row + 1):
                if (row + 1) < self.n_dims:
                    x_idx = col
                    y_idx = row + 1
                    z_projection = np.stack((z0[:, x_idx], z0[:, y_idx]), axis=1)
                    self.scats[row][col].set_offsets(z_projection)

        self.text.set_text(text)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # self._plot_effects(z0, z1_hat, a, ax=self.effects, title=r'$T(\phi(x),a) - \phi(x)$')
        self._plot_effects(z0, z1, a, ax=self.effects, title=r'$\phi(x\') - \phi(x)$')

        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3, ))
        return frame

class CleanVisualization:
    def __init__(self, env, obs, batch_size, n_dims, colors=None, cmap=None):
        self.env = env
        self.fig = plt.figure(figsize=(8, 8))
        self.cmap = cmap
        self.colors = colors

        z0 = np.zeros((batch_size, n_dims))
        z1 = np.zeros((batch_size, n_dims))

        plt.rcParams.update({'font.size': 22})

        self.phi_ax, self.phi_scat = self._plot_rep(z0, subplot=111, title='')

        self.fig.tight_layout()  #pad=5.0, h_pad=1.1, w_pad=2.5)
        self.fig.show()

    def _plot_rep(self, z, subplot=(1, 1, 1), title=''):
        ax = self.fig.add_subplot(*subplot)
        x = z[:, 0]
        y = z[:, 1]
        sc = ax.scatter(x, y, c=self.colors, cmap=self.cmap)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlabel(r'$z_0$')
        ax.set_ylabel(r'$z_1$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return ax, sc

    def update_plots(self, z0, a, z1, text):
        self.phi_scat.set_offsets(z0[:, :2])
        plt.rcParams.update({'font.size': 22})
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3, ))
        return frame
