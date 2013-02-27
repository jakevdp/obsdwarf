import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# From astroML.plotting.mcmc
def convert_to_stdev(logL):
    """
    Given a grid of log-likelihood values, convert them to cumulative
    standard deviation.  This is useful for drawing contours from a
    grid of likelihoods.
    """
    sigma = np.exp(logL)

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)


class OffsetLikelihood:
    @classmethod
    def from_file(cls, which='1e-06', cut=True):
        X_u = np.loadtxt('Offset_jake_u_%s.txt' % which, unpack=True)
        X_s = np.loadtxt('Offset_jake_s_%s.txt' % which, unpack=True)

        if cut:
            i_u = X_u[1] < 10
            i_s = X_s[1] < 10
            return cls(X_u[0, i_u], X_u[1, i_u], X_u[2, i_u],
                       X_s[0, i_s], X_s[1, i_s], X_s[2, i_s])
        else:
            return cls(X_u[0], X_u[1], X_u[2] * 1.2,
                       X_s[0], X_s[1], X_s[2] * 1.2)

    @classmethod
    def simulated(cls, sigma_int=1.0, sigma_MG=0.5,
                  seed=0, dmin=2, dmax=40, Nu=85, Ns=160, d_factor=0.1):
        np.random.seed(0)
        D_u = dmin + (dmax - dmin) * np.random.random(Nu)
        D_s = dmin + (dmax - dmin) * np.random.random(Ns)
        
        dS_u = d_factor * D_u
        dS_s = d_factor * D_s

        S_u = abs(np.random.normal(0, np.sqrt(dS_u ** 2 + sigma_int ** 2
                                              + sigma_MG ** 2)))
        S_s = abs(np.random.normal(0, np.sqrt(dS_s ** 2 + sigma_int ** 2)))

        return cls(D_u, S_u, dS_u, D_s, S_s, dS_s)

    def __init__(self, D_u, S_u, dS_u,
                 D_s, S_s, dS_s):
        self.D_u, self.S_u, self.dS_u, self.D_s, self.S_s, self.dS_s =\
            map(np.asarray, (D_u, S_u, dS_u, D_s, S_s, dS_s))

    def randomize_offsets(self, sigma_int=1.0, sigma_MG=0.5, d_factor=0.1):
        self.dS_u = d_factor * self.D_u
        self.dS_s = d_factor * self.D_s
        self.S_u = abs(np.random.normal(0, np.sqrt(self.dS_u ** 2
                                                   + sigma_int ** 2
                                                   + sigma_MG ** 2)))
        self.S_s = abs(np.random.normal(0, np.sqrt(self.dS_s ** 2
                                                   + sigma_int ** 2)))

    def scatter_obs(self, plot_best_fit=True):
        fig, ax = plt.subplots()
        ax.errorbar(self.D_u, self.S_u, self.dS_u,
                    fmt='.r', ecolor='#FFAAAA', label='unscreened')
        ax.errorbar(self.D_s, self.S_s, self.dS_s,
                    fmt='.b', ecolor='#AAAAFF', label='screened')
        
        ax.set_xlabel('D (Mpc)')
        ax.set_ylabel('S (kpc)')
        ax.legend(loc=2)

    def logL(self, v):
        """v is a vector, [sigma_int, sigma_MG]"""
        sigma_s2 = v[0] ** 2 + self.dS_s ** 2
        sigma_u2 = v[0] ** 2 + v[1] ** 2 + self.dS_u ** 2

        logL_s = -0.5 * np.sum(np.log(2 * np.pi * sigma_s2)
                               + (self.S_s ** 2) / sigma_s2)
        logL_u = -0.5 * np.sum(np.log(2 * np.pi * sigma_u2)
                               + (self.S_u ** 2) / sigma_u2)
        return logL_s + logL_u

    def logL_grid(self, sigma_int, sigma_MG):
        logL = np.zeros((len(sigma_int), len(sigma_MG)))
        for i in range(len(sigma_int)):
            for j in range(len(sigma_MG)):
                logL[i, j] = self.logL([sigma_int[i], sigma_MG[j]])
        return logL

    def plot_logL_grid(self, sigma_int, sigma_MG):
        logL = self.logL_grid(sigma_int, sigma_MG)

        fig, ax = plt.subplots()
        im = ax.contourf(sigma_int, sigma_MG, np.exp(logL).T, 50)
        ax.set_xlabel(r'$\sigma_{\rm int}$')
        ax.set_ylabel(r'$\sigma_{\rm MG}$')
        cb = fig.colorbar(im)
        cb.set_label('Likelihood')

    def plot_posterior(self, sigma_int, sigma_MG, ax=None, **kwargs):
        posterior = convert_to_stdev(self.logL_grid(sigma_int, sigma_MG))

        if ax is None:
            ax = plt.gca()
        im = ax.contour(sigma_int, sigma_MG, posterior.T,
                        [0.63, 0.95, 0.997], **kwargs)
        ax.set_xlabel('astrophysical scatter')
        ax.set_ylabel('scatter due to MG')

    def maximize_logL(self, x0=None):
        if x0 is None:
            x0 = [0.5, 0.5]
        x_best = optimize.fmin(lambda x: -self.logL(x), x0)
        return abs(np.asarray(x_best))

    def bootstrap(self, Nbootstraps, rseed=None):
        if rseed is not None:
            np.random.seed(rseed)

        S_s = self.S_s
        dS_s = self.dS_s
        S_u = self.S_u
        dS_u = self.dS_u

        i_s = np.random.randint(0, len(S_s), (Nbootstraps, len(S_s)))
        i_u = np.random.randint(0, len(S_u), (Nbootstraps, len(S_u)))
        maxvals = np.zeros((Nbootstraps, 2))

        for i in range(Nbootstraps):
            self.S_s = S_s[i_s[i]]
            self.dS_s = dS_s[i_s[i]]
            self.S_u = S_u[i_u[i]]
            self.dS_u = dS_u[i_u[i]]
            maxvals[i] = self.maximize_logL()

        self.S_s = S_s
        self.dS_s = dS_s
        self.S_u = S_u
        self.dS_u = dS_u

        return maxvals


def plot_data_hist():
    fig = plt.figure(figsize=(8, 4))
    scatter_ax = [fig.add_axes((0.09, 0.15, 0.4, 0.59)),
                  fig.add_axes((0.55, 0.15, 0.4, 0.59))]

    hist_ax = [fig.add_axes((0.09, 0.76, 0.4, 0.2)),
               fig.add_axes((0.55, 0.76, 0.4, 0.2))]


    for i, cutoff in enumerate(['1e-06', '2e-07']):
        osl = OffsetLikelihood.from_file(cutoff)

        scatter_ax[i].errorbar(osl.S_s, osl.D_s, xerr=osl.dS_s,
                               fmt='ok', ecolor='#BBBBBB', ms=4)
        scatter_ax[i].errorbar(osl.S_u, osl.D_u, xerr=osl.dS_u,
                               fmt='or', ecolor='#FFAAAA', ms=4)

        #scatter_ax[i].plot(osl.S_s / osl.dS_s, osl.D_s,
        #                   'ok', ms=4)
        #scatter_ax[i].plot(osl.S_u / osl.dS_u, osl.D_u,
        #                   'or', ms=4)


        scatter_ax[i].text(0.95, 0.15, r'${\rm N_S = %i}$' % len(osl.S_s),
                           ha='right', va='bottom',
                           color='black', transform=scatter_ax[i].transAxes)
        scatter_ax[i].text(0.95, 0.05, r'${\rm N_U = %i}$' % len(osl.S_u),
                           ha='right', va='bottom',
                           color='red', transform=scatter_ax[i].transAxes)

        scatter_ax[i].set_ylim(0, 44)
        scatter_ax[i].set_xlim(-2, 15)

        scatter_ax[i].set_xlabel('offset (kpc)')
    
        hist_ax[i].hist(osl.S_s, np.linspace(-2, 15, 30),
                        histtype='stepfilled', fc='k', alpha=0.3)
        hist_ax[i].hist(osl.S_u, np.linspace(-2, 15, 30),
                        histtype='stepfilled', fc='r', alpha=0.3)
        hist_ax[i].set_ylim(0, 40)


        hist_ax[i].text(0.98, 0.9, "$f_{R0} = 10^{-%s}$" % cutoff[-1],
                        ha='right', va='top', transform=hist_ax[i].transAxes)
        
        hist_ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        hist_ax[i].yaxis.set_major_locator(plt.MultipleLocator(20))
        

    scatter_ax[0].set_ylabel('distance (Mpc)')
    hist_ax[0].set_ylabel('N')

    
    fig.savefig('sec4.1_scatter.pdf')
    


def plot_likelihood():
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    kwargs = [dict(colors='black', linewidths=2, linestyles='solid'),
              dict(colors='gray', linewidths=1, linestyles='solid')]
    
    for kwds, cutoff in zip(kwargs, ['1e-06', '2e-07']):
        osl = OffsetLikelihood.from_file(cutoff)
        osl.plot_posterior(np.linspace(-0.05, 2.0, 50),
                           np.linspace(-0.05, 2.5, 49),
                           ax=ax[0], **kwds)

        np.random.seed(0)
        osl.randomize_offsets(d_factor=0.01)
        osl.plot_posterior(np.linspace(-0.05, 2.0, 50),
                           np.linspace(-0.05, 2.5, 49),
                           ax=ax[1], **kwds)

    ax[0].set_title('Observed')
    ax[1].set_title('Simulated (10x accuracy)')
    ax[1].plot([1.0], [0.5], 'xk')
    ax[1].plot([0], [0], '-', c='gray', lw=1,
               label=r'$f_{R0} = 2 \times 10^{-7}$')
    ax[1].plot([0], [0], '-', c='black', lw=2,
               label='$f_{R0} = 10^{-6}$')
    ax[1].legend(loc=1)

    fig.savefig('Offset_fig.pdf')
        
        

if __name__ == '__main__':
    plot_likelihood()
    plot_data_hist()
    plt.show()
