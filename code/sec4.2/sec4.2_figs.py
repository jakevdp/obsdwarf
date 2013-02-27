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
    sigma = np.exp(logL - logL.max())

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)


def get_data(vlim=1000, which='-6.5',
             catalog='PersictoJake_section42.dat'):
    X = np.loadtxt(catalog, skiprows=1, dtype=float)

    # truncate catalog
    vmax = X[:, 0]
    X = X[vmax < vlim]

    vmax, offset, optical, distance, phi1, phi2, phi3 = X.T

    rel_offset = offset / optical
    rel_offset_err = 0.5 / optical

    fR0 = 10 ** float(which)

    if which == '-7':
        unscreened = ((vmax / 100.)**2. < fR0/2e-7) & (phi1 < fR0)

    elif which == '-6.5':
        unscreened = ((vmax / 100.)**2. <= fR0/2e-7) & (phi2 <= fR0)

    elif which == '-6':
        unscreened = ((vmax / 100.)**2. < fR0/2e-7) & (phi3 < fR0)

    else:
        raise ValueError, 'fR0 does not correspond to any of the known scale' 

    screened = ~unscreened

    return distance, rel_offset, rel_offset_err, vmax, screened


class OffsetLikelihood:
    @classmethod
    def from_file(cls, which='-6.5', vlim=200):
        D, S, dS, vmax, screened = get_data(vlim=vlim, which=which)
        return cls(D[~screened], S[~screened], dS[~screened],
                   D[screened], S[screened], dS[screened])

    def __init__(self, D_u, S_u, dS_u, D_s, S_s, dS_s):
        self.D_u, self.S_u, self.dS_u, self.D_s, self.S_s, self.dS_s =\
            map(np.asarray, (D_u, S_u, dS_u, D_s, S_s, dS_s))

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

    def plot_logL_grid(self, sigma_int, sigma_MG, ax=None, **kwargs):
        logL = self.logL_grid(sigma_int, sigma_MG)

        if ax is None:
            ax = plt.gca()
        im = ax.contourf(sigma_int, sigma_MG, logL.T, 50)
        ax.set_xlabel(r'$\sigma_{\rm int}$')
        ax.set_ylabel(r'$\sigma_{\rm MG}$')
        return im

    def plot_posterior(self, sigma_int, sigma_MG, ax=None, **kwargs):
        posterior = convert_to_stdev(self.logL_grid(sigma_int, sigma_MG))

        if ax is None:
            ax = plt.gca()
        im = ax.contour(sigma_int, sigma_MG, posterior.T,
                        [0.63, 0.95, 0.997], **kwargs)
        ax.set_xlabel('relative astrophysical scatter')
        ax.set_ylabel('relative scatter due to MG')

    def maximize_logL(self, x0=None):
        if x0 is None:
            x0 = [0.5, 0.5]
        x_best = optimize.fmin(lambda x: -self.logL(x), x0)
        return abs(np.asarray(x_best))


def plot_data():
    fig = plt.figure(figsize=(8, 4))
    scatter_ax = [fig.add_axes((0.09, 0.15, 0.4, 0.59)),
                  fig.add_axes((0.55, 0.15, 0.4, 0.59))]

    hist_ax = [fig.add_axes((0.09, 0.76, 0.4, 0.2)),
               fig.add_axes((0.55, 0.76, 0.4, 0.2))]

    for ax in hist_ax:
        ax.xaxis.set_major_formatter(plt.NullFormatter())

    hist_ax[0].yaxis.set_major_locator(plt.MultipleLocator(20))
    hist_ax[1].yaxis.set_major_locator(plt.MultipleLocator(25))

    #fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    #fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

    D, S, dS, vmax, screened = get_data(vlim=1000, which='-6.5')

    scatter_ax[0].plot(D[screened], vmax[screened], 'k.', ms=4)
    scatter_ax[0].plot(D[~screened], vmax[~screened], 'r.', ms=4)
    scatter_ax[0].plot([0, 150], [200, 200], '--', color='gray')

    hist_ax[0].hist(D[screened], np.linspace(0, 140, 30),
                    histtype='stepfilled', fc='k', alpha=0.3)
    hist_ax[0].hist(D[~screened], np.linspace(0, 140, 30),
                    histtype='stepfilled', fc='r', alpha=0.3)

    scatter_ax[1].plot(S[screened], vmax[screened], 'k.', ms=4)
    scatter_ax[1].plot(S[~screened], vmax[~screened], 'r.', ms=4)
    scatter_ax[1].plot([-0.01, 0.2], [200, 200], '--', color='gray')

    hist_ax[1].hist(S[screened], np.linspace(0, 0.2, 20),
                    histtype='stepfilled', fc='k', alpha=0.3)
    hist_ax[1].hist(S[~screened], np.linspace(0, 0.2, 20),
                    histtype='stepfilled', fc='r', alpha=0.3)

    scatter_ax[0].set_xlim(0, 140)
    hist_ax[0].set_xlim(0, 140)

    scatter_ax[1].set_xlim(-0.01, 0.2)
    hist_ax[1].set_xlim(-0.01, 0.2)

    scatter_ax[0].set_ylim(0, 390)
    scatter_ax[1].set_ylim(0, 390)

    hist_ax[1].set_ylim(0, 100)

    scatter_ax[0].set_ylabel('Rotation velocity (km/s)')
    hist_ax[0].set_ylabel('N')
    scatter_ax[0].set_xlabel('Distance (Mpc)')
    scatter_ax[1].set_xlabel('Offset / Optical radius')

    hist_ax[1].text(0.95, 0.9, r'${\rm N_S = %i}$' % sum(screened),
                    ha='right', va='top',
                    color='black', transform=hist_ax[1].transAxes)
    hist_ax[1].text(0.95, 0.6, r'${\rm N_U = %i}$' % sum(~screened),
                    ha='right', va='top',
                    color='red', transform=hist_ax[1].transAxes)
    fig.savefig('sec4.2_scatter.pdf')


def plot_likelihood():
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    kwargs = [dict(colors='black', linewidths=2, linestyles='solid'),
              dict(colors='gray', linewidths=1, linestyles='solid')]
    
    for axi, vmax in zip(ax, [1000, 200]):
        for kwds, cutoff, in zip(kwargs, ['-6', '-7']):
            osl = OffsetLikelihood.from_file(cutoff, vmax)

            osl.plot_posterior(np.linspace(-0.001, 0.05, 50),
                               np.linspace(-0.001, 0.05, 49),
                               axi, **kwds)

        if vmax > 400:
            axi.set_title(r'${\rm All\, Galaxies}$')
        else:
            axi.set_title(r'${\rm v\, <\, %i km/s}$' % vmax)

    ax[1].set_ylabel('')
    ax[0].plot([0], [0], '-', c='gray', lw=1,
               label=r'$f_{R0} = 2 \times 10^{-7}$')
    ax[0].plot([0], [0], '-', c='black', lw=2,
               label='$f_{R0} = 10^{-6}$')
    ax[0].legend(loc=2, fontsize=12)
    fig.savefig('sec4.2_offset.pdf')
        
        

if __name__ == '__main__':
    for which in ['-6', '-6.5', '-7']:
        res = get_data(vlim=1000, which=which)
        screened = res[-1]
        print which, np.sum(screened), np.sum(~screened)
    plot_data()
    plot_likelihood()
    plt.show()
