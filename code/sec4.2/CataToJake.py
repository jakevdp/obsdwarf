import os
import numpy as np


if __name__=='__main__':

    # Set limits to generate sample
    vlim = 200 # CHANGE
    fR0 = 10**(-6.5) #CHANGE

    f = np.genfromtxt('PersictoJake_section42.dat', skip_header=1)
    vmax = f[:,0]
    offset = f[:,1][vmax < vlim]
    optical = f[:,2][vmax < vlim]
    distance = f[:,3][vmax < vlim]
    phi1 = f[:,4][vmax < vlim]
    phi2 = f[:,5][vmax < vlim]
    phi3 = f[:,6][vmax < vlim]
    vmax = f[:,0][vmax < vlim]

    off_opt = offset / optical
    error = 0.5 / optical
    screened = np.zeros(off_opt.shape[0])

    if fR0 == 10**(-7):
        unscr = ((vmax / 100.)**2. < fR0/2e-7) & (phi1 < fR0)

    elif fR0 == 10**(-6.5):
        unscr = ((vmax / 100.)**2. <= fR0/2e-7) & (phi2 <= fR0)

    elif fR0 == 10**(-6):
        unscr = ((vmax / 100.)**2. < fR0/2e-7) & (phi3 < fR0)
    else:
        raise ValueError, 'fR0 does not correspond to any of the known scale' 

    scr = np.logical_not(unscr)
    
    screened[scr] = 1
    print unscr[unscr].shape

    np.savetxt('Un_Screened_fR0_-6.5_vmax_200.cat', np.transpose((off_opt, error, screened)), fmt='%2.5f %2.5f %d') #CHANGE FILE NAME


