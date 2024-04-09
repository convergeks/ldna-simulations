import datetime as dt
import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import math
# local imports
import utils.util_routine_calcs as calc


def get_XY1_XY2_alpha(ref_coords, coord_diag1, coord_diag2, ang):
    ref_lonlat = ref_coords
    LL_lonlat = coord_diag1
    X1, Y1, bm1 = calc.util_convert_htwidth_GPS2Cartcoord(ref_lonlat, LL_lonlat)
    UR_lonlat = coord_diag2
    X2, Y2, _ = calc.util_convert_htwidth_GPS2Cartcoord(ref_lonlat, UR_lonlat)
    # print(LL_lonlat, X1, Y1)
    # print(UR_lonlat, X2, Y2)
    # summarize
    D = np.linalg.norm([X2-X1, Y2-Y1])
    # Dx, Dy = (X2-X1), (Y2-Y1)
    Dvec = D #, Dx, Dy
    # print(X2-X1, Y2-Y1)
    Theta = np.arctan2(X2-X1, Y2-Y1)
    Alpha = ang
    # print(D, Theta*180/np.pi, Alpha*180/np.pi)
    return (X1, Y1), (X2, Y2), (Dvec, Theta, Alpha), bm1

def get_midpoint(C1, C2):
    C1_1, C1_2 = C1[0], C1[1]
    C2_1, C2_2 = C2[0], C2[1]
    mid_c1 = C1_1 + (C2_1 - C1_1)/2
    mid_c2 = C1_2 + (C2_2 - C1_2)/2
    mid_coord = mid_c1, mid_c2
    return mid_coord

# --- ellipse realted functions ---
def define_ellipse(XY1, XY2, DThAlf):
    X0Y0 = get_midpoint(XY1, XY2)
    D, Th, Alpha = DThAlf
    X0, Y0 = X0Y0
    # ellipse semi-major/minor axes
    a1 = np.abs(D*np.cos(Th))/2
    a2 = np.abs(D*np.sin(Th))/2
    # print(a1, a2)
    if a1>a2:
        a, b = a1, a2 
        x, y = get_coords(a, b, X0, Y0, Alpha)
    else :
        a, b = a2, a1
        y, x = get_coords(a, b, X0, Y0, Alpha)
    # plot
    return x, y

def ellipse(a, b):
    e = (1-b**2/a**2)**0.5
    xy = []
    for theta in np.linspace(0, 2*np.pi, 100):
        r = b/(1-(e*np.cos(theta))**2)**0.5
        xy.append((r*np.cos(theta), r*np.sin(theta)))
    return xy

def global_coord_enable(X0, Y0):
    Theta = np.arctan2(X0, Y0)
    return Theta

def get_coords(a, b, X0, Y0, Alpha):
    ThetaAlpha = global_coord_enable(X0, Y0) + Alpha
    xy = ellipse(a, b)
    x, y = list(zip(*xy))
    # gloabl coords
    X = [X0+xx*np.cos(ThetaAlpha)+y[i]*np.sin(ThetaAlpha) for i, xx in enumerate(x)]
    Y = [Y0-xx*np.sin(ThetaAlpha)+y[i]*np.cos(ThetaAlpha) for i, xx in enumerate(x)]
    return X, Y
