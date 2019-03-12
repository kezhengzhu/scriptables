#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from dervar import *
import defconst as cst

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

def checkwarn(cond, message):
    if not (cond):
        print("#"*32)
        print(message)
        print("#"*32)

# _s for star, for reduced units

def pre_mie(rep, att):
    return rep/(rep-att) * pow(rep/att, att/(rep-att))

def mie_s(r_s, rep, att=6):
    '''
    Take reduced dist ndarray or float as input, return mie potential value(s)
    '''
    checkerr(isinstance(r_s, np.ndarray) or isinstance(r_s, float), "Use ndarray or float for mie_s")
    checkerr((isinstance(r_s, np.ndarray) and r_s.ndim == 1) or isinstance(r_s, float), "ndarray in wrong dimensions!")
    c_mie = rep/(rep-att) * pow(rep/att, att/(rep-att))
    u_s = c_mie * (pow(r_s,-rep) - pow(r_s,-att))

    return u_s

def hsdiam(T_s, rep, att=6, x_inf=0):
    '''
    Calc hard sphere diameter based on reduced temp and x_inf (reduced hard rep dist)
    '''
    checkerr(x_inf >= 0 and x_inf <= 1, "x_inf value error")
    xgl = np.array([0.97390652852, 0.86506336669, 0.67940956830, 0.43339539413, 0.14887433898])
    wgl = np.array([0.06667134431, 0.14945134915, 0.21908636252, 0.26926671931, 0.29552422471])
    x1i = 0.5 * (1 + x_inf + xgl *(1-x_inf))
    x2i = 0.5 * (1 + x_inf - xgl *(1-x_inf))
    # xsum = sum(wgl * ((x1i**2 * np.exp(-mie_s(x1i, rep, att) / T_s)) + (x2i**2 * np.exp(-mie_s(x2i, rep, att) / T_s))) )
    # Loop to cater to Var 
    xsum = 0.
    for i in range(len(x1i)):
        xsum = xsum + wgl[i] * ((x1i[i]**2 * exp(-mie_s(x1i[i], rep, att) / T_s)) + (x2i[i]**2 * exp(-mie_s(x2i[i], rep, att) / T_s)))

    dcube_s = 1 - (3 * (1-x_inf) / 2 * xsum)
    dhs_s = pow(dcube_s, 1/3)

    # r_s = np.flip(np.linspace(1.,0.,100, endpoint=False))
    # uu = mie_s(r_s,rep,att)
    # expu = np.exp(-uu/T_s)
    # d3 = np.trapz(expu, r_s)
    # dhs_s = pow(1 - np.trapz(expu, r_s),1/3)
    return dhs_s

def inv_angle(angle):
    '''
    Takes tuple angle input and negative it
    '''
    return tuple( [ -i for i in angle ] )

def thdebroglie(mass, temp):
    '''
    Takes in component mass (au) and temperature (K) and return thermal de broglie wavelength
    '''
    Lambda_sq = cst.h**2 / (2 * pi * mass * cst.au * cst.k * temp)
    return sqrt(Lambda_sq)

def intI(x0kl, lam):
    num = 1 - pow(x0kl, 3-lam)
    den = lam - 3

    return num / den

def intJ(x0kl, lam):
    num = 1 - pow(x0kl, 4-lam) * (lam - 3) + pow(x0kl, 3-lam) * (lam - 4)
    den = (lam - 3) * (lam - 4)

    return num / den

def bkl_sum(xi_x, x0kl, lam): # doesn't include prefactor which has segden, dkl and ekl
    t1 = (1 - xi_x/2) / pow(1 - xi_x, 3) * intI(x0kl, lam)
    t2 = 9 * xi_x * (1+xi_x) / (2 * pow(1 - xi_x, 3)) * intJ(x0kl, lam)

    return t1 - t2

def xi_x_eff(xi_x, lam):
    cm = cst.cm
    lam_vec = np.array([1., 1/lam, 1/pow(lam,2), 1/pow(lam,3)])
    c = np.matmul(cm, lam_vec)
    if isinstance(xi_x, Var):
        result = 0.
        for i in range(len(c)):
            result = result + c[i] * pow(xi_x,i+1)
        return result

    xis = np.array([xi_x, pow(xi_x,2), pow(xi_x,3), pow(xi_x,4)])

    result = sum(c * xis)
    return result

def der_xi_x_eff(xi_x, der_xi_x, lam):
    cm = cst.cm
    lam_vec = np.array([1., 1/lam, 1/pow(lam,2), 1/pow(lam,3)])
    c = np.matmul(cm, lam_vec)

    if isinstance(xi_x, Var):
        result = 0.
        for i in range(len(c)):
            result = result + (i+1) * c[i] * pow(xi_x,i) * der_xi_x
        return result
        
    xis = np.array([der_xi_x, 2*xi_x*der_xi_x, 3*pow(xi_x,2)*der_xi_x, 4*pow(xi_x,3)*der_xi_x])

    result = sum(c * xis)
    return result

def f_m(alpha, m):
    phi = cst.phi
    num = 0.
    for n in range(0,4):
        num += phi[m-1, n] * pow(alpha, n)
        # print('phi_{:1d}_{:1d}: {:9.6f}'.format(m,n,phi[m-1,n]), end=' ')
    den = 1.
    for n in range(4,7):
        den += phi[m-1, n] * pow(alpha, n-3)
    #     print('phi_{:1d}_{:1d}: {:9.6f}'.format(m,n,phi[m-1,n]), end=' ')
    # print()
    return num/den


def gdhs(x0ii, xi_x):
    k0 = -log(1 - xi_x) + (42 * xi_x - 39 * pow(xi_x, 2) + 9 * pow(xi_x, 3) - 2 * pow(xi_x, 4)) / (6 * pow(1-xi_x,3))
    k1 = (pow(xi_x,4) + 6 * pow(xi_x, 2) - 12*xi_x) / (2*pow(1-xi_x,3))
    k2 = -3 * pow(xi_x,2) / (8*pow(1-xi_x, 2))
    k3 = (-pow(xi_x,4) + 3*pow(xi_x, 2) + 3*xi_x) / (6*pow(1-xi_x, 3))
    result = exp(k0 + k1 * x0ii + k2 * pow(x0ii, 2) + k3 * pow(x0ii, 3))
    return result

def m3mol_to_nm(m3molv, molecules=1000):
    '''
    Default assumes 1000 molecules
    '''
    return molecules * m3molv / (cst.Na * pow(cst.nmtom,3))
