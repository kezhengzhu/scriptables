#!/usr/bin/env python
import numpy as np 
import pandas as pd 
from math import pi,tanh
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, least_squares

from dervar import *

import defconst as cst
import methods as mt 
from plotxvg import *

class System(object):
    def __init__(self, temperature=293, volume=1000, **kwargs):
        for comp in kwargs:
            mt.checkerr(isinstance(kwargs[comp], Component), "Use Component object to specify groups used, with format CompName = Component obj")
        self.comps = kwargs
        self.moles = {}
        self.molfrac = {}
        for comp in self.comps:
            self.moles[comp] = 0
            self.molfrac[comp] = 0
        self.temp = temperature     # K
        self.volume = volume     # nm^3
        self.pressure = 1   # bar (temporarily useless)
        self.sdt = 1

    def __repr__(self):
        return '<System obj with {:2d} components>'.format(self.ncomp())

    def ncomp(self):
        return len(self.comps)

    def add_comp(self, **kwargs):
        for comp in kwargs:
            mt.checkerr(isinstance(kwargs[comp], Component), "Use Component object to specify groups used, with format CompName = Component obj")

        for comp in kwargs:
            mt.checkwarn(kwargs[comp] in self.comps.values(), "Component object already exist, {} not added".format(comp))
            mt.checkwarn(comp in self.comps, "Component name taken, {} not added".format(comp))
            if (comp not in self.comps) and (kwargs[comp] not in self.comps.values()):
                self.comps[comp] = kwargs[comp]
                self.moles[comp] = 0
                self.molfrac[comp] = 0

    def list_comps(self):
        print(list(self.comps.keys()))
        return list(self.comps.keys())

    def add_moles(self, **kwargs):
        for comp in kwargs:
            mt.checkerr(comp in self.comps, "Component name {} is not in system".format(comp))
            mt.checkerr(isinstance(kwargs[comp], int), "Int values for components only")

        for comp in kwargs:
            self.moles[comp] = self.moles[comp] + kwargs[comp]

        for comp in self.molfrac:
            self.molfrac[comp] = self.moles[comp]/self.__moltol()


    def quick_set(self, *args):
        complist = []
        molelist = []
        for arg in args:
            mt.checkerr(isinstance(arg, tuple) or isinstance(arg, list), "Use iterable tuple or list for quick_set")
            mt.checkerr(isinstance(arg[0], Component), "First element of iterable must be of Component type")
            mt.checkerr(isinstance(arg[1], int), "Second element of iterable must be of int type")
            complist.append(arg[0])
            molelist.append(arg[1])

        num = []
        count = 1
        for i in range(len(complist)):
            name = 'COMP{:03d}'.format(count)
            while (name in self.comps) or (name in num):
                count = count+1
                name = 'COMP{:03d}'.format(count)
            num.append(name)

        faults = 0
        for i in range(len(complist)):
            if (complist[i] not in self.comps.values()):
                self.comps[num[i-faults]] = complist[i]
                self.moles[num[i-faults]] = molelist[i]
                self.molfrac[num[i-faults]] = 0
            else:
                print("Component already in system, molecules not added")
                faults = faults + 1

        for comp in self.molfrac:
            self.molfrac[comp] = self.moles[comp]/self.__moltol()

        return self

    def getgtypes(self):
        gtypes = []
        for comp in self.comps:
            for g in self.comps[comp].gtypes:
                if g not in gtypes:
                    gtypes.append(g)
        return gtypes

    def helmholtz(self):
        mole_tol = self.__moltol()
        a = self.a_ideal() + self.a_mono() + self.a_chain() + self.a_assoc()
        A = a * mole_tol * cst.k * self.temp
        return A # this A value is for the System. Not per mol or per molecule!

    def get_state(self, volume, temperature=None):
        '''
        Using SAFT gamma mie and automatic differentiation, return all the properties
        that could be derived from the EoS: (A, P, S, G, H)
        '''
        Var.set_order(2)
        self.volume = Var(mt.m3mol_to_nm(volume, molecules=self.__moltol())) # nm3
        if isinstance(temperature, float):
            self.temp = Var(temperature)
        else:
            self.temp = Var(self.temp)

        A = self.helmholtz() # J total for current system, Var
        molcs = self.__moltol()
        d1 = derivative(A, self.volume, self.temp, order=1, getvar=False)

        P = -d1[0] / pow(cst.nmtom,3) # Pa, float
        S = -d1[1] / molcs # J/K per molecule, float
        G = (A + P * self.volume*pow(cst.nmtom,3)) / molcs # (J + Pa*m3) for each molecule, Var
        H = G + S * self.temp # J + J/K * K (it is already in per molecule here), Var
        A = A.value/molcs
        # Reset system back to floats
        self.volume = self.volume.value
        self.temp = self.temp.value
        return (A, P, S, G.value, H.value)

    def __eqpg(self, x, print_, scaler):
        '''
        Takes in ndarray to find Pressure1 - Pressure2
        '''
        P = []
        G = []
        v_ = [x[0]*scaler[0], x[1]*scaler[1]]
        for i in v_:
            self.volume = Var(mt.m3mol_to_nm(i, molecules=self.__moltol()))
            A = self.helmholtz()
            Pi = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)
            P.append(Pi)
            G.append((A.value + Pi * self.volume.value*pow(cst.nmtom,3))*cst.Na)

        if print_: print(f'Current: v\'s = {v_[0]:7.3e}, {v_[1]:7.3e}; dP = {(P[0]-P[1]):7.3e}, dG = {(G[0]-G[1]):7.3e}')
        return np.array([(P[0]-P[1]), (G[0]-G[1])])

    def vapour_pressure(self, temperature=None, initial_guess=(1e-4,.1), get_volume=False, get_gibbs=False, print_results=True, solver=least_squares, solver_kwargs={'bounds': ((0.1,1e-2),(1e2,1e2))}, print_progress=False):
        if isinstance(temperature, float) or isinstance(temperature, int):
            self.temp = temperature
        scaler = initial_guess
        x0 = np.array([1.,1.])
        if not isinstance(solver_kwargs, dict): solver_kwargs = {}
        vle = solver(self.__eqpg, x0, args=(print_progress, scaler), **solver_kwargs)
        if print_progress: print()
        if print_results: 
            print(f'Scaled wrt initial guess of ({scaler[0]:7.3e},{scaler[1]:7.3e})')
            print(vle)
        vlv = (vle.x[0]*scaler[0], vle.x[1]*scaler[1])
        v = vlv[1]
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.__moltol()))

        A = self.helmholtz()
        P = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)
        G = (A.value + P * self.volume.value*pow(cst.nmtom,3)) / self.__moltol()
        self.volume = mt.m3mol_to_nm(v, molecules=self.__moltol())
        result = (P,)
        if get_volume: result += (vlv,)
        if get_gibbs: result += (G,)
        return result

    def p_v_isotherm(self, volume, temperature=None, gibbs=False):
        '''
        Get pressure profile from volume inputs. Volume in m3 per mol
        '''
        if isinstance(temperature, float) or isinstance(temperature, int):
            self.temp = temperature

        mt.checkerr(isinstance(volume,float) or isinstance(volume, np.ndarray), "Use floats or numpy array for volume")
        volume = self.__moltol() * volume / (cst.Na * pow(cst.nmtom,3))
        Var.set_order(1)
        # Case single volume value
        if isinstance(volume, float):
            self.volume = Var(volume)
            A = self.helmholtz()
            result = -derivative(A,self.volume,order=1) / pow(cst.nmtom,3)

            # Reset system back to float
            self.volume = volume
            return result

        # Case numpy array
        old_v = self.volume
        vlen = np.size(volume)
        P = np.zeros(vlen)
        if gibbs: G = np.zeros(vlen)
        print('='*5, f'Pv Isotherm data from {vlen:5d} points', '='*5)
        tenp = vlen // 10
        for i in range(vlen):
            self.volume = Var(volume[i])
            A = self.helmholtz()
            P[i] = -derivative(A,self.volume) / pow(cst.nmtom,3)

            if gibbs: G[i] = (A.value + P[i] * self.volume.value*pow(cst.nmtom,3))/self.__moltol()
            if (i+1) % tenp == 0:
                print(f'Progress at {(i+1)//tenp * 10:3d}%')
        # Reset system back to float
        self.volume = old_v
        return (P, G) if gibbs else P

    def p_rho_isotherm(self, nden, temperature=None):
        '''
        nden in mol/m3
        '''
        if isinstance(temperature, float):
            self.temp = temperature
        mt.checkerr(isinstance(nden,float), "Use floats for rho")
        volume = self.__moltol()/(cst.Na*nden * pow(cst.nmtom,3))
        self.volume = Var(volume)
        A = self.helmholtz()
        result = -derivative(A,self.volume) / pow(cst.nmtom,3)

        # Reset system back to float
        self.volume = volume
        return result

    def __getp(self, x, targetP):
        '''
        Takes in single value to get P
        '''
        x = x[0]
        self.volume = Var(mt.m3mol_to_nm(x, molecules=self.__moltol()))
        A = self.helmholtz()
        P = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)

        return P - targetP

    def single_phase_v(self, P, T=None, print_results=True, get_density=False):
        if isinstance(T, float):
            self.temp = T

        if isinstance(P, float) or isinstance(P, int):
            Pv, vle = self.vapour_pressure(temperature=T, get_volume=True)
            v_init = vle[0] if P > Pv else vle[1]
            getv = least_squares(self.__getp, v_init, kwargs={"targetP": P})

            if print_results: print(getv)

            return getv.x[0] if get_density == False else (getv.x[0], 1/getv.x[0])

        mt.checkerr(isinstance(P, np.ndarray), "Use float/int or ndarray for pressure inputs")
        varr = np.zeros(np.size(P))
        for i in range(len(P)):
            Pv, vle = self.vapour_pressure(temperature=T, get_volume=True)
            v_init = vle[0] if P[i] > Pv else vle[1]
            getv = least_squares(self.__getp, v_init, kwargs={"targetP": P[i]})

            varr[i] = getv.x[0]

        return varr if get_density == False else (varr, 1./varr)

    def critical_point(self, initial_t=300, initial_v=3e-4, v_nd=None, get_volume=False, get_density=False, print_results=True, solver=least_squares, solver_kwargs={'bounds': (0.1,10)}, xtol=1e-8, print_progress=False):
        tscale = initial_t
        x0 = np.array([1.])
        if not isinstance(solver_kwargs, dict): solver_kwargs = {}
        if not isinstance(v_nd, np.ndarray): v_nd = np.logspace(-4,-2, 50)
        Var.set_order(2)
        crit_pt = solver(self.__crit_t, x0, xtol=xtol, args=(v_nd, tscale, print_progress),**solver_kwargs)

        if print_progress: print()
        if print_results: print(crit_pt)
        T = crit_pt.x[0]*tscale

        crit_v = solver(self.__crit_v, np.array([initial_v]), xtol=xtol, args=(T,), bounds=(min(v_nd), max(v_nd)))
        v = crit_v.x[0]
        
        Var.set_order(1)
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.__moltol()))
        self.temp = T

        A = self.helmholtz()
        P = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)

        self.volume = mt.m3mol_to_nm(v, molecules=self.__moltol())
        results = (P, T)
        if get_volume: results += (v,)
        if get_density: results += (1/v,)

        return results

    def __crit_v(self, v, T):
        self.temp = T
        v = v[0]
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.__moltol()))
        A = self.helmholtz()
        _, dpdv = derivative(A, self.volume, order=2)
        dpdv = -dpdv / (pow(cst.nmtom,6) * cst.Na) / (cst.R * T)

        return dpdv

    def __crit_t(self, x, v_nd, scale, print_):
        T = x[0] * scale
        dp = np.zeros(np.size(v_nd))
        for i in range(len(v_nd)):
            v = v_nd[i]
            self.volume = Var(mt.m3mol_to_nm(v, molecules=self.__moltol()))
            self.temp = T
            A = self.helmholtz()
            _, dpdv = derivative(A, self.volume, order=2)
            dp[i] = -dpdv / (pow(cst.nmtom,6) * cst.Na) / (cst.R * T)
        if print_: 
            print(f'Current: T = {T:7.3f}: max dP/dV = {max(dp):7.3e}', end='\r')
        return max(dp)

    def __critpt(self, x, print_, scale):
        '''
        Takes in (v, T) to solve for dP/dV = 0 and d2P/dV2 = 0
        '''
        v_ = scale[0]
        T_ = scale[1]
        v = x[0]*v_
        T = x[1]*T_
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.__moltol()))
        self.temp = T

        A = self.helmholtz()
        dP = -np.array(derivative(A, self.volume, order=3)) / pow(cst.nmtom,3)
        dP[1] = dP[1] / (pow(cst.nmtom,3) * cst.Na) / (cst.R * T)
        dP[2] = dP[2] / (pow(cst.nmtom,6) * pow(cst.Na,2)) / (cst.R * T)
        dP[0] = dP[0] / (cst.R * T)
        if print_:
            print(f'Current: v = {v:7.3e}, T = {T:7.3f}: P, dP/dV, d2P/dV2 = {dP[0]:7.3e},{dP[1]:7.3e},{dP[2]:7.3e}', end='\r')
        dP[0] = min(dP[0], 0)
        return dP

    def get_critical_point(self, initial_guess=(3e-4, 300), get_volume=False, get_density=False, print_results=True, solver=least_squares, solver_kwargs={'bounds': ((1e-1,1e-1), (10,10))}, xtol=1e-8, print_progress=False):
        scaler = initial_guess
        x0 = np.array([1.,1.])
        if not isinstance(solver_kwargs, dict): solver_kwargs = {}
        Var.set_order(3)
        crit_pt = solver(self.__critpt, x0, xtol=xtol, args=(print_progress, scaler),**solver_kwargs)

        if print_progress: print()
        if print_results: print(crit_pt)
        vT = crit_pt.x
        v = vT[0]*scaler[0]
        T = vT[1]*scaler[1]

        Var.set_order(1)
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.__moltol()))
        self.temp = T

        A = self.helmholtz()
        P = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)

        self.volume = mt.m3mol_to_nm(v, molecules=self.__moltol())
        results = (P, T)
        if get_volume: results += (v,)
        if get_density: results += (1/v,)

        return results

    def a_ideal(self):
        result = 0.
        mole_tol = sum(self.moles.values())
        for comp in self.comps:
            debrogv = pow(self.comps[comp].thdebroglie(self.temp),3)
            molfrac = self.molfrac[comp]
            nden = molfrac * self.__nden()
            result = result + molfrac * log(nden * debrogv)
        return result - 1

    def a_mono(self):
        return self.__a_hs() + self.__a_1() + self.__a_2() + self.__a_3()

    def a_chain(self):
        result = 0.
        mole_tol = sum(self.moles.values())
        for comp in self.comps:
            molfrac = self.molfrac[comp]
            c_cont = 0.
            gmieii = self.__gmieii(self.comps[comp].get_gtypeii())
            for g in self.comps[comp].gtypes:
                numg_ki = self.comps[comp].gtypes[g]
                c_cont += (numg_ki * g.vk * g.sk - 1) 
            result += molfrac * c_cont * log(gmieii)
        result = -result
        return result

    def a_assoc(self):
        return 0

    ### for all
    def __cgshapesum(self):
        '''
        Returns sum of (mol frac * sum of (no. groups * shape stuff for all groups ))
        '''
        result = 0.
        mole_tol = sum(self.moles.values())
        for comp in self.comps:
            molfrac = self.molfrac[comp]
            c_cont = 0
            for gtype in self.comps[comp].gtypes:
                numg_ki = self.comps[comp].gtypes[gtype]
                c_cont = c_cont + numg_ki * gtype.vk * gtype.sk
            result = result + molfrac * c_cont
        return result

    def __gshapesum(self, thistype):
        '''
        Returns sum of (mol frac * no. of groups in c * shape stuff for one group)
        i.e. the sum across the group for all components
        '''
        result = 0.
        mole_tol = sum(self.moles.values())
        for comp in self.comps:
            molfrac = self.molfrac[comp]
            c_cont = 0
            for gtype in self.comps[comp].gtypes:
                if gtype == thistype:
                    numg_ki = self.comps[comp].gtypes[gtype]
                    c_cont = c_cont + numg_ki * gtype.vk * gtype.sk
            result = result + molfrac * c_cont
        return result

    def __xsk(self, thistype):
        return self.__gshapesum(thistype) / self.__cgshapesum

    def __moltol(self):
        mole_tol = sum(self.moles.values())
        return mole_tol

    def __nden(self):
        return self.__moltol() / (self.volume * pow(cst.nmtom, 3))

    def __segden(self):
        return self.__nden() * self.__cgshapesum() * self.sdt
    
    def __xi_x(self):
        result = 0.
        gtypes = self.getgtypes()
        cgss = self.__cgshapesum()
        for g1 in gtypes:
            xsk = self.__gshapesum(g1)/cgss
            for g2 in gtypes:
                xsl = self.__gshapesum(g2)/cgss
                dkl = (g1+g2).hsdiam(self.temp) # nm
                result += xsk * xsl * pow(dkl * cst.nmtom, 3)
        result = result * pi * self.__segden() / 6 
        return result

    def __xi_sx(self):
        result = 0.
        gtypes = self.getgtypes()
        cgss = self.__cgshapesum()
        for g1 in gtypes:
            xsk = self.__gshapesum(g1)/cgss
            for g2 in gtypes:
                xsl = self.__gshapesum(g2)/cgss
                skl = (g1+g2).sigma # nm
                result += xsk * xsl * pow(skl * cst.nmtom, 3)
        result = result * pi * self.__segden() / 6
        return result

    def __bkl(self, gcomb, lam):
        '''
        Use 2 groups as input
        set rep = True for lam_r, rep = False for lam_a
        2 * pi * segden * dkl^3 epsikl * bkl_sum(lambda_r or a, xi_x)
        '''
        segden = self.__segden()
        dkl = gcomb.hsdiam(self.temp)
        dkl_3 = pow(dkl * cst.nmtom, 3)
        epsikl = gcomb.epsilon * cst.k

        x0kl = gcomb.x0kl(self.temp)
        xix = self.__xi_x()

        bsum = mt.bkl_sum(xix, x0kl, lam)
        result = 2*pi*segden*dkl_3*epsikl*bsum

        return result

    def __as1kl(self, gcomb, lam):
        '''
        - 2 * pi * segden * (epsikl * dkl^3/ (lam - 3)) * (1 - xi_x_eff/2)/(1-xi_x_eff)^3
        '''
        segden = self.__segden()
        dkl = gcomb.hsdiam(self.temp)
        dkl_3 = pow(dkl * cst.nmtom, 3)
        epsikl = gcomb.epsilon * cst.k

        xix = self.__xi_x()

        xixeff = mt.xi_x_eff(xix, lam)

        num = 1 - xixeff/2
        den = pow(1 - xixeff, 3)

        result = -2 * pi * segden * (epsikl * dkl_3 / (lam - 3)) * (num/den)

        return result

    ### for mono
    ##### for hs term
    def __a_hs(self):

        xi3 = self.__xi_m(3) # dimless
        # print('With pure imp:',(4*xi3 - 3*pow(xi3,2))/pow(1-xi3,2))
        # print('This ans but with segden:', 6 * self.__a_hs_xiterm() / (pi* self.__segden()))
        return 6 * self.__a_hs_xiterm() / (pi * self.__nden())
    
    def __a_hs_xiterm(self):
        xi1 = self.__xi_m(1) # nm-2
        xi2 = self.__xi_m(2) # nm-1
        xi3 = self.__xi_m(3) # dimless
        xi0 = self.__xi_m(0) # nm-3

        if len(self.getgtypes()) == 1:
            t1 = 0
        else:
            t1 = (pow(xi2, 3) / pow(xi3, 2) - xi0) * log(1-xi3) # t1 is HUGE, and because of numerical discrepancy, even though
                                                                # if there is only 1 component, t1 approx 0,
                                                                # but because each term is ~ 10^27 so discrepancies could be millions even
        t2 = 3 * xi1 * xi2 / (1 - xi3)
        t3 = pow(xi2, 3) / (xi3 * pow((1-xi3),2))
        # print('t2 + t3 gives', t2+t3)
        # print('With pure imp:',(4*xi3 - 3*pow(xi3,2))/pow(1-xi3,2))

        return t1 + t2 + t3 # nm-3

    def __xi_m(self, m):
        nden = self.__nden()
        fac = pi * nden / 6 # nm-3
        gtypes = self.getgtypes()
        gsum = 0.
        g2 = 0.
        for g in gtypes:
            xskxcg = self.__gshapesum(g)
            # xsk = self.__gshapesum(g)/self.__cgshapesum()
            dkk = g.hsdiam(self.temp) # nm
            gsum += (xskxcg * pow(dkk*cst.nmtom, m))
        return fac * gsum

    ##### for A1 term
    def __a_1(self):
        '''
        1/kT * cgss * sum of sum of _g1/cgss _g2/cgss * a1kl
        '''
        a1sum = 0.
        gtypes = self.getgtypes()
        cgss = self.__cgshapesum()
        for g1 in gtypes:
            xsk = self.__gshapesum(g1)/cgss
            for g2 in gtypes:
                xsl = self.__gshapesum(g2)/cgss
                a1kl = self.__a_1kl(g1 + g2)
                a1sum += xsk * xsl * a1kl 

        result = 1 / (cst.k * self.temp) * cgss * a1sum 
        return result

    def __a_1kl(self, gcomb):
        '''
        Ckl * [ x0kl^att(as1kl(att) + bkl(att)) - x0kl^rep(as1kl(rep) + bkl(rep)) ]
        '''
        x0kl = gcomb.x0kl(self.temp)
        rep = gcomb.rep
        att = gcomb.att
        ckl = gcomb.premie()

        as1kl_a = self.__as1kl(gcomb, att)
        bkl_a = self.__bkl(gcomb, att)

        as1kl_r = self.__as1kl(gcomb, rep)
        bkl_r = self.__bkl(gcomb, rep)

        t1 = pow(x0kl, att) * (as1kl_a + bkl_a)
        t2 = pow(x0kl, rep) * (as1kl_r + bkl_r)
        result = ckl * (t1 - t2)

        return result
    
    ##### for A2 term
    def __a_2(self):
        '''
        (1/kT)**2 * cgss * sum of sum of _g1/cgss _g2/cgss * a2kl
        '''
        a2sum = 0.
        gtypes = self.getgtypes()
        cgss = self.__cgshapesum()
        for g1 in gtypes:
            xsk = self.__gshapesum(g1)/cgss
            for g2 in gtypes:
                xsl = self.__gshapesum(g2)/cgss
                a2kl = self.__a_2kl(g1 + g2)
                a2sum += xsk * xsl * a2kl

        result = 1 / (pow(cst.k * self.temp,2)) * cgss * a2sum 
        return result

    def __a_2kl(self, gcomb):
        '''
        1/2 * khs * (1-corrf) * epsikl * ckl^2 *
        { x0kl^(2att) * (as1kl(2*att) + bkl(2*att))
          - 2*x0kl^(att+rep) * (as1kl(att+rep) + bkl(att+rep))
          + x0kl^(2rep) * (as1kl(2rep) + bkl(2rep)) }
        '''
        khs = self.__khs()
        corrf = self.__corrf(gcomb)
        epsikl = gcomb.epsilon * cst.k
        ckl = gcomb.premie()
        x0kl = gcomb.x0kl(self.temp)
        rep = gcomb.rep
        att = gcomb.att

        t1 = pow(x0kl, 2*att) * (self.__as1kl(gcomb, 2*att) + self.__bkl(gcomb, 2*att))
        t2 = 2*pow(x0kl, att+rep) * (self.__as1kl(gcomb, att+rep) + self.__bkl(gcomb, att+rep))
        t3 = pow(x0kl, 2*rep) * (self.__as1kl(gcomb, 2*rep) + self.__bkl(gcomb, 2*rep))

        result = 0.5 * khs * (1+corrf) * epsikl * pow(ckl, 2) * (t1 - t2 + t3)

        return result

    def __khs(self):
        xix = self.__xi_x()
        num = pow(1-xix, 4)
        den = 1 + 4*xix + 4*pow(xix,2) - 4*pow(xix,3) + pow(xix,4)
        return num/den

    def __corrf(self, gcomb):
        xisx = self.__xi_sx()
        alkl = self.__alphakl(gcomb)

        t1 = mt.f_m(alkl, 1) * xisx
        t2 = mt.f_m(alkl, 2) * pow(xisx, 5)
        t3 = mt.f_m(alkl, 3) * pow(xisx, 8)
        result = t1 + t2 + t3
        return result

    def __alphakl(self, gcomb):
        t1 = 1 / (gcomb.att - 3)
        t2 = 1 / (gcomb.rep - 3)
        result = gcomb.premie() * (t1 - t2)

        return result

    ##### for A3 term
    def __a_3(self):
        '''
        (1/kT)**3 * cgss * sum of sum of _g1/cgss _g2/cgss * a3kl
        '''
        a3sum = 0.
        gtypes = self.getgtypes()
        cgss = self.__cgshapesum()
        for g1 in gtypes:
            xsk = self.__gshapesum(g1)/cgss
            for g2 in gtypes:
                xsl = self.__gshapesum(g2)/cgss
                a3kl = self.__a_3kl(g1 + g2)
                a3sum += xsk * xsl * a3kl

        result = 1 / (pow(cst.k * self.temp,3)) * cgss * a3sum 
        return result

    def __a_3kl(self, gcomb):
        xisx = self.__xi_sx()
        alkl = self.__alphakl(gcomb)
        epsikl = gcomb.epsilon * cst.k

        preexp = - pow(epsikl,3) * mt.f_m(alkl, 4) * xisx
        expt1 = mt.f_m(alkl, 5) * xisx 
        expt2 = mt.f_m(alkl, 6) * pow(xisx,2)

        result = preexp * exp(expt1 + expt2)
        return result

    ### for chain
    def __der_xi_x(self):
        result = 0.
        gtypes = self.getgtypes()
        cgss = self.__cgshapesum()
        for g1 in gtypes:
            xsk = self.__gshapesum(g1)/cgss
            for g2 in gtypes:
                xsl = self.__gshapesum(g2)/cgss
                dkl = (g1+g2).hsdiam(self.temp) # nm
                result += xsk * xsl * pow(dkl * cst.nmtom, 3)
        result = result * pi / 6 
        return result

    def __der_as1kl(self, gcomp, lam):
        '''
        -2 * pi * (ep*d^3/(lam-3)) * (1/(1-xieff)^3) * (1 - xieff/2 + segden * derxieff * (3*(1-xieff/2)/(1-xieff) - 1/2))
        '''
        segden = self.__segden()
        dkl = gcomp.hsdiam(self.temp)
        d_3 = pow(dkl * cst.nmtom, 3)
        epsi = gcomp.epsilon * cst.k

        derxix = self.__der_xi_x()
        xix = self.__xi_x()

        derxixeff = mt.der_xi_x_eff(xix, derxix, lam)
        xixeff = mt.xi_x_eff(xix, lam)

        fac = 2 * pi * epsi * d_3 / (lam-3) / pow(1-xixeff,3)
        t1 = 1 - xixeff / 2
        t2 = segden * derxixeff
        t3 = 3 * (1-xixeff/2) / (1-xixeff)

        result = - fac * (t1 + t2 * (t3 - 1/2))

        return result

    def __der_bkl(self, gcomp, lam):
        segden = self.__segden()
        dkl = gcomp.hsdiam(self.temp)
        d_3 = pow(dkl * cst.nmtom, 3)
        epsi = gcomp.epsilon * cst.k

        x0ii = gcomp.x0kl(self.temp)
        xix = self.__xi_x()
        derxix = self.__der_xi_x()

        fac = 2 * pi * epsi * d_3 / pow(1-xix,3)

        t11 = 1 - xix/2
        t12 = segden * derxix
        t13 = 3 * (1-xix/2) / (1-xix) - 1/2
        t1 = t11 + t12 * t13

        t21 = 1 + pow(xix,2)
        t22 = segden*derxix
        t23 = 1 + 2 * xix + 3 * xix * (1+xix) / (1-xix) 
        t2 = 9/2 * (t21 + t22 * t23)

        result = fac * (t1 * mt.intI(x0ii,lam) - t2 * mt.intJ(x0ii,lam))
        return result

    def __der_a1kl(self, gcomp):
        x0ii = gcomp.x0kl(self.temp)
        rep = gcomp.rep
        att = gcomp.att
        ckl = gcomp.premie()

        tatt = self.__der_as1kl(gcomp, att) + self.__der_bkl(gcomp, att)
        trep = self.__der_as1kl(gcomp, rep) + self.__der_bkl(gcomp, rep)

        result = ckl * (pow(x0ii, att) * tatt - pow(x0ii, rep) * trep)
        return result

    def __gmieii(self, gcomp):
        gdhs = self.__gdhs(gcomp)
        g1 = self.__g1(gcomp)
        g2 = self.__g2(gcomp)
        b = 1 / (cst.k * self.temp)
        epsi = gcomp.epsilon * cst.k

        expt = b * epsi * g1 / gdhs + pow(b*epsi,2) * g2 / gdhs
        result = gdhs * exp(expt)
        return result

    def __gdhs(self, gcomp):
        xi_x = self.__xi_x()
        x0ii = gcomp.x0kl(self.temp)

        result = mt.gdhs(x0ii, xi_x)
        return result

    def __g1(self, gcomp):
        '''
        1 / (2pi epsiii hsdii^3) * [ 3 da1kl/dp - premieii * attii * x0ii^attii * (as1kl(attii) + Bkl(attii))/segden 
                                     +  premieii * repii * x0ii^repii * (as1kl(repii) + Bkl(repii))/segden ]
        '''
        premie = gcomp.premie()
        epsi = gcomp.epsilon * cst.k
        rep = gcomp.rep
        att = gcomp.att
        x0ii = gcomp.x0kl(self.temp)
        hsd = gcomp.hsdiam(self.temp) * cst.nmtom

        segden = self.__segden()

        t1 = 3* self.__der_a1kl(gcomp)
        t2 = premie * att * pow(x0ii, att) * (self.__as1kl(gcomp,att) + self.__bkl(gcomp,att)) / segden
        t3 = premie * rep * pow(x0ii, rep) * (self.__as1kl(gcomp,rep) + self.__bkl(gcomp,rep)) / segden

        result = 1 / (2 * pi * epsi * pow(hsd,3)) * (t1 - t2 + t3)

        return result

    def __g2(self, gcomp):
        '''
        (1+gammacii) * g2MCA(hsdii)
        '''
        xisx = self.__xi_sx()
        alii = self.__alphakl(gcomp)
        theta = exp(gcomp.epsilon /  self.temp) -1

        gammacii = cst.phi[6,0] * (-tanh(cst.phi[6,1] * (cst.phi[6,2]-alii)) + 1) * xisx * theta * exp(cst.phi[6,3]*xisx + cst.phi[6,4] * pow(xisx, 2))
        g2mca = self.__g2mca(gcomp)
        result = (1 + gammacii) * g2mca
        return result

    def __der_khs(self):
        xix = self.__xi_x()
        derxix = self.__der_xi_x()

        den = 1 + 4*xix + 4*pow(xix,2) - 4*pow(xix,3) + pow(xix,4)
        t1 = 4 * pow(1-xix,3) / den
        t2 = pow(1-xix,4) * (4 + 8*xix -12*pow(xix,2) + 4*pow(xix,3)) / pow(den,2)
        result = derxix * -(t1 + t2)

        return result

    def __der_a2kl(self, gcomp):
        khs = self.__khs()
        derkhs = self.__der_khs()
        epsi = gcomp.epsilon * cst.k
        ckl = gcomp.premie()
        x0kl = gcomp.x0kl(self.temp)
        rep = gcomp.rep
        att = gcomp.att

        t11 = pow(x0kl, 2*att) * (self.__as1kl(gcomp, 2*att) + self.__bkl(gcomp, 2*att))
        t12 = 2*pow(x0kl, att+rep) * (self.__as1kl(gcomp, att+rep) + self.__bkl(gcomp, att+rep))
        t13 = pow(x0kl, 2*rep) * (self.__as1kl(gcomp, 2*rep) + self.__bkl(gcomp, 2*rep))
        t1 =  derkhs * (t11 - t12 + t13)

        t21 = pow(x0kl, 2*att) * (self.__der_as1kl(gcomp, 2*att) + self.__der_bkl(gcomp, 2*att))
        t22 = 2*pow(x0kl, att+rep) * (self.__der_as1kl(gcomp, att+rep) + self.__der_bkl(gcomp, att+rep))
        t23 = pow(x0kl, 2*rep) * (self.__der_as1kl(gcomp, 2*rep) + self.__der_bkl(gcomp, 2*rep))
        t2 = khs * (t21 - t22 + t23)

        result = 0.5 * epsi * pow(ckl, 2) * (t1 + t2)
        return result

    def __g2mca(self, gcomp):
        '''
        1 / (2pi epsiii^2 hsdii^3) *
        [
        3 * d/dp (a2ii/(1+chi))
        - epsiii * KHS * premie^2 * rep * x0ii^2rep * (as1kl(2rep) + Bkl(2rep))/segden 
        + epsiii * KHS * premie^2 * (rep+att) * x0ii^(rep+att) * (as1kl(rep+att) + Bkl(rep+att))/segden 
        - epsiii * KHS * premie^2 * att * x0ii^2att * (as1kl(2att) + Bkl(2att))/segden 
        ]
        '''
        premie = gcomp.premie()
        epsi = gcomp.epsilon * cst.k
        rep = gcomp.rep
        att = gcomp.att
        x0ii = gcomp.x0kl(self.temp)
        hsd = gcomp.hsdiam(self.temp) * cst.nmtom
        khs = self.__khs()

        segden = self.__segden()

        f = lambda gcomp: self.__a_2kl(gcomp) / self.__corrf(gcomp)

        t1 = 3 * self.__der_a2kl(gcomp)
        t2 = epsi * khs * pow(premie,2) * rep * pow(x0ii, 2*rep) * (self.__as1kl(gcomp,2*rep) + self.__bkl(gcomp,2*rep)) / segden
        t3 = epsi * khs * pow(premie,2) * (rep+att) * pow(x0ii, rep+att) * (self.__as1kl(gcomp,rep+att) + self.__bkl(gcomp,rep+att)) / segden
        t4 = epsi * khs * pow(premie,2) * att * pow(x0ii, 2*att) * (self.__as1kl(gcomp,2*att) + self.__bkl(gcomp,2*att)) / segden
        
        result = 1 / (2 * pi * pow(epsi,2) * pow(hsd, 3)) * (t1 - t2 + t3 - t4)

        return result

    ### for assoc


class Component(object):
    def __init__(self, mass, *args):
        self.gtypes = {}
        for group in args:
            mt.checkerr(isinstance(group, GroupType), "Use GroupType object to specify groups used as optional arguments")
            self.gtypes[group] = 0
        self.groups = {}
        self.mass = mass

    def __repr__(self):
        return '<Component obj with {:2d} groups of {:2d} group types>'.format(len(self.groups),self.ngtype())

    def ngtype(self):
        return len(self.gtypes)

    def ngroups(self):
        return len(self.groups)

    def add_group(self, name, gtype):
        mt.checkerr(isinstance(gtype, GroupType), "Group type invalid, use GroupType object")
        mt.checkerr(isinstance(name, str), "Use str for name")
        mt.checkerr(name not in self.groups, "Group name taken")
        if (gtype not in self.gtypes):
            self.gtypes[gtype] = 0
        self.groups[name] = Group(gtype)
        self.gtypes[gtype] = self.gtypes[gtype] + 1

        return self.groups[name]

    def connect(self, name1, name2, angle=None):
        mt.checkerr(name1 in self.groups and name2 in self.groups, "Name missing from list of groups")

        self.groups[name1].connect(self.groups[name2], angle)

    def quick_set(self, *args):
        gtypel = []
        numl = []
        for arg in args:
            mt.checkerr(isinstance(arg, tuple) or isinstance(arg, list), "Use iterable tuple or list for components and no. of molecules")
            mt.checkerr(isinstance(arg[0], GroupType), "First element of iterable must be of GroupType")
            mt.checkerr(isinstance(arg[1], int), "Second element of iterable must be of int")
            gtypel.append(arg[0])
            numl.append(arg[1])

        names = []
        count = 1
        groupsAdding = sum(numl)

        for i in range(groupsAdding):
            name = 'G{:03d}'.format(count)
            while (name in self.groups) or (name in names):
                count = count+1
                name = 'G{:03d}'.format(count)
            names.append(name)

        faults = 0
        for i in range(len(gtypel)):
            if isinstance(gtypel[i], GroupType):
                for j in range(numl[i]):
                    self.add_group(names[sum(numl[0:i])+j-faults], gtypel[i])
            else:
                print("Not GroupType obj error, groups not added")
                faults = faults + numl[i]
        print('{:3d} groups were to be added, {:3d} groups were added'.format(groupsAdding, groupsAdding-faults))

        return self

    def thdebroglie(self, temp):
        '''
        Takes in component mass (au) and temperature (K) and return thermal de broglie wavelength
        '''
        Lambda_sq = pow(cst.h,2) * 1e3 * cst.Na / (2 * pi * self.mass * cst.k * temp)
        # return sqrt(Lambda_sq)
        return cst.h / sqrt(2 * pi * cst.mass_e * cst.k * temp)

    def get_gtypeii(self):
        sig = 0.
        epi = 0.
        rep = 0.
        att = 0.

        for g1 in self.gtypes:
            zki = self.__zki(g1)
            for g2 in self.gtypes:
                zli = self.__zki(g2)
                sig += zki * zli * (g1+g2).sigma
                epi += zki * zli * (g1+g2).epsilon
                rep += zki * zli * (g1+g2).rep
                att += zki * zli * (g1+g2).att

        g = GroupType(rep, att, sig, epi, shape_factor=None, id_seg=None)
        return g

    def __gshape(self, gtype):
        '''
        Returns this group shape stuff in this component
        '''
        if gtype in self.gtypes:
            numg_ki = self.gtypes[gtype]
            vk = gtype.vk
            sk = gtype.sk
            result = numg_ki * vk * sk
        else:
            result = 0
        return result

    def __gshapemol(self):
        '''
        Returns sum of all group shape stuff in this component
        '''
        result = 0.
        for g in self.gtypes:
            numg_ki = self.gtypes[g]
            vk = g.vk
            sk = g.sk
            result += numg_ki * vk * sk

        return result

    def __zki(self, gtype):
        return self.__gshape(gtype) / self.__gshapemol()

class GroupType(object):
    def __init__(self, lambda_r, lambda_a, sigma, epsilon, shape_factor=1, id_seg=1):
        self.rep = lambda_r
        self.att = lambda_a
        self.sigma = sigma # units nm
        self.epsilon = epsilon # units K input, divided by cst.k (epsi / k), so multiply k here
        self.sk = shape_factor # dimensionless segments
        self.vk = id_seg # identical segments in a group

    def __repr__(self):
        return '<GroupType({:4.3f} nm, {:4.3f} K, rep={:5.3f}, att={:5.3f})>'.format(self.sigma, self.epsilon, self.rep, self.att)

    def hsdiam(self, si_temp, x_inf=0): # returns in nm
        return mt.hsdiam(si_temp / self.epsilon, self.rep, self.att, x_inf) * self.sigma

    def __add__(self, other):
        sig = (self.sigma + other.sigma) / 2
        epsi = sqrt(pow(self.sigma, 3) * pow(other.sigma, 3)) / pow(sig,3) * sqrt(self.epsilon * other.epsilon)
        rep = 3 + sqrt( (self.rep - 3) * (other.rep - 3) )
        att = 3 + sqrt( (self.att - 3) * (other.att - 3) )

        return GroupType(rep, att, sig, epsi, shape_factor=None, id_seg=None)

    def premie(self):
        return self.rep/(self.rep-self.att) * pow(self.rep/self.att, self.att/(self.rep-self.att))

    def x0kl(self, si_temp):
        x = self.sigma / self.hsdiam(si_temp)
        mt.checkwarn(x >= 1, "x0kl/x0ii below 1, could result in inaccurate representation of values")
        mt.checkwarn(x <= sqrt(2), "x0kl/x0ii above sqrt 2, could result in inaccurate representation of values")
        return x

class Group(object):
    def __init__(self, gtype):
        self.gtype = gtype
        self.connections = 0
        self.connectedTo = {};

    def __repr__(self):
        return '<Group obj of {}>'.format(repr(self.gtype))

    def connected(self, other):
        if other in self.connectedTo:
            return True
        return False

    def connect(self, other, angle=None):
        mt.checkerr(isinstance(other, Group), "Groups can only connect to other groups")
        mt.checkerr(angle==None or (isinstance(angle, tuple) and len(angle)==3), "Either don't specify angle, or use 3d vector tuple")
        self.connections = self.connections+1;
        self.connectedTo[other] = angle

        if not other.connected(self):
            other.connect(self)

    def clear_angle(self, other):
        if other in self.connectedTo:
            self.connectedTo[other] = None
            other.connectedTo[self] = None

    def set_angle(self, other, angle):
        mt.checkerr(isinstance(angle, tuple) and len(angle)==3, "Use 3d vector tuple to specify angle")
        if other in self.connectedTo:
            self.connectedTo[other] = angle
            other.connectedTo[self] = mt.inv_angle(angle)

def main():
    ch = GroupType(15.050,6,0.40772,256.77, shape_factor=0.5, id_seg=1)
    lj = GroupType(12,6,0.4,250)

    c6h = GroupType(19.32993437, 6., 0.450874, 377.0118945, shape_factor=1)

    ch4 = GroupType(16.39077548, 6., 0.375227, 170.7540156)
    methane = Component(16.04)
    methane.quick_set((ch4,1))

    to = GroupType(11.79626055, 6., 0.368461, 268.2422859)
    toluene = Component(92.14)
    toluene.quick_set((to,3))

    hexane = Component(86.1754)
    hexane.quick_set((c6h,2)) 

    s = System()
    # s.quick_set([meth, ljmol], [100, 100])
    s.quick_set((hexane,1000)) 
    # print(s.comps, s.moles)
    # print(lj.hsdiam(273), ch.hsdiam(273))

    # print(meth.param_ii("sigma"), meth.param_ii("hsd",273.), meth.param_ii("rep"))
    # print(octane.param_ii("sigma"), octane.param_ii("hsd",273.), octane.param_ii("rep"))
    # print(ljmol.param_ii("sigma"), ljmol.param_ii("hsd",273.), ljmol.param_ii("rep"))

    tm = 300.
    s.temp = tm
    vm = 0.12725355948027697
    vn = s._System__moltol() * vm / (cst.Na * pow(cst.nmtom,3))
    s.volume = s._System__moltol() * vm / (cst.Na * pow(cst.nmtom,3))
    print('cgss is given by: ', s._System__cgshapesum())
    print('A-IDEAL term: =======')
    print('{:18s}'.format('value: '), s.a_ideal())
    print('{:18s}'.format('thermal debrog: '), hexane.thdebroglie(s.temp))
    print('A-MONO term: ========')
    print('{:18s}'.format('value: '), s.a_mono())
    print('{:18s}'.format('a_hs: '), s._System__a_hs())
    print('{:18s}'.format('a_1: '), s._System__a_1()*(cst.k*s.temp))
    print('{:18s}'.format('a_2: '), s._System__a_2()*(cst.k*s.temp)**2)
    print('{:18s}'.format('a_3: '), s._System__a_3()*(cst.k*s.temp)**3)
    print('A-CHAIN term: =======')
    print('{:18s}'.format('value: '), s.a_chain())
    print('{:18s}'.format('g-mie: '), s._System__gmieii(hexane.get_gtypeii()))
    print('{:18s}'.format('gdhs: '), s._System__gdhs(hexane.get_gtypeii()))
    print('{:18s}'.format('g1: '), s._System__g1(hexane.get_gtypeii()))
    print('{:18s}'.format('g2: '), s._System__g2(hexane.get_gtypeii()))
    print('=====================')
    print('{:18s}'.format('A/NkT: '), s.a_ideal() + s.a_mono() + s.a_chain())
    print('{:18s}'.format('Density (mol/m3):'), s._System__nden()/cst.Na)
    print('{:18s}'.format('System size:'), s._System__moltol())
    print('=====================')
    
    (A, P, S, G, H) = s.get_state(vm, temperature=tm)
    print('{:18s}'.format('Pressure (MPa):'), P*1e-6)
    print('{:18s}'.format('A per mol:'), A*cst.Na)
    print('{:18s}'.format('S (J/(K mol)):'), S*cst.Na)
    print('{:18s}'.format('G (J/mol):'), G*cst.Na)
    print('{:18s}'.format('H (J/mol):'), H*cst.Na)
    print('=====================')

    # print('Testing vapour_pressure')
    # VP, vx, EQG = s.vapour_pressure(490, initial_guess=(0.0001,0.06), get_volume=True, get_gibbs=True, print_progress=True)
    # print('{:18s}'.format('Vapour Pressure:'), VP*1e-6)
    # getv = s.single_phase_v(20e5, T=400)
    # print('{:18s}'.format('v at 20bar 400K:'), getv)
    print()
    print('='*21)
    print('Testing locating critical point')
    Pc, Tc, vc = s.critical_point(initial_t=500., print_progress=True, get_volume=True)
    print('{:18s}'.format('P_crit (bar):'), Pc*cst.patobar)
    print('{:18s}'.format('T_crit (K):'), Tc)
    print('{:18s}'.format('V_crit (m3/mol):'), vc)
    print('{:18s}'.format('rho_crit (mol/m3):'), 1/vc)

    # v = np.logspace(-4,2,100)
    # P = s.p_v_isotherm(v, temperature=Tc)
    # fig, ax = plt.subplots()
    # ax.semilogx(v, P*cst.patobar,'r-')
    # ax.semilogx([vc, vc], [-15,Pc*cst.patobar], 'b-')
    # ax.set_ylim([-15,30])
    # plt.show()

    # vn = mt.m3mol_to_nm(vx[0], 1000)
    # s.volume = vn*(1+1e-10)
    # Ap = s.helmholtz()
    # s.volume = vn*(1-1e-10)
    # Am = s.helmholtz()
    # print((Am-Ap)/(2e-10*vn)/pow(cst.nmtom,3))
    # v = np.logspace(-4,2,500)
    # P, G = s.p_v_isotherm(v, temperature=400, gibbs=True)
    # pl = Plot(v,P*cst.patobar, label="temp = {:5.1f}".format(tm), color='r', axes="semilogx")
    # pl2 = Plot(v,G*cst.Na, label="temp = {:5.1f}".format(tm), color='r', axes="semilogx")
    # hline = Plot([1e-4,1e2],[VP*cst.patobar, VP*cst.patobar], color='k', axes="semilogx")
    # hline2 = Plot([1e-4,1e2],[EQG*cst.Na, EQG*cst.Na], color='k', axes="semilogx")
    # vline11 = Plot([vx[0],vx[0]],[0,15], color='b', axes="semilogx")
    # vline12 = Plot([vx[1],vx[1]],[0,15], color='b', axes="semilogx")
    # vline21 = Plot([vx[0],vx[0]],[0.5e4,-10e4], color='b', axes="semilogx")
    # vline22 = Plot([vx[1],vx[1]],[0.5e4,-10e4], color='b', axes="semilogx")
    # g = Graph(legends=True, subplots=2)
    # g.add_plots(pl,hline,vline11,vline12, subplot=1)
    # g.add_plots(pl2,hline2,vline21,vline22, subplot=2)
    # g.set_xlabels("volume (m3/mol)", "volume (m3/mol)")
    # g.set_ylabels("pressure (bar)", "Gibbs free energy")
    # g.ylim(0,15,1)
    # g.ylim(-5e4,5e4,2)
    # g.xlim(1e-4,1e2,1)
    # g.xlim(1e-4,1e2,2)
    # g.draw()

    # print("="*5,"Generating hexane (19.33/6.0) data for ML (v,T->P)","="*5)

    # temp_range = np.linspace(300,450,200)
    # p_vap = np.zeros(np.size(temp_range))
    # vv = np.zeros(np.size(temp_range))
    # vl = np.zeros(np.size(temp_range))
    # gf = np.zeros(np.size(temp_range))
    # tlen = len(temp_range)
    # tenp = tlen//10
    # pl_p = []
    # pl_g = []
    # pl_pr = []
    # pl_gr = []
    # v = np.logspace(-4,2,1000)
    # trows = []
    # for i in range(len(temp_range)):
    #     t = temp_range[i]
    #     Pv, vlv, Geq = s.vapour_pressure(t, get_volume=True, get_gibbs=True, print_results=False)
    #     P, G = s.p_v_isotherm(v, temperature=t, gibbs=True)
    #     p_vap[i] = Pv*cst.patobar
    #     vv[i] = np.max(vlv)(i)
    #     vl[i] = np.min(vlv)
    #     gf[i] = Geq*cst.Na
    #     p_real = np.copy(P)
    #     p_real[(v > vl[i]) & (v < vv[i])] = Pv
    #     g_real = np.copy(G)
    #     g_real[(v > vl[i]) & (v < vv[i])] = Geq
    #     pl_p.append(Plot(v,P*cst.patobar, label="T = {:5.1f}K".format(t), color='r', axes="semilogx"))
    #     pl_g.append(Plot(v,G*cst.Na, label="T = {:5.1f}K".format(t), color='r', axes="semilogx"))
    #     pl_pr.append(Plot(v,p_real*cst.patobar, label="T = {:5.1f}K".format(t), color='r', axes="semilogx"))
    #     pl_gr.append(Plot(v,g_real*cst.Na, label="T = {:5.1f}K".format(t), color='r', axes="semilogx"))
    #     l_phase = np.where(v <= vl[i], 1, 0)
    #     v_phase = np.where(v >= vv[i], 1, 0)
    #     vl_2phase = np.where((v > vl[i]) & (v < vv[i]), 1, 0)
    #     l_frac = np.where((v > vl[i]) & (v < vv[i]), (vv[i] - v)/(vv[i] - vl[i]), v <= vl[i])
    #     v_frac = np.where((v > vl[i]) & (v < vv[i]), (v - vl[i])/(vv[i] - vl[i]), v >= vv[i])

    #     if (i+1) % tenp == 0:
    #         print(f'Vapour pressure and p-v isotherm calculations at {(i+1):3d} iterations')

    #     trows.append(np.column_stack([v,1/v, np.array([temp_range[i]]*len(v)), P*cst.patobar, p_real*cst.patobar, G*cst.Na, g_real*cst.Na, l_phase, v_phase, vl_2phase, l_frac, v_frac]))


    # df = pd.DataFrame(np.vstack(trows))
    # df.to_csv('n-hexane-vle.csv', index=False)
    # g = Graph(subplots=4, titles=["P vs v","G vs v","P vs v","G vs v"])
    # g.add_plots(*pl_p,Plot(vv, p_vap, label="Vapour phase boundary", color='b', axes="semilogx"),Plot(vl, p_vap, label="Liquid phase boundary", color='b', axes="semilogx"), subplot=1)
    # g.add_plots(*pl_g,Plot(vv, gf, label="Vapour phase boundary", color='b', axes="semilogx"),Plot(vl, gf, label="Liquid phase boundary", color='b', axes="semilogx"), subplot=2)
    # g.add_plots(*pl_pr,Plot(vv, p_vap, label="Vapour phase boundary", color='b', axes="semilogx"),Plot(vl, p_vap, label="Liquid phase boundary", color='b', axes="semilogx"), subplot=3)
    # g.add_plots(*pl_gr,Plot(vv, gf, label="Vapour phase boundary", color='b', axes="semilogx"),Plot(vl, gf, label="Liquid phase boundary", color='b', axes="semilogx"), subplot=4)
    # g.set_xlabels(*["volume (m3/mol)"]*4)
    # g.set_ylabels(*["pressure (bar)","gibbs free energy (J/mol)"]*2)
    # g.ylim(0,20,1)
    # g.ylim(-1e4,1e4,2)
    # g.xlim(1e-4,1e1,1)
    # g.xlim(1e-4,1e1,2)
    # g.ylim(0,20,3)
    # g.ylim(-1e4,1e4,4)
    # g.xlim(1e-4,1e1,3)
    # g.xlim(1e-4,1e1,4)
    # g.draw()

    # colors = list('rbgcmyk')
    '''
    {    T,   p0/10^6,  1/vlo,   1/vvo, dhf/1000.0,     aact,     aid,   amono,        ac}
    { 300., 0.0253446, 7631.7, 10.2824,    31.5377, -1823.69, 4.90344, -5.2177, -0.416868}
    {    ahs1,            a1m1,            a2m1,            a3m1,  gmie1,   gdhs,      g11,       g22}
    { 3.41204, -2.38726*10^-20, -4.41287*10^-42, -4.35973*10^-66, 1.5172, 3.7868, -2.83953, 0.0663955}
    vl, vv
    (0.00013103240431358676, 0.09725355948027697)
    '''
if __name__ == '__main__':
    main()
