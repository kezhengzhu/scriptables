
def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

def isAnyIn(d, *args):
    for arg in args:
        if arg in d:
            return True
    return False

def isAllIn(d, *args):
    for arg in args:
        if arg not in d:
            return False
    return True

def getVal(d, *keylist):
    for item in keylist:
        if item in d:
            return d[item]
    return None

class MDPOption(object):
    def __init__(self, opt, default, comment=None, alt=None):
        assert(isinstance(opt, str))
        checkerr(isinstance(default, str) or isinstance(default, int) or isinstance(default, float), 
            "Value type not str, int or float")

        self.opt = opt
        self.default = default
        self.set_value(self.default)
        self.set_comment(comment)
        self.set_alt(alt)

    def __repr__(self):
        s = ' '.join(self.optval().split())
        return "MDP Option <{}>: ".format(self.opt) + (s.replace("\n", "") if (len(s) > 0) else "no value assigned".format(self.opt))

    def set_value(self, val):
        checkerr(isinstance(val, str) or isinstance(val, int) or isinstance(default, float) or val == None, 
            "Value type not str, int or float")
        self.val = val

    def set_comment(self, comment):
        if isinstance(comment, str):
            self.comment = comment
        else:
            self.comment = None

    def set_alt(self, alt):
        if isinstance(alt, MDPOption):
            self.alt = alt
        else:
            self.alt = None

    def optval(self, dp=2):
        val = self.val
        if val == None:
            try:
                val = self.alt.val
            except:
                return ""

        if isinstance(val, str):
            optval = "{:<24}= {:<16}".format(self.opt, val)
        elif isinstance(val, int):
            optval = "{:<24}= {:<16d}".format(self.opt, val)
        elif isinstance(val, float):
            s = "{:<24}= {:<16." + str(dp) + "f}"
            optval = s.format(self.opt, val)
        else:
            return ""
        if isinstance(self.comment,str):
            return optval + "; {}\n".format(self.comment)
        return optval + "\n"

    def reset(self):
        self.val = self.default


class MDRun(object):
    ensembles = ["nvt", "npt", "nve", "em"]
    units = {
        't': {'ps': 1, 'fs': 1e-03, 'ns': 1e+03, 's': 1e+12},
        'x': {'nm': 1, 'mm': 1e+06, 'am': 1e-01, 'm': 1e+09}
    }
    defaults = {
        'dt': 0.002, # ps
        'tunit': 'ps',
        'xunit': 'nm',
        'nsteps': 1000000,
        'xvfop': 1000,
        'enop': 10,
        'temperature': None,
        'pressure': None,
        'title': "MD Run",
        'emtol': None,
    }
    temp_syn = ["temperature", "temp", "T"]
    pres_syn = ["pressure", "pres", "P"]
    emtol_syn = ["emtol", "etol", "emtolerance", "emt"]
    mdp_options = {
        'integrator', 'tinit', 'dt', 'nsteps', 'emtol', 'emstep',
        'nstxout', 'nstvout', 'nstfout', 'nstcalcenergy', 'nstenergy',
        'nstlog', 'nstxtcout', 'xtc-precision', 'comm-mode', 'nstcomm',
        'energygrps', 'energygrp_table',
    }
    mdp_opt_alt = {}

    def __init__(self, ens, **kwargs):
        self.ens = ens.lower()
        checkerr(not self.ens == "nvt" or isAnyIn(kwargs, *MDRun.temp_syn),
            "NVT ensemble requires \'temperature\' argument as input")
        checkerr(not self.ens == "npt" or (isAnyIn(kwargs, *MDRun.pres_syn) and isAnyIn(kwargs, *MDRun.temp_syn)),
            "NPT ensemble requires \'pressure\' argument as input")
        checkerr(not self.ens == "em" or isAnyIn(kwargs, *MDRun.emtol_syn),
            "Energy minimisation requires \'emtol\' argument as input")

        if isAnyIn(kwargs, *MDRun.temp_syn):
            kwargs['temperature'] = getVal(kwargs, *MDRun.temp_syn)
        if isAnyIn(kwargs, *MDRun.pres_syn):
            kwargs['pressire'] = getVal(kwargs, *MDRun.pres_syn)
        if isAnyIn(kwargs, *MDRun.emtol_syn):
            kwargs['emtol'] = getVal(kwargs, *MDRun.emtol_syn)

        self.params = MDRun.defaults.copy()
        self.params.update(kwargs)

        self.set_params(**self.params)

    def set_params(self, **kwargs):

        self.dt = kwargs['dt'] # Time step
        self.nsteps = kwargs['nsteps'] # No. of steps
        self.tinit = kwargs['tinit'] # Initial timestep
        self.tunit = kwargs['tunit'] # time unit
        self.xunit = kwargs['xunit'] # positional units
        self.xvfop = kwargs['xvfop'] # position, velocity, force for how many steps
        self.enop = kwargs['enfop'] # energy output for how many steps

        return

    def __repr__(self):
        return "MDRun of {} ensemble".format(self.ens.upper())


def write_opt(opt, val, comment=None, dp=2):
    assert(isinstance(opt, str))
    if isinstance(val, str):
        optval = "{:<24}= {:<16}".format(opt,val)
    elif isinstance(val, int):
        optval = "{:<24}= {:<16d}".format(opt,val)
    elif isinstance(val, float):
        s = "{:<24}= {:<16." + str(dp) + "f}"
        optval = s.format(opt,val)
    else:
        raise Exception("Value type not str, int or float")

    if isinstance(comment,str):
        return optval + "; {}\n".format(comment)
    return optval + "\n"

def run_param(ens, dt, nsteps, tinit=0, units="ps", emtol=20.0):
    if ens.lower() == "em":
        emstep = dt
    dt_in = dt
    if units.lower() == "fs":
        dt *= 1e-3
    elif units.lower() == "ns":
        dt *= 1e3
    elif units.lower() == "s":
        dt *= 1e12
    elif units.lower() != "ps":
        raise Exception("Error in units input: please use only s, ps, ns or fs.")

    s = ""
    s += "; Run parameters \n"
    if ens.lower() in ["nve", "nvt", "npt"]:
        s += write_opt("integrator", "md" , comment="leap-frog integrator for Newton's equations of motion")
        s += write_opt("tinit", tinit) # in units ps by default
        s += write_opt("dt", dt, comment="time step of {:<9.4f}{}".format(dt_in, units.lower()))
        s += write_opt("nsteps", nsteps, comment="no. of steps, total time = {:<9.4f}{}".format(dt_in*nsteps, units.lower()))
    elif ens.lower() == "em":
        s += write_opt("integrator", "steep", comment="steepest descent algorithm for energy minimization")
        s += write_opt("nsteps", nsteps, comment="maximum no. of steps to minimize")
        s += write_opt("init_step", 0, comment="init step for non-eqm parameters")
        s += write_opt("emtol", emtol, comment="stop minimization when max force < {} kJ/mol/nm".format(emtol))
        s += write_opt("emstep", emstep, comment="em step size of {} nm".format(emstep))
    else:
        raise Exception("Unrecognised ensemble input!")

    return s + "\n"

def op_control(xvfop, enop, dt, units="ps"):
    s = ""
    s += "; Output control \n"
    s += write_opt("nstxout", xvfop, comment="save coordinates every {:5.2f}{}".format(dt*xvfop, units))
    s += write_opt("nstvout", xvfop, comment="save velocities every {:5.2f}{}".format(dt*xvfop, units))
    s += write_opt("nstfout", xvfop, comment="save forces every {:5.2f}{}".format(dt*xvfop, units))
    s += write_opt("nstlog", xvfop, comment="update log file every {:5.2f}{}".format(dt*xvfop, units))
    s += write_opt("nstenergy", enop, comment="save energies every {:5.2f}{}".format(dt*enop, units))
    s += write_opt("nstcalcenergy", enop, comment="calculate energies every {:5.2f}{}".format(dt*enop, units))


    return s + "\n"



def write_mdp(fname, ens, dt, nsteps, xvfop=1000, enop=1, temperature=None, pressure=None, title="MD Run", emtol=None):
    emargs = []
    if ens.lower() == "em":
        if not (isinstance(emtol, float) or isinstance(emtol, int)):
            raise Exception("Energy minimization ensemble was not given emtol keyword argument!")
        emargs.append(emtol)
        # dt functions as emstep for em ensemble

    fout = open(fname, "w")
    fout.write("; Title = {}, Filename = {}\n\n".format(title,fname))
    
    fout.write(run_param(ens, dt, nsteps, *emargs))



def main():
    print(write_opt("integrator","md",comment="leap-frog integrator"), end="")
    print(write_opt("dt",0.005,dp=3, comment="who this"))
    print(run_param("em", 0.01, 10000, emtol=20.0))
    print(op_control(1000, 100, 0.01))

if __name__ == '__main__':
    main()


# def write_mdp(ensemble="NVE", temperature=None, pressure=None, )
# template = ''';title      = Methane

# ; Run parameters
# integrator  = md        ; leap-frog integrator
# tinit = 0
# dt =  0.005
# nsteps = 1000000 

# ; Output control
# nstxout     = 1000      ; save coordinates every 0.01 ps
# nstvout     = 1000      ; save velocities every 0.01 ps
# nstfout     = 1000      ; save forces every 0.01 ps
# nstcalcenergy   = 1
# nstenergy   = 1         ; save energies every 0.01 ps
# nstlog      = 1000      ; update log file every 0.01 ps
# nstxtcout       = 1000
# xtc-precision   = 1000
# comm-mode       = Linear
# nstcomm         = 1
# energygrps               = MET
# energygrp_table          = MET MET

# ; Bond parameters
# continuation            = no        ; first dynamics run
# constraint_algorithm    = lincs   ; holonomic constraints 
# constraints             = none 
# lincs_iter              = 3         ; accuracy of LINCS
# lincs_order             = 4         ; also related to accuracy

# ; Neighborsearching
# cutoff-scheme   = Group
# ns_type         = grid      ; search neighboring grid cells
# nstlist         = 10        ; 20 fs, largely irrelevant with Verlet
# rcoulomb        = 2         ; short-range electrostatic cutoff (in nm)  
# rlist           = 2         ; nblist cut-off 

# ; Electrostatics
# coulombtype     = cut-off   ; Particle Mesh Ewald for long-range electrostatics
# epsilon-r       = 1

# ; Method for doing Van der Waals
# vdw-type        = user
# ;vdw-modifier   = Potential-shift-Verlet      
# rvdw            = 2

# ; Spacing for the PME/PPPM FFT grid
# fourierspacing  = 0.12

# ; FFT grid size, when a value is 0 fourierspacing will be used
# fourier-nx      = 0
# fourier-ny      = 0
# fourier-nz      = 0

# ; EWALD/PME/PPPM parameters
# pme_order       = 4     ; cubic interpolation
# ewald-rtol      = 1e-05

# ; Apply long range dispersion corrections for Energy and Pressure
# DispCorr        = No

# ; Temperature coupling is on
# tcoupl      = nose-hoover   ; modified Berendsen thermostat
# tc-grps     = MET       ; two coupling groups - more accurate
# tau_t       = 2     ; time constant, in ps
# ref_t       = 293.15    ; reference temperature, one for each group, in K

# ; Pressure coupling is on
# pcoupl           = no  ; no pressure coupling in NVT

# ; Periodic boundary conditions
# pbc         = xyz       ; 3-D PBC

# ; SIMULATED ANNEALING CONTROL = 
# annealing   = no

# ; Velocity generation
# gen_vel     = no        ; assign velocities from Maxwell distribution
# gen_temp    = 333.15        ; temperature for Maxwell distribution
# gen_seed =  536373
# '''
# print(template)