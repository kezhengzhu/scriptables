
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


def write_mdp(fname, ens, dt, nsteps, xvfop=1000, enop=1, temperature=None, pressure=None, title="MD Run"):
    fout = open(fname, "w")
    fout.write("; Title = {}, Filename = {}\n\n".format(title,fname))
    fout.write("; Run parameters \n")
    if fname.lower() == "nve":

    elif fname.lower() == "nvt":

    elif fname.lower() == "npt":

    elif fname.lower() == "em":
        fout.write(write_opt("integrator", "steep", comment="steepest descent algorithm for energy minimization"))
        fout.write(write_opt("nsteps", 50000, comment="maximum no. of steps to minimize"))
        fout.write(write_opt("", , comment=""))
    else:
        raise Exception("Unrecognised ensemble input!")

    fout.write(write_opt("integrator", "md" , comment=""))


def main():
    print(write_opt("integrator","md",comment="leap-frog integrator"), end="")
    print(write_opt("dt",0.005,dp=3, comment="who this"))


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