
# This is program that writes function which can generate molecule.itp file 
# and molecule.gro file automatically for coarse-grained beads
# write

# Version info: this program currently only works for long chained beads of 
# length m beads.


import numpy as np 
import pandas as pd 

def write_grochain(molecule, bead, purefile, mbeads=1, title=""):
    '''
    Function takes following inputs:
    molecule name: such as MET, ETH, DEC, WATER, etc etc 
    bead name: such as CH4, mustn't be the same as molecule
    purefile: xlsx or csv file for pure components
    mbeads: how many beads in a chain
    title: just name of molecule file
    '''
    assert(type(mbeads) == int and type(molecule) == str and type(bead) == str and type(title) == str)

    if (type(purefile) is str) and (purefile[-5:].lower() == ".xlsx"):
        df = pd.read_excel(purefile)
    elif (type(purefile) is str) and (purefile[-4:].lower() == ".csv"):
        df = pd.read_csv(purefile)
    else:
        raise Exception("Unknown variable of file name or wrong file extension (only .xlsx and .csv)")

    if len(title) == 0:
        title = "CG molecule of {} with {} {} beads".format(molecule,mbeads,bead)
        longname = "cgmolecule{}".format(molecule.lower())
    else:
        longname = title.replace(" ", "").lower()

    # Linear chain coordinates calculations
    ind = df.index[df.iloc[:,0] == bead].tolist()
    sigma = np.asscalar(df.iloc[ind[0],6])*1.0

    chainlen = sigma * (mbeads-1)

    # Writing .gro file
    fout = open("{}.gro".format(longname), "w")
    fout.write("{}\n".format(title))
    fout.write("{:>5d}\n".format(mbeads))

    for i in range(mbeads):
        x = -chainlen/2 + i*sigma
        y = 0.0
        z = 0.0
        # Writing has the following format {:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.4f}{:>8.4f}{:>8.4f}
        fout.write("{:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(1,molecule,bead,i+1,x,y,z))
    fout.write("{:>10.5f}{:>10.5f}{:>10.5f}\n".format(0.0,0.0,0.0))
    fout.close()

    # Writing .itp forcefield files
    fout = open("{}.itp".format(longname), "w")
    fout.write("; {}\n\n".format(longname)) 
    fout.write("[ moleculetype ]\n; molname nrexcl\n") 
    # nrexcl indicates from which neighbouring (bonded) atom away should non-bonded interactions be excluded.
    # For now it's gonna be 1
    fout.write("  {:<7}{:<3d}\n\n".format(molecule,1))
    strs = ["nr", "type", "resnr", "res", "atom", "cgnr", "charge", "mass"]
    fout.write("[ atoms ]\n; {:<6}{:<6}{:<7}{:<6}{:<6}{:<6}{:<8}{:<6}\n".format(*strs))
    # cgnr = charge group number. no charges so 1 atom = 1 charge group, ignore last 3 columns
    # resnr = residue number. we consider 1 atom = 1 residuce
    for i in range(mbeads):
        fout.write("  {:<6d}{:<6}{:<7d}{:<6}{:<6}\n".format(i+1,bead,1,molecule,bead,i+1))
    if mbeads > 1:
        strs = ["i","j","func","r0 (nm)","kb (kJ/(mol nm2))"]
        fout.write("\n[ bonds ]\n; {:<5}{:<5}{:<6}{:<10}{:<10}\n".format(*strs))
        for i in range(mbeads-1):
            fout.write("{:<5d}{:<5d}{:<6d}{:<10.4f}{:<10.1f}\n".format(i+1,i+2,1,sigma,kb))
    fout.write("\n")
    fout.close()
    return