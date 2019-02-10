# Write topology based on available inputs

import numpy as np 
import pandas as pd 
from nbtable import *
from gencg import *
import os

def writeff(pureCompList, outname="forcefield.itp",nonbonded="ffnonbonded.itp"):
    assert(type(pureCompList) == list and len(pureCompList) > 0)
    fout = open(outname,"w")
    fout.write("; Coarse-grained forcefield with Mie potential parameters\n\n")
    fout.write("[ defaults ]\n; {:<12}{:<12}{:<12}{:<12}{:<12}\n".format("nbfunc","comb_rule","gen-pairs","fudgeLJ","fudgeQQ"))
    fout.write("  {:<12d}{:<12d}\n\n".format(1,1))
    fout.write("#include \"{}\"\n".format(nonbonded))
    if type(pureCompList) == list:
        for fin in pureCompList:
            fout.write("#include \"{}.itp\"\n".format(fin.lower()))
    else:
        fout.write("#include \"{}.itp\"\n".format(pureCompList.lower()))
    fout.close()
    return

def write_topol_base(molList,nmol,outname="topol.top",sys="Pure",ff="forcefield.itp"):
    fout = open(outname, "w")
    fout.write("; Topology file\n\n")
    fout.write("#include \"{}\"\n\n".format(ff.lower()))
    fout.write("[ system ]\n{}\n\n".format(sys))
    fout.write("[ molecules ]\n")

    if type(molList) == list and type(nmol) == list:
        for i in range(len(molList)):
            fout.write("{:<8}{:<6d}\n".format(molList[i],nmol[i]))
    elif type(molList) == list and type(nmol) == int:
        for i in range(len(molList)):
            fout.write("{:<8}{:<6d}\n".format(molList[i].upper(),nmol))
    else:
        fout.write("{:<8}{:<6d}\n".format(molList.upper(),nmol))
    fout.close()

def main():
    os.chdir("octtest")
    writeffnb("pure_comp2.xlsx")
    gen_nb_table("pure_comp2.xlsx")

    molList, titleList= gen_grochain("pure_comp2.xlsx")
    writeff(titleList)
    write_topol_base(molList,500)
    return

if __name__ == '__main__':
    main()
else:
    print("Loading functions from writetopol.py")