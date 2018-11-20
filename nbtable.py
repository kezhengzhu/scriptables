
# This is a program that auto generates table and non-bonded forcefield .itp
# files. This can also be loaded as a module for the following functions:
# writeffnb(): creates non-bonded forcefield itp file
# genTable(): creates forcefield interaction tables

import numpy as np
import pandas as pd 

def writeffnb(purefile,nbfile="",output="ffnonbonded"):
    '''
    Function writes ffnonbonded.itp based on excel or csv input
    For pure components, excel csv files should be in format:
    comp, mass, charge, lambda_r, lambda_a, epsilon, sigma
    For comp-comp interactions, excel and csv file should be in the format:
    comp1, comp2, lambda_r, lambda_a, epsilon, sigma
    '''
    # Ensure reading file is correct
    if (type(purefile) is str) and (type(nbfile) is str):
        if(purefile[-5:].lower() == ".xlsx"):
            puredf = pd.read_excel(purefile)
        elif (purefile[-4:].lower() == ".csv"):
            puredf = pd.read_csv(purefile)
        else:
            raise Exception("Unknown file type for pure components, only .xlsx or .csv accepted")
        if(nbfile[-5:].lower() == ".xlsx"):
            nbdf = pd.read_excel(nbfile)
        elif (purefile[-4:].lower() == ".csv"):
            nbdf = pd.read_csv(nbfile)
        else:
            print("No non-bonded interactions table found for different compounds when writing non-bonded forcefield")
            nbdf = pd.DataFrame()
    else:
        raise Exception("Unknown variable types for pure compounds or non-bonded interactions files, please use str")

    # Opening file to write
    fout = open(output+".itp","w")
    fout.write("; Non-bonded forcefield with coarse grained Mie potential\n\n")

    # Write header of atomtypes: for same atom interactions
    strs = ["name","bond_type","mass","charge","ptype","C","A"]
    fout.write("[ atomtypes ]\n; {:<6}{:<11}{:<11}{:<9}{:<11}{:<13}{:<11}\n".format(*strs))
    atype = ""

    # Create string to append for unlike atom interactions
    nb_param = ""

    # Default parameters
    ptype = "A"

    # 1) Pure compounds intramolecular interactions definitions:
    for i in range(len(puredf)):
        comp = puredf.iloc[i,0]
        mass = np.asscalar(puredf.iloc[i,1])*1.0
        charge = np.asscalar(puredf.iloc[i,2])*1.0
        l_r = np.asscalar(puredf.iloc[i,3])*1.0
        l_a = np.asscalar(puredf.iloc[i,4])*1.0
        epsi = np.asscalar(puredf.iloc[i,5])*1.0
        sig = np.asscalar(puredf.iloc[i,6])*1.0

        c_mie = l_r/(l_r-l_a) * pow(l_r/l_a, l_a/(l_r-l_a))

        A = c_mie * epsi * pow(sig,l_r) # repulsion coefficient
        C = c_mie * epsi * pow(sig,l_a) # attractive coefficient

        atype += "  {:<6}{:<11}{:<11.5f}{:<9.3f}{:<11}{:<13.5e}{:<11.5e}\n".format(comp,"C",mass,charge,ptype,C,A)

    # 2) Non-bonded interactions parameters between different compounds
    for i in range(len(nbdf)):
        comp1 = nbdf.iloc[i,0]
        comp2 = nbdf.iloc[i,1]
        l_r = np.asscalar(nbdf.iloc[i,2])*1.0
        l_a = np.asscalar(nbdf.iloc[i,3])*1.0
        epsi = np.asscalar(nbdf.iloc[i,4])*1.0
        sig = np.asscalar(nbdf.iloc[i,5])*1.0

        c_mie = l_r/(l_r-l_a) * pow(l_r/l_a, l_a/(l_r-l_a))

        A = c_mie * epsi * pow(sig,l_r) # repulsion coefficient
        C = c_mie * epsi * pow(sig,l_a) # attractive coefficient

        nb_param += "  {:<5}{:<6}{:<6}{:<15.5e}{:<15.5e}\n".format(comp1,comp2,"1",C,A)

    # Write [ atomtypes ]
    fout.write(atype)
    fout.write("\n")

    # nonbond_params
    # ; i   j   func    C   A
    if len(nb_param) > 0:
        strs = ["i", "j", "func", "C", "A"]
        fout.write("[ nonbond_params ]\n; {:<5}{:<6}{:<6}{:<15}{:<15}\n".format(*strs))
        fout.write(nb_param)

    fout.close()

def gen_nb_table(purefile, nbfile="", cutoff=5.0, delr=0.002):
    '''
    Function writes table_{COMP1}_{COMP2}.itp based on excel or csv input
    For pure components, excel csv files should be in format:
    comp, mass, charge, lambda_r, lambda_a, epsilon, sigma
    For comp-comp interactions, excel and csv file should be in the format:
    comp1, comp2, lambda_r, lambda_a, epsilon, sigma
    '''
    # Ensure file reading is correct
    if (type(purefile) is str) and (type(nbfile) is str):
        if(purefile[-5:].lower() == ".xlsx"):
            puredf = pd.read_excel(purefile)
        elif (purefile[-4:].lower() == ".csv"):
            puredf = pd.read_csv(purefile)
        else:
            raise Exception("Unknown file type for pure components, only .xlsx or .csv accepted")
        if(nbfile[-5:].lower() == ".xlsx"):
            nbdf = pd.read_excel(nbfile)
        elif (purefile[-4:].lower() == ".csv"):
            nbdf = pd.read_csv(nbfile)
        else:
            print("No non-bonded interactions table found for different compounds in generating nb_tables")
            nbdf = pd.DataFrame()
    else:
        raise Exception("Unknown variable types for pure compounds or non-bonded interactions files, please use str")
    
    # No. of rows generated depends on cutoff, generate till cutoff+1
    nbins=int((cutoff+1)/delr)+1

    # Pure component table generation
    for i in range(len(puredf)):
        comp = puredf.iloc[i,0]
        l_r = np.asscalar(puredf.iloc[i,3])*1.0
        l_a = np.asscalar(puredf.iloc[i,4])*1.0

        c_mie = l_r/(l_r-l_a) * pow(l_r/l_a, l_a/(l_r-l_a))

        fout = open("table_{}_{}.xvg".format(comp,comp),"w")
        r=0.
        fout.write("# Production of pure component non-bonded intermolecular interactions\n")
        fout.write("# Using Mie potential of lambda_r = {:<7.5f} and lambda_a = {:<7.5f}\n".format(l_r,l_a))
        fout.write("# Revised from CHerdes and Lynn Zheng\n")
        fout.write("# Calculate C and A (or V(6) W(12)) parameters in ffnonbonded.itp using:\n")
        fout.write("# C = {:<7.5f} x epsilon x sigma ^ {:<7.5f}, A = {:<7.5f} x epsilon x sigma ^ {:<7.5f}\n".format(c_mie,l_r,c_mie,l_a))

        for j in range(nbins):
            r = delr * j
            if r == 0:
                fout.write("{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,0.0,0.0,0.0,0.0,0.0,0.0))
            elif l_r/(pow(r,l_r+1)) > 1e27:
                fout.write("{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,0.0,0.0,0.0,0.0,0.0,0.0))
            else:
                # Format provided as: f(r), -f'(r), g(r), -g'(r), h(r), -h'(r)
                f = 1 / r
                fp = 1 / (pow(r,2))
                g = -1 / (pow(r,l_a))
                gp = -l_a / (pow(r,l_a+1))
                h = 1 / (pow(r,l_r))
                hp = l_r / (pow(r,l_r+1))

                fout.write("{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,f,fp,g,gp,h,hp))
        fout.close()

    # Comp-comp interactions
    if len(nbdf) == 0:
        return
    for i in range(len(puredf)):
        comp1 = nbdf.iloc[i,0]
        comp2 = nbdf.iloc[i,1]
        l_r = np.asscalar(nbdf.iloc[i,2])*1.0
        l_a = np.asscalar(nbdf.iloc[i,3])*1.0

        c_mie = l_r/(l_r-l_a) * pow(l_r/l_a, l_a/(l_r-l_a))

        fout = open("table_{}_{}.xvg".format(comp1,comp2),"w")
        r=0.
        fout.write("# Production of non-bonded intermolecular interactions between different components\n")
        fout.write("# Using Mie potential of lambda_r = {:<7.5f} and lambda_a = {:<7.5f}\n".format(l_r,l_a))
        fout.write("# Revised from CHerdes and Lynn Zheng\n")
        fout.write("# Calculate C and A (or V(6) W(12)) parameters in ffnonbonded.itp using:\n")
        fout.write("# C = {:<7.5f} x epsilon x sigma ^ {:<7.5f}, A = {:<7.5f} x epsilon x sigma ^ {:<7.5f}\n".format(c_mie,l_r,c_mie,l_a))

        for j in range(nbins):
            r = delr * j
            if r == 0:
                fout.write("{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,0.0,0.0,0.0,0.0,0.0,0.0))
            elif l_r/(pow(r,l_r+1)) > 1e27:
                fout.write("{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,0.0,0.0,0.0,0.0,0.0,0.0))
            else:
                # Format provided as: f(r), -f'(r), g(r), -g'(r), h(r), -h'(r)
                f = 1 / r
                fp = 1 / (pow(r,2))
                g = -1 / (pow(r,l_a))
                gp = -l_a / (pow(r,l_a+1))
                h = 1 / (pow(r,l_r))
                hp = l_r / (pow(r,l_r+1))

                fout.write("{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,f,fp,g,gp,h,hp))
        fout.close()

def main():
    writeffnb("pure_comp.xlsx","nb_table.xlsx")
    gen_nb_table("pure_comp.xlsx","nb_table.xlsx")
    return

if __name__ == '__main__':
    main()
else:
    print("Loading functions from genNBTable.py")