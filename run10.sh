runcase_1()
{
    gmx grompp -f nvt.mdp -c nvt_eq.gro -n index.ndx -p topol.top -o $1.tpr
    gmx mdrun -table table.xvg -v -deffnm $1
    # read -p "Press any key to continue... " -n1 -s
}

runs="1 2 3 4 5 6 7 8 9 10"
for run in $runs; do
    runcase_1 nvt_trj${run}
done

runcase_2()
{
    echo 19 | gmx energy -f nvt_trj$1.edr -o pressure_xy_$1.xvg
    echo 20 | gmx energy -f nvt_trj$1.edr -o pressure_xz_$1.xvg
    echo 23 | gmx energy -f nvt_trj$1.edr -o pressure_yz_$1.xvg
}

runs="1 2 3 4 5 6 7 8 9 10"
for run in $runs; do
    runcase_2 ${run}
done

