import math
import numpy as np 

from plotxvg import Graph, Plot

def plotmie(sigma, epsilon, lambda_rep=12, lambda_att=6, nint=500, cutoff=5, color="r"):
    x = np.linspace(0., cutoff*sigma, nint)
    
    r = lambda_rep
    a = lambda_att
    c_mie = r/(r-a) * pow(r/a, a/(r-a))

    srratio = sigma / x[1:]
    y = c_mie * epsilon * (srratio**r - srratio**a)

    return Plot(x[1:], y, label="Plot of Mie potential ({:>3.1f},{:>3.1f})".format(r,a), color=color)

def main():
    p = [plotmie(3.49,258.8,lambda_rep=11.58, color='r'), plotmie(5.293,658.17,lambda_rep=32., color='b'), plotmie(4.629,414.9,lambda_rep=19.61, color='m')]
    g = Graph(legends=True)
    g.add_plots(*p)
    g.ylim(-1000,4000)
    g.xlim(0,20)
    g.add_hline(0)
    g.set_xlabels("r (A)")
    g.set_ylabels("u(r)/kb (K)")
    g.set_titles("Mie Potential Plot")
    g.draw(savefig="test.png")

if __name__ == '__main__':
    main()

