import sys
from operator import itemgetter

def auc(l):
    l = sorted(l, key=itemgetter(2), reverse=True)

    total_nonclk = 0
    total_clk = 0
    total_area = 0.0
    for show, clk, pctr in l:
        nonclk = show - clk
        area = 0.5*(total_clk + total_clk + clk)*nonclk
        total_area += area
        total_clk += clk
        total_nonclk += nonclk
    
    if total_clk * total_nonclk == 0:
        return 0.5
    return total_area / (total_clk * total_nonclk)


if __name__ == '__main__':
    print auc([(1, 0, 0), (1, 1, 1)])
