#######################################################
##### This is the Python code of LipOpt algorithm #####
#######################################################

import numpy as np
import polyopt as po
import torch
from polyopt import utils

layer_configs = [(10,10,1)]
repeats = 1
degs = [3]
sparsities = [6]

def lp_ub(weights, biases, layer_config, p=2, deg=3, typ="global", lower=None, upper=None, decomp='', \
          sparsity=-1, reload=False, filename="", solver='mosek'):
    fc = po.FullyConnected(weights, biases, typ=typ, lower=lower, upper=upper)
    f = fc.grad_poly
    g = fc.krivine_constr(p=p, typ=typ, lower=lower, upper=upper)
    start_indices = fc.start_indices
    m, t = po.KrivineOptimizer.maximize(f, g, deg=deg, start_indices=start_indices, \
                                        layer_config=layer_config, solver=solver, decomp=decomp, \
                                        sparsity=sparsity, weights=weights, n_jobs=-1, name='', \
                                        reload=reload, use_filename=filename)
    if solver == 'mosek': 
        return m.dualObjValue(), t
    elif solver == 'gurobi':
        return m.objVal, t
    else:
        raise NotImplementedError

def main():
    solver = 'gurobi'
    for layer_config in layer_configs:
        for sparsity in sparsities:
            for deg in degs:
                bounds = []
                times = []
                reload = False
                for i in range(repeats):
                    print("Run config {}, s = {}, deg = {}, trial = {}".format(layer_config, sparsity, deg, i+1))
                    np.random.seed(i)
                    torch.manual_seed(i)
                    network = utils.fc(layer_config, sparsity=sparsity)
                    weights, biases = utils.weights_from_pytorch(network)
                    bound, time = lp_ub(weights, biases, layer_config, p="inf", deg=deg, typ="global", \
                                                lower=None, upper=None, decomp="multi layers", \
                                                sparsity=sparsity, reload=reload, filename="", solver=solver)
                    reload = False
                    bounds.append(float(bound))
                    times.append(float(time))
                print("LipOpt-{} Total Bound: {}".format(deg, bounds))
                print("LipOpt-{} Average Bound: {}".format(deg, np.mean(bounds)))
                print("LipOpt-{} Total Time: {}".format(deg, times))
                print("LipOpt-{} Average Time: {}".format(deg, np.mean(times)))

if __name__ == '__main__':
    main()
