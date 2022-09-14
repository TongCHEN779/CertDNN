import pdb
import numpy as np
import sympy as sp
import torch
import networkx as nx

from itertools import combinations, product
from collections import defaultdict
from joblib import Parallel, delayed
from numpy import linalg as la
from scipy.special import comb
from tqdm import tqdm
from timeit import default_timer as timer
from mosek.fusion import (Domain, Model, Matrix, ObjectiveSense, Expr)


import ast, collections, pickle
import os

class KrivineOptimizer:
    def __init__(self):
        """
        Polynomial optimizer based on the Krivine/Stengle/Vasilescu
        Positivstellensatz. Optimization boils down to a linear program.
        """
        pass

    @classmethod
    def maximize(
            cls, f, g, deg=2, start_indices=None, layer_config=None, decomp='full', sparsity=-1, weights=None, solver='gurobi', n_jobs=-1, name='', reload=False, use_filename = ""):
        """
        Maximize the polynomial f subject to the constraint 0 <= g <= 1
        using a level of the LP hierarchy

        Args:
            f (dictionary): dictionary of coefficients of the polynomial
                to be maximized. Keys are monomials (sympy) and values
                are float.
            g (list): list of sympy expressions corresponding to the
                constraints 0 <= g_i(x) <= 1
            variables (iterable): variables (sympy) of the polynomial
            deg (int): degree of the LP relaxation
            solver (str): name of solver to use.
            certificate (str): name of certificate to use
            n_jobs (int): number of workers for parallel execution
            name (str): id for the problem
        """
        # global G_g
        # G_g = g
        
        filename = use_filename
        if use_filename == "":
            filename = "matrices_A/"
            for i in layer_config[:-1]:
                filename += "{}x".format(i)
            filename = filename[:-1]
            filename += "_deg={}".format(deg)
            if sparsity > 0:
                filename += "_s={}".format(sparsity)
            if decomp == 'full':
                filename += "_full"
            filename += ".txt"
        
        if os.path.isfile(filename) and not reload:
            with open(filename, 'rb') as handle:
                d = pickle.loads(handle.read())
            n = max([max(x) for [x,i] in d.values()])+ 1
        else:
            certificate, n = cls._full_certificate(g, deg=deg, start_indices=start_indices, decomp=decomp, n_jobs=n_jobs, weights=weights)
            d = dict(certificate)
            #return d
            #with open(filename, 'wb') as file:
            #    pickle.dump(d, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        if solver == 'gurobi':
            return cls._maximize_gurobi(f, d, n, name=name)
        elif solver == 'mosek':
            return cls._maximize_mosek(f, d, n, name=name)
        else:
            raise NotImplementedError

    @classmethod
    def _maximize_mosek(cls, f, certificate, n, name=''):
        m = Model(name + ' Krivine-LP')
        #m.Params.OutputFlag = 1 #enables solver output
        #m.Params.Method = 2 #barrier method
        #gr_vars = m.variable('y', n, Domain.greaterThan(0.))
        gr_vars = [m.variable('y_' + str(i), Domain.greaterThan(0.)) for i in range(n)]
        tp, tm = m.variable('tp', Domain.greaterThan(0.)), m.variable('tm', Domain.greaterThan(0.))
        m.objective(ObjectiveSense.Minimize, Expr.sub(tp, tm))

        keys = set()
        keys.update(certificate.keys())
#        for monomial, value in f.items():
#            if monomial not in keys:
#                raise ValueError('infeasible LP')
            
        for mon, (indices, coeffs) in tqdm(certificate.items(), desc='constraints'):
            expr = 0
            k = 0
            for i in indices:
                expr = Expr.add(expr, Expr.mul(gr_vars[i], float(coeffs[k])))
                k += 1
            #expr = np.dot([gr_vars[i] for i in indices], coeffs)
            if mon == 1:
                m.constraint(Expr.sub(expr, Expr.sub(tp, Expr.add(tm, float(f[mon])))), Domain.equalsTo(0.))
            else:
                m.constraint(Expr.add(expr, float(f[mon])), Domain.equalsTo(0.))

        m.setSolverParam("intpntCoTolRelGap", 1.0e-12)
        start = timer()
        m.solve()
        end = timer()
        print('time elapsed: ', end - start)
        #objVal = m.dualObjValue()
        #status = m.getDualSolutionStatus()
        return m
    
    @classmethod
    def _maximize_gurobi(cls, f, certificate, n, name=''):
        import gurobipy as gp
        m = gp.Model(name + ' Krivine-LP') #needs license
        m.Params.OutputFlag = 1
        m.Params.Method = 2
        gr_vars = np.array([m.addVar(
            lb=0.0, name='y_' + str(i)) for i in range(n)])
        tp, tm = m.addVar(lb=0.0, name='tp'), m.addVar(lb=0.0, name='tm')
        m.setObjective(tp - tm, sense=gp.GRB.MINIMIZE)
        m.update()

        keys = set()
        keys.update(certificate.keys())
#        for monomial, value in f.items():
#            if monomial not in keys:
#                raise ValueError('infeasible LP')
            
        for mon, (indices, coeffs) in tqdm(
                certificate.items(), desc='constraints'):
            expr = np.dot(gr_vars[indices], coeffs)
            if mon == 1:
                m.addConstr(expr == tp - tm - f[mon])
            else:
                m.addConstr(expr == - f[mon])

        m.update()
        m.optimize()
        return m

    @classmethod
    def _h_poly(cls, idx, coefs, g):
        """
        Polynomials composing the certificate

        Args:
            idx (np.array): indices of nonzero exponents (alpha_i, beta_i)
            coefs (np.array): coefficients alpha_i, beta_i corresponding to
                the nonzero elements of the partition
        """
        poly = sp.sympify(1)
        for i in range(len(idx)):
            alpha, beta = coefs[i]
            poly *= g[idx[i]] ** alpha * (1 - g[idx[i]]) ** beta

        coeffs = poly.expand().as_coefficients_dict()
        return coeffs

    @classmethod
    def _alpha_beta_combinations_full(cls,var):
        x_list = []
        for v in var:
            if len(x_list) == 0:
                x_list += [(np.array([v]), np.array([[1,0]]))] + [(np.array([v]), np.array([[0,1]]))]
            else:
                x_list = [(np.append(i,[v]), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_list] \
                    + [(np.append(i,[v]), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_list]
        return x_list

    @classmethod
    def _alpha_beta_combinations(cls, var, deg, emptyset=False):
        x_list = []
        if emptyset:
            x_list = [(np.array([]), np.array([]))]
        for v in list(combinations(var,deg)):
            x_list += cls._alpha_beta_combinations_full(v)
        return x_list

    @classmethod
    def _alpha_beta(cls, len_g, deg, offset=0, var = []):
        for x in combinations(
                range(1, 2 * len_g + deg + 1), 2 * len_g):
            x = np.array(x)
            x[1:] -= x[:-1]
            x -= 1
            x = x.reshape(len_g, 2)
            colsums = x.sum(axis=1)
            nonzero = np.nonzero(colsums)[0]
            if x[nonzero].sum() == deg:
                if len(var) > 0:
                    yield var[nonzero], x[nonzero]
                else:
                    yield nonzero + offset, x[nonzero]

    @classmethod
    def _alpha_beta_two_layers(cls, len_g, start_indices, deg):
        x_full = list(cls._alpha_beta(len_g-start_indices[1], deg-1, offset=start_indices[1]))
        x_list = list(cls._alpha_beta(len_g-start_indices[1], deg, offset=start_indices[1])) #[(np.array([]), np.array([]))]
        for var in range(start_indices[1]):
            x_list += [(np.append([var],i), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_full] + [(np.append([var],i), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_full]
        return x_list

    @classmethod
    def _alpha_beta_two_layers_sparse(cls, len_g, start_indices, deg, weights):
        #assert(len_g == 2*start_indices[1])
        # Can optimize using x_full
        #x_full = list(cls._alpha_beta(len_g-start_indices[1], deg-1, offset=start_indices[1]))
        x_list = [(np.array([]), np.array([]))]
        for var in range(start_indices[1]):
            var_right = []
            for i,w in enumerate(weights[0][:,var]):
                if w != 0:
                    var_right.append(start_indices[1]+i)
            x_right = list(cls._alpha_beta(len(var_right), deg-1, var=np.array(var_right)))
            x_list += [(np.append([var],i), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_right] + [(np.append([var],i), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_right]
        return x_list

    @classmethod
    def _alpha_beta_multi_layers(cls, len_g, start_indices, deg):
        print("Compute decomp for multi layers with degree {}".format(deg))
        var_right = list(range(start_indices[1], len_g))
        x_right = cls._alpha_beta_combinations(var_right, deg-1)
        x_list = [(np.array([]), np.array([]))]
        for v in range(start_indices[1]):
            x_list += ([(np.append([v],i), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_right] \
                       + [(np.append([v],i), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_right])
        x_list += cls._alpha_beta_combinations(var_right, deg)
        return x_list
    
    @classmethod
    def _alpha_beta_multi_layers_super_sparse(cls, len_g, start_indices, weights):
        print("Compute decomp for super sparse multi layers with degree {}".format(deg))
        if len(start_indices) == 0:
            return [(np.array([]), np.array([]))]
        x_list = [(np.array([]), np.array([]))]
        x_right = cls._alpha_beta_multi_layers_sparse(len_g, start_indices[1:], weights[1:])
        x_right = x_right[1:]
        if len(start_indices) == 1:
            final_var = len_g-1
        else:
            final_var = start_indices[1]-1
        for var in range(start_indices[0],final_var+1):
            if len(x_right) == 0:
                x_list += [(np.array([var]), np.array([[1,0]]))] + [(np.array([var]), np.array([[0,1]]))]
            else:
                x_right_connected = [(i,j) for (i,j) in x_right if abs(weights[0][i[0]-start_indices[1], var-start_indices[0]]) > 1e-7]
                x_list += [(np.append([var],i), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_right_connected] \
                    + [(np.append([var],i), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_right_connected]
        return x_list

    @classmethod
    def _alpha_beta_multi_layers_sparse(cls, len_g, start_indices, deg, weights):
        assert(deg >= len(weights))
        def variables_connected_to(v, len_g, start_indices, weights):
            # returns all variables connected to v from 1st layer in the graph given by weights
            if len(weights) == 1:
                return []
            if len(start_indices) == 2:
                final_var = len_g-1
            else:
                final_var = start_indices[2]-1
            variables = set([])
            for i, var_right in enumerate(range(start_indices[1], final_var+1)):
                if abs(weights[0][var_right - start_indices[1],v - start_indices[0]]) > 1e-8:
                    variables.add(var_right)
                    variables = variables.union(variables_connected_to(var_right, len_g, start_indices[1:], weights[1:]))
            return variables
        x_list = [(np.array([]), np.array([]))]
        var_right_list = set()
        G = nx.Graph()
        for v in range(start_indices[1]):
            var_right = variables_connected_to(v, len_g, start_indices, weights)
            var_right_list.add(tuple(var_right))
            x_right = cls._alpha_beta_combinations(var_right, deg-1)
            for (i,j) in combinations(var_right, 2):
                G.add_edge(i,j)
            x_list += ([(np.append([v],i), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_right] \
                     + [(np.append([v],i), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_right])

        for v in G.nodes():
            connected_vars = G.neighbors(v)
            if len(connected_vars) >= deg-1:
                x_conn = cls._alpha_beta_combinations(connected_vars, deg-1)
                x_list += [(np.append([v],i), np.append([[1,0]], np.array(j), axis=0)) for (i,j) in x_conn] \
                        + [(np.append([v],i), np.append([[0,1]], np.array(j), axis=0)) for (i,j) in x_conn]
            G.remove_node(v)
        return x_list
    
    @classmethod
    def _alpha_beta_sparse(cls, len_g, start_indices):
        end_indices = np.zeros(len(start_indices))
        end_indices[:-1] = start_indices[1:]
        end_indices[-1] = int(len_g)
        end_indices = [int(i) for i in end_indices]
        lists = (
                range(start_indices[i], end_indices[i])
                for i in range(len(start_indices)))
        x_list = [(np.array([]), np.array([]))]

        for i in product(*lists):
            for j in product([[1, 0], [0, 1]], repeat=2):
                x = np.zeros([len_g, len(start_indices)], dtype=int)
                x[i[0], :] = j[0]
                x[i[1], :] = j[1]
                colsums = x.sum(axis=1)
                nonzero = np.nonzero(colsums)[0]
                x_list.append((nonzero, x[nonzero]))

        return x_list

    @classmethod
    def _full_certificate(cls, g, deg=2, start_indices=None, decomp='full', n_jobs=-1, weights=None):
        certificate = defaultdict(lambda: [[], []])
        multiproc = Parallel(
                n_jobs=n_jobs, backend='multiprocessing', verbose=0)
        fn = delayed(cls._h_poly)
        if decomp == 'full':
            n_hpoly = int(comb(2 * len(g) + deg, deg))
            iter_ = tqdm(
                cls._alpha_beta(len(g), deg), total=n_hpoly, desc='h_poly')
        elif decomp == 'two layers sparse':
            iter_ = tqdm(
                         #cls._alpha_beta_two_layers(len(g), start_indices, deg),
                         cls._alpha_beta_two_layers_sparse(len(g), start_indices, deg, weights),
                         desc='h_poly')
        elif decomp == 'multi layers':
            iter_ = tqdm(
                         #cls._alpha_beta_two_layers(len(g), start_indices, deg),
                         #cls._alpha_beta_two_layers_sparse(len(g), start_indices, deg, weights),
                    cls._alpha_beta_multi_layers_sparse(len(g), start_indices, deg, weights),
                    desc='h_poly')
        else:
            raise("Unknown decomposition")
        print("Number of LP constraints: {}".format(len(iter_)))
        result = multiproc(fn(idx, coefs, g) for idx, coefs in iter_)

        for i, h_poly in enumerate(result):
            for k, v in h_poly.items():
                certificate[k][0].append(i)
                certificate[k][1].append(v)
        return certificate, len(result)


class FullyConnected:
    def __init__(
            self, weights, biases=None, lb=None, ub=None):
        self._weights, self._biases = weights, biases
        self.layer_sizes = np.array([w.shape[1] for w in self.weights])
        self.n_layers = len(self.layer_sizes)
        self.start_indices = np.zeros(self.n_layers, dtype=int)
        self.start_indices[1:] = np.cumsum(self.layer_sizes, dtype=int)[:-1]
        self.vars_layer = [None] * self.n_layers
        for i, (start, length) in enumerate(
                zip(self.start_indices, self.layer_sizes)):
            self.vars_layer[i] = np.array(
                    sp.symbols('x_{}:{}'.format(start, start + length)))
        self.variables = np.array(
                sp.symbols('x_0:{}'.format(self.layer_sizes.sum())))
        self.lb, self.ub = lb, ub

        if lb is not None and ub is not None:
            self.grad_poly_bounds = self._compute_grad_poly_bounds()
        else:
            self.grad_poly_bounds = None

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def n_hidden_layers(self):
        return self.n_layers - 1

    @property
    def grad_poly(self):
        try:
            return self._grad_poly
        except AttributeError:
            self._grad_poly = self._compute_grad_poly()
        return self._grad_poly

    def _compute_grad_poly(self):
        r"""
        Compute the symbolic gradient polynomial of the network
        """
        var_indices = [range(n_i) for n_i in self.layer_sizes]
        total = 1.
        for n_i in self.layer_sizes:
            total *= n_i
        terms = defaultdict(lambda: 0.0)

        for idx in tqdm(
                product(*var_indices), total=total, desc='grad_poly'):
            coeff = self.weights[-1][0, idx[-1]]
            monomial = self.vars_layer[0][idx[0]]
            for i in range(self.n_layers - 1):
                coeff *= self.weights[i][idx[i+1], idx[i]]
                monomial *= self.vars_layer[i+1][idx[i+1]]
            terms[monomial] = coeff

        return terms

    def _compute_grad_poly_bounds(self):  # TODO
        r"""
        Compute and store the bounds for the variables of the
        gradient polynomial
        """
        pass

    def krivine_constr(self, p=1):
        r"""
        Compute the symbolic constraints that define the gradient
        polynomial. The constraints have the form 0 <= g(x) <= 1.
        """
        if p == "inf":
            norm_constr = (self.vars_layer[0] + 1) / 2
#            norm_constr = self.vars_layer[0]
        elif p == 2:
            norm_constr = np.sum(self.vars_layer[0]**2)
        else:
            raise NotImplementedError

        var_idx = self.start_indices[1]
        if not self.lb:
            var_constr = self.variables[var_idx:]
        else:
            raise NotImplementedError
        
        if p == "inf":
            return np.concatenate([norm_constr, var_constr])
        elif p == 2:
            return np.concatenate([[norm_constr], var_constr])

    def sparse_krivine_constr(self, p=1):  # Why do we need this?
        r"""
        Compute the symbolic constraints that define the gradient polynomial in
        the sparse case. The constraints have the form 0 <= g(x) <= 1.
        """
        if self.n_hidden_layers > 1:
            raise NotImplementedError
        if p == 1:
            norm_constr = (self.vars_layer[0] + 1) / 2
        else:
            raise NotImplementedError

        var_idx = self.start_indices[1]
        if not self.lb:
            var_constr = self.variables[var_idx:]
        else:
            raise NotImplementedError

        pattern = onelayer_sparsity_pattern(self.weights[0])
        g = [None] * len(pattern)

        for i, pat in enumerate(pattern):
            g[i] = np.append(norm_constr[pat[:-1]], var_constr[pat[-1]])

        return g


def upper_bound_product(weights, p=1): #UBP
    r"""
    Computes the naive upper bound of the Lipschitz
    constant of a neural network using the function composition
    lemma. The metrics for all hidden layers is given by the
    $l_\infty$ norm.
    """
    bound = 1.0
    for weight in weights:
        norms = la.norm(weight, ord=p, axis=1)
        bound *= np.amax(norms)

    return bound


def lower_bound_product(weights, p=1): #LBP
    a = weights[0]
    for weight in weights[1:]:
        a = weight @ a

    norms = la.norm(a, ord=p, axis=1)
    return np.amax(norms)


def lower_bound_sampling(f, d, p=1, n=10000): #LBS global
    return torch.max(norm_gradients(f, d, n, p=p)).item()


def norm_gradients(f, d, n=10000, p=1):
    x = 10 * torch.rand(n, d)
    x.requires_grad = True
    f(x).sum().backward()
    norms = torch.norm(x.grad.data, p=p, dim=1)
    return norms

def lower_bound_sampling_local(f, d, x00, eps=0.1, p=1, n=10000): #LBS local
    return torch.max(norm_gradients_local(f, d, x00, eps=eps, n=n, p=p)).item()


def norm_gradients_local(f, d, x00, eps=0.1, n=10000, p=1):
    x = 2 * eps * torch.rand(n, d) + x00 - eps
    x.requires_grad = True
    f(x).sum().backward()
    norms = torch.norm(x.grad.data, p=p, dim=1)
    return norms


def elu(x):
    r"""ELU activation function"""
    return x * (x > 0) + (np.exp(x) - 1.) * (x <= 0)


def d_elu(x):
    r"""Derivative of ELU activation function"""
    return (x > 0) * 1. + (x <= 0.) * np.exp(x)



def bounds(weights, biases, lb, ub):
    r"""
    Computes all the preactivation, activation and derivative
    of the activation function bounds for each layer. Currently
    the only activation used is the ELU.

    As convention, the activation bounds of the 0-th layer correspond
    to the bounds on the input to the network, and their corresponding
    preactivation and activation derivative bound are None.
    """
    bounds = {
            'act': [None] * (len(weights)),
            'd_act': [None] * (len(weights)),
            'pre': [None] * (len(weights)),
    }
    bounds['act'][0] = [lb, ub]

    for i in range(len(weights) - 1):
        bounds['pre'][i+1] = _preactivation_bound(
                A=weights[i], b=biases[i],
                lb=bounds['act'][i][0], ub=bounds['act'][i][1])
        bounds['act'][i+1] = [
                elu(bounds['pre'][i+1][0]),
                elu(bounds['pre'][i+1][1])]
        bounds['d_act'][i+1] = [
                d_elu(bounds['pre'][i+1][0]),
                d_elu(bounds['pre'][i+1][1])]

    return bounds


def _preactivation_bound(A, b, lb, ub):
    r"""
    Computes the preactivation bounds of the following
    layer, given upper and lower bounds of the current input
    """
    M = np.repeat((ub - lb).reshape(1, -1), A.shape[0], axis=0)
    x = A @ ((ub - lb) / 2 + lb)
    y = la.norm(A * M, ord=1, axis=1) / 2
    return x - y + b, x + y + b


def onelayer_sparsity_pattern(W):
    sets = [None] * W.shape[0]

    for i in range(W.shape[0]):
        t_vars_idx = np.nonzero(W[i])[0]
        sets[i] = np.append(t_vars_idx, i)

    return sets


def _sdp_cost_old(w, v):
    d = w.shape[1]
    n = w.shape[0] + d + 1
    #a = np.diag(v[0]) @ w
    #b = a @ np.ones(d)
    #m = np.zeros([n, n])
    #m[1+d:, 0] = b
    #m[1+d:, 1:1+d] = a
    
    a = np.diag(v[0]) @ w
    b = np.ones(w.shape[0]).T @ a
    m = np.zeros([n, n])
    m[0,1:1+d] = b
    m[1+d:, 1:1+d] = a
    return m + m.T

def _sdp_cost(weights):
    n_hidden_layers = len(weights) - 1
    if n_hidden_layers == 1:
        w = weights[0]
        v = weights[1]
        d = w.shape[1]
        n = w.shape[0] + d + 1
        
        a = np.diag(v[0]) @ w
        b = np.ones(w.shape[0]).T @ a
        
        m = np.zeros([n, n])
        m[0,1:1+d] = b
        m[1+d:, 1:1+d] = a
    elif n_hidden_layers == 2:
        W0 = weights[0]
        W1 = weights[1]
        v = weights[2]
        d = W0.shape[1]
        n1 = W1.shape[1]
        n2 = v.shape[1]
        
        A = np.zeros((n1, n1*n2))
        for i in range(n1):
            A[i,i*n2:(i+1)*n2] = np.multiply(W1.T[i,:], v[0])

        m = np.zeros([1 + d + n1 + n2 + n1*n2 + 1, 1 + d + n1 + n2 + n1*n2 + 1])
        m[0, 1:1+d] = np.ones(v.shape[1]).T @ np.diag(v[0]) @ W1 @ W0
        m[1:d+1, 1+d:1+d+n1] = W0.T @ np.diag(W1.T @ v[0])
        m[1:d+1, 1+d+n1:1+d+n1+n2] = W0.T @ W1.T @ np.diag(v[0])
        m[1:d+1, 1+d+n1+n2:1+d+n1+n2+n1*n2] = W0.T @ A
    return m + m.T
