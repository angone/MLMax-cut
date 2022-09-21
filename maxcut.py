import networkit as nw
from dwave_qbsolv import QBSolv
import matplotlib.pyplot as plt
import random
from datetime import datetime
import argparse
import time
import numpy as np
import networkx as nx
from qiskit import BasicAer
from qiskit.algorithms import QAOA
import qiskit.aqua.components.optimizers as optimizers
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.algorithms.optimizers import COBYLA
from QAOAKit.utils import (
    precompute_energies,
    maxcut_obj,
    get_adjacency_matrix,
    qaoa_maxcut_energy
)
from QAOAKit import (
    beta_to_qaoa_format,
    gamma_to_qaoa_format
)
from QAOAKit.parameter_optimization import get_median_pre_trained_kde
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from functools import partial
from QAOAKit.qaoa import get_maxcut_qaoa_circuit
import operator
import pyomo.environ as pyo
import scipy

median, kde = get_median_pre_trained_kde(3)

parser = argparse.ArgumentParser()
parser.add_argument("-gname", type = str, default = "None", help = "graph file")
parser.add_argument("-method", type = str, default = "None", help = "ref method")
parser.add_argument("-spsize", type = int, default = 20, help = "size of subproblems")
parser.add_argument("-solver", type = str, default = "qbsolv", help = "qubo slver")
parser.add_argument("-optimizer", type = str, default = "COBYLA", help = "qaoa optimizer")
parser.add_argument("-p", type = int, default = 1, help = "p value of qaoa")
parser.add_argument("-gformat", type = str, default = "alist", help = "graph format")
parser.add_argument("-nomultilvl", type = bool, default = False, help = "Use/Dont Use Multilevel")

#median, kde = get_median_pre_trained_kde(3)
args = parser.parse_args()
useml = args.nomultilvl
method = args.method
gname = args.gname
spsize = args.spsize
solver = args.solver
optimizer = args.optimizer
p = args.p
gformat = args.gformat

mapGain = {}
S = []
T = []
gainTime = 0
pairTime = 0
buildSpTime = 0
solveTime = 0
qSolves = 0
cSolves = 0
ties = 0
minweight = 0
qct = 0

def readGraph():
    f = open(gname, "r")
    line = f.readline().split()
    n = int(eval(line[0]))
    G = nw.graph.Graph(n=0,weighted=False, directed=False)
    G.addNodes(n)
    cn = 0
    line = f.readline().split()
    while(line != []):
        for x in line:
            G.addEdge(cn, int(x)-1)
        cn += 1
        line = f.readline().split()
    G.removeMultiEdges()
    G.indexEdges()
    G.removeSelfLoops()
    return G


def readGraphEList():
    f = open(gname, "r")
    line = f.readline().split()
    n = int(line[0])
    G = nw.graph.Graph(n=n, weighted=True, directed = False)
    line = f.readline().split()
    while line != []:
        u = int(line[0])-1
        v =  int(line[1])-1
        if len(line) > 2:
            w = int(eval(line[2]))
        else:
            w = 1
        G.addEdge(u, v, w, addMissing=True)
        line = f.readline().split()
    G.removeMultiEdges()
    G.indexEdges()
    G.removeSelfLoops()
    return G
        
def get_exact_energy(G, p):
    obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    precomputed_energies = precompute_energies(obj, G.number_of_nodes())
    def f(theta):
        gamma = theta[:p]
        beta = theta[p:]
        return -qaoa_maxcut_energy(G, beta, gamma, precomputed_energies=precomputed_energies)
    return f



def pyomo(G, solver):
    if solver == "gurobi":
        opt = pyo.SolverFactory('gurobi')
    elif solver == "cbc":
        opt = pyo.SolverFactory('cbc')
    opt.options['TimeLimit'] = 10
    model = pyo.ConcreteModel()
    model.n = pyo.Param(default=G.numberOfNodes())
    model.x = pyo.Var(pyo.RangeSet(0,model.n-1), within=pyo.Binary)
    model.obj = pyo.Objective(expr = 0)
    model.c = pyo.Constraint(rule=model.x[2]<=1)
    for u,v in G.iterEdges():
        w = G.weight(u,v)
        model.obj.expr += (2 * w * model.x[u] * model.x[v])
        model.obj.expr += (-w * model.x[u]) + (-w * model.x[v])
    results = opt.solve(model)
    solution = {}
    for i in range(G.numberOfNodes()):
        solution[i] = model.x[i].value
        if solution[i] == None:
            solution[i] = 0
    return solution


def randSampleSolve(G):
    bestobj = 0
    bestsol = {}
    for _ in range(1024):
        sol = {}
        for i in G.iterNodes():
            sol[i] = random.randint(0, 1)
        obj = calc_obj(G, sol)
        if obj > bestobj:
            bestsol = sol.copy()
            bestobj = obj
    return bestsol

    
            
def build_qubo(G):
    Q = {}
    n = G.numberOfNodes()
    print('average degree: ' + str(d_w))
    for i in range(n):
        for j in range(i,n):
            Q[(i,j)] = 0            
    for i in range(n):
        for j in range(i+1, n):
            if G.hasEdge(i,j):
                weight = G.weight(i, j)
                Q[(i, i)] -= weight
                Q[(j,j)] -= weight
                Q[(i, j)] = 2*weight
    return Q 

def fineSolFromCoarseSol(cSol, ctfM):
    fSol = {}
    for (c, fl) in ctfM.items():
        for x in fl:
            fSol[x] = cSol[c]
    return fSol



def calc_obj(G, solution):
    obj = 0
    n = G.numberOfNodes()
    for u, v in G.iterEdges():
        obj += G.weight(u, v)*(2*solution[u]*solution[v] - solution[u] - solution[v])
    return -1 * obj


def swapGain(G, sol, u, v):
    gain = 0
    for x in G.iterNeighbors(u):
        if x != v:
            if sol[x] == sol[u]:
                gain += G.weight(u, x)
            else:
                gain -= G.weight(u, x)


    for x in G.iterNeighbors(v):
        if x != u:
            if sol[x] == sol[v]:
                gain += G.weight(v, x)
            else:
                gain -= G.weight(v, x)
    return gain


def buildGainMap(G, sol):
    start = time.perf_counter()
    global mapGain
    global gainTime
    if mapGain != {} or mapGain == None:
        mapGain = {}
    for x in range(G.numberOfNodes()):
        mapGain[x] = 0
    for u, v, w in G.iterEdgesWeights():
        if sol[u] == sol[v]:
            mapGain[u] += w
        else:
            mapGain[v] -= w
    end = time.perf_counter()
    gainTime += (end - start)

def buildParts(G, sol):
    S.clear()
    T.clear()
    for x in range(G.numberOfNodes()):
        if sol[x] == 0:
            S.append(x)
        elif sol[x] == 1:
            T.append(x)
    return

def pairwiseGain(G, sol):
    global pairTime
    pwGain = []
    start = time.perf_counter()
    for i in range(len(S)):
        for j in range(len(T)):
            pwGain.append((mapGain[S[i]] + mapGain[T[j]] + 2*G.weight(S[i], T[j]), S[i], T[j]))
    end = time.perf_counter()
    pairTime += (end - start)
    return sorted(pwGain)

    
def pairwiseSubProb(G, sol, sp_size):
    n = G.numberOfNodes()
    subprob = nw.graph.Graph(n=2*(1+sp_size), weighted = True, directed = False)
    pwGain = pairwiseGain(G, sol)
    pwGain.reverse()
    used = {}
    mapProbToSubProb = {}
    totalGain = 0
    ct = 0
    i = 0
    idx = 0
    while(ct < sp_size and i < len(pwGain)):
        v1 = pwGain[i][1]
        v2 = pwGain[i][2]
        if v1 not in used and v2 not in used:
            mapProbToSubProb[v1] = idx
            idx += 1
            mapProbToSubProb[v2] = idx
            idx += 1
            ct += 1
            totalGain += pwGain[i][0]
        i += 1

    for x in G.iterNodes():
        if x not in mapProbToSubProb.keys():
            if sol[x] == 0:
                mapProbToSubProb[x] = idx
            if sol[x] == 1:
                mapProbToSubProb[x] = idx + 1

    for u, v in G.iterEdges():
        spu = mapProbToSubProb[u]
        spv = mapProbToSubProb[v]
        if spu != spv:
            subprob.increaseWeight(spu, spv, G.weight(u,v))

    return (subprob, mapProbToSubProb, totalGain)

def calcDensity(G):
    e = G.numberOfEdges()
    n = G.numberOfNodes()
    return (2*e) / (n *(n-1))
            

def randSubProb(G, sp_size, sol):
    subprob = nw.graph.Graph(n=2*(1 + sp_size), weighted = True, directed = False )
    random.shuffle(S)
    random.shuffle(T)
    mapProbToSubProb = {}
    idx = 0
    for i in range(sp_size):
        mapProbToSubProb[S[i]] = idx
        idx += 1

    for i in range(sp_size):
        mapProbToSubProb[T[i]] = idx
        idx += 1


    for i in range(sp_size, len(S)):
        mapProbToSubProb[S[i]] = idx

        
    for i in range(sp_size, len(T)):
        mapProbToSubProb[T[i]] = idx+1

        
    n = G.numberOfNodes()
    for u, v in G.iterEdges():
        spu = mapProbToSubProb[u]
        spv = mapProbToSubProb[v]
        if spu != spv:
            subprob.increaseWeight(spu, spv, G.weight(u,v))

    return (subprob, mapProbToSubProb, 0)


def randPairSubProb(G, sol, sp_size):
    subprob = nw.graph.Graph(n=2*(1 + sp_size), weighted = True, directed = False )
    rpS = []
    rpT = []
    sampleSize = 10*sp_size
    if len(S) < sampleSize:
        rpS = S[0:len(S)]
    else:
        for _ in range(sampleSize):
            i = random.randint(0, len(S)-1)
            rpS.append(S[i])
    if len(T) < sampleSize:
        rpT = T[0:len(T)]
    else:
        for _ in range(sampleSize):
            i = random.randint(0, len(T)-1)
            rpT.append(T[i])
            
    pwGain = []
    for i in range(len(rpS)):
        for j in range(len(rpT)):
            pwGain.append((swapGain(G, sol, rpS[i], rpT[j]), rpS[i], rpT[j]))

    pwGain = sorted(pwGain)
    pwGain.reverse()
    
    used = {}
    mapProbToSubProb = {}
    totalGain = 0
    ct = 0
    i = 0
    idx = 0
    while(ct < sp_size and i < len(pwGain)):
        v1 = pwGain[i][1]
        v2 = pwGain[i][2]
        if v1 not in used and v2 not in used:
            mapProbToSubProb[v1] = idx
            idx += 1
            mapProbToSubProb[v2] = idx
            idx += 1
            ct += 1
            totalGain += pwGain[i][0]
        i += 1

    for x in G.iterNodes():
        if x not in mapProbToSubProb.keys():
            if sol[x] == 0:
                mapProbToSubProb[x] = idx
            if sol[x] == 1:
                mapProbToSubProb[x] = idx + 1

    for u, v in G.iterEdges():
        spu = mapProbToSubProb[u]
        spv = mapProbToSubProb[v]
        if spu != spv:
            subprob.increaseWeight(spu, spv, G.weight(u,v))
            
    return (subprob, mapProbToSubProb, totalGain)



def qaoa(G):
    n = G.numberOfNodes()
    mw = 0
    d_w = 0
    for x in G.iterNodes():
        d_w += G.degree(x)
    d_w = d_w / n
    for u,v,w in G.iterEdgesWeights():
        mw += abs(w)
    mw = mw / G.numberOfEdges()
    G = nw.nxadapter.nk2nx(G)
    w = nx.adjacency_matrix(G)

    problem = QuadraticProgram()
    _ = [problem.binary_var(f"x{i}") for i in range(n)]
    linear = w.dot(np.ones(n))
    quadratic = -w
    problem.maximize(linear=linear, quadratic=quadratic)
    c = [1]
    for _ in range(n-1):
        c.append(0)
    problem.linear_constraint(c, '==', 1)

    cobyla = COBYLA()
    backend = BasicAer.get_backend('qasm_simulator')
    beta = median[p:]
    if d_w <= 1:
        d_w = 1.001
    gamma = median[:p] * np.arctan(1/np.sqrt(d_w-1)) / mw
    init = np.concatenate((beta, gamma))
    qaoa = QAOA(optimizer=cobyla, reps=3, quantum_instance=backend, initial_point = init)
    algorithm=MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(problem)
    L = result.x
    i = 0
    res = {}
    for x in L:
        res[i] = x
        i += 1
    return res


def refine(G, sol, sp_size, obj, rmethod, spsolver, sp=None):
    global buildSpTime
    global solveTime
    start = time.perf_counter()
    if sp != None:
        subprob = sp
    elif rmethod == "spectral":
        subprob = spectralGainSubProb(G, sol, sp_size)
    elif rmethod == "pairwise":
        subprob = pairwiseSubProb(G, sol, sp_size)
    elif rmethod == "randpair":
        subprob = randPairSubProb(G, sol, sp_size)
    else:
        subprob = randSubProb(G, sp_size, sol)
    end = time.perf_counter()
    buildSpTime += (end - start)
    eGain = subprob[2]
    mapProbToSubProb = subprob[1]

    start = time.perf_counter()
    if spsolver == "qbsolv":
        Q = build_qubo(subprob[0])
        response = QBSolv().sample_qubo(Q)
        solution = response.samples()[0]
    
    elif spsolver == "qaoa":
        solution = qaoa(subprob[0])
    elif spsolver == "gurobi":
        solution = pyomo(subprob[0], spsolver)
    elif spsolver == "cbc":
        solution = pyomo(subprob[0], spsolver)
    elif spsolver == "sampling":
        solution = randSampleSolve(subprob[0])
    end = time.perf_counter()
    solveTime += (end - start)
    n = G.numberOfNodes()
    new_sol = {}
    changed = set()
    for i in range(n):
        new_sol[i] = solution[mapProbToSubProb[i]]
        if sol[i] != new_sol[i]:
            changed.add(i)
    new_obj = calc_obj(subprob[0], solution)
    rGain = new_obj - obj
          
    if new_obj > obj:
        for x in G.iterNodes():
            if sol[x] != new_sol[x]:
                if sol[x] == 0:
                        S.remove(x)
                        T.append(x)
                elif sol[x] == 1:
                        T.remove(x)
                        S.append(x)
        if rmethod == 'pairwise':
            for x in changed:
                for y in G.iterNeighbors(x):
                    if new_sol[x] == new_sol[y]:
                        mapGain[x] -= 2*G.weight(x, y)
                        mapGain[y] -= 2*G.weight(x, y)
                    if new_sol[x] != new_sol[y]:
                        mapGain[x] += 2*G.weight(x, y)
                        mapGain[y] += 2*G.weight(x, y)
        return (new_sol, new_obj, subprob)

    else:
        return (sol, obj, subprob)




def debug_graph(G):
    print("NUMBER OF NODES: " + str(G.numberOfNodes()))
    print("NUMBER OF EDGES: " + str(G.numberOfEdges()))
    esum = 0
    emin = 10 * G.numberOfEdges()
    emax = -10 * G.numberOfEdges()
    for u,v,w in G.iterEdgesWeights():
        esum += w
        if w < emin:
            emin = w
        if w > emax:
            emax = w
    print("AVERAGE EDGE WEIGHT: " + str(esum/G.numberOfEdges()))
    print("MIN EDGE WEIGHT: " + str(emin))
    print("MAX EDGE WEIGHT: " + str(emax))
    
def calc_imbalance(sol,n):
    s = 0
    t = 0
    for i in range(n):
        if sol[i] == 1:
            t += 1
        if sol[i] == 0:
            s += 1
    return abs(t - s) / ((s + t)/2)





def spectralCoarsening(G, GVs):
    edgesDropped = 0
    edgesAggregated = 0

    def column(m, i):
        return [row[i] for row in m]
    
    matrix = nw.algebraic.laplacianMatrix(G)

    w,v = scipy.sparse.linalg.eigsh(matrix,3, which="LM",maxiter=10000,tol=0.00001)
    orderlist = zip(w, range(0, len(w)))
    orderlist = sorted(orderlist)
    orderedW = column(orderlist, 0)
    orderedV = [v[:,i] for i in column(orderlist, 1)]
    eigvectors = (orderedW, orderedV)

    eigvec = eigvectors[1][1]
    print('calced eigenvector')
    orderedNodes = []
    for i in range(len(eigvec)):
        orderedNodes.append((eigvec[i], i))
    orderedNodes.sort()
    orderedNodes.reverse()
    n = len(orderedNodes)
    i = 0
    j = int(n/2)
    mapCoarseToFine = {}
    mapFineToCoarse = {}
    idx = 0
    newGVs = {}
    while i < int(n/2) and j < n:
        u = orderedNodes[i][1]
        v = orderedNodes[j][1]
        if G.weight(u, v) != 0:
            edgesDropped += 1
        for x in G.iterNeighbors(u):
            if G.weight(v, x) != 0:
                edgesAggregated += 1
                
        mapCoarseToFine[idx] = [u, v]
        mapFineToCoarse[u] = idx
        mapFineToCoarse[v] = idx
        newGVs[idx] = G.weight(u, v)
        if GVs != {}:
            newGVs[idx] += GVs[u]
            newGVs[idx] += GVs[v]
        idx += 1
        i += 1
        j += 1
    if n % 2 == 1:
        u = orderedNodes[j][1]
        mapCoarseToFine[idx] = [u]
        mapFineToCoarse[u] = idx
        newGVs[idx] = 0
        if GVs != {}:
            newGVs[idx] += GVs[u]
        idx += 1
        
        
    cG = nw.graph.Graph(n=idx, weighted=True, directed=False)
    for u,v in G.iterEdges():
        cu = mapFineToCoarse[u]
        cv = mapFineToCoarse[v]
        cG.increaseWeight(cu, cv, G.weight(u, v))
    cG.removeSelfLoops()
    cG.indexEdges()
    return (cG, mapCoarseToFine, newGVs)




def randInitialSolution(G):
    sol = {}
    for x in G.iterNodes():
        sol[x] = random.randint(0,1)
    return sol
       
    



def no_ML(G):
    solution = randInitialSolution(G)
    obj = 0
    new_solution = {}
    refinements = 0
    for j in range(G.numberOfNodes()):
            new_solution[j] = solution[j]
    obj = calc_obj(G, solution)
    buildParts(G, solution)
    ct = 0
    improvect = 0
    maximprove = 0
    minimprove = obj
    improveavg = 0
    
    while ct < 5:
        res = refine(G, solution, spsize, obj, method, solver)                
        refinements += 1
        solution = res[0]
        new_obj = res[1]
        if new_obj == obj:
            ct += 1
        else:
            ct = 0
            obj = new_obj
    return obj

def sparsify_graph(G, ratio):
    ffs = nw.sparsification.ForestFireSparsifier(0.1, ratio)
    sparse = ffs.getSparsifiedGraphOfSize(G, ratio)
    sparse.indexEdges()
    return sparse

def calc_density(G):
    e = G.numberOfEdges()
    n = G.numberOfNodes()
    return (2*e) / (n*(n-1))


def maxcut_solve(G):
    global S
    global T
    global cSolves
    global qSolves
    global ties
    global dbf
    refinements = 0
    print(gname)
    print(str(G))
    print(method)
    start = time.perf_counter()
    problem_graph = G

    density_cutoff = calc_density(G)
    density = density_cutoff
    if density > 0.4:
        G = sparsify_graph(G, 0.4)
        density_cutoff = calc_density(G)
        density = density_cutoff
    hierarchy = [G]
    hierarchy_map = []
    old = G.numberOfNodes()
    new = 0
    GVs = {}
    while(abs(new - old) > 2*spsize):
        old = G.numberOfNodes()
        if old <= 2*(1+spsize):
            break
        coarse= spectralCoarsening(G, GVs)    
        G = coarse[0]
        GVs = coarse[2]
        if calc_density(G) > density_cutoff:
            G = sparsify_graph(G, density_cutoff)
        hierarchy.append(G)
        hierarchy_map.append(coarse[1])
        new = G.numberOfNodes()
    end = time.perf_counter()
    
    hierarchy_map.reverse()
    hierarchy.reverse()
    solution = randInitialSolution(G)
    obj = 0
    for i in range(len(hierarchy_map)):
        fG = hierarchy[i+1]
        cG = hierarchy[i]
        fMap = hierarchy_map[i]
        print("\n\nLEVEL " + str(i))
        debug_graph(fG)
        print(len(solution))
        print("OBJECTIVE BEFORE REFINING: " + str(calc_obj(cG,solution)))
        new_solution = {}
                
        for j in range(cG.numberOfNodes()):
            for x in fMap[j]:
                new_solution[x] = solution[j]
        solution = new_solution
        obj = calc_obj(fG, solution)
        buildParts(fG, solution)
        ct = 0
        improvect = 0
        maximprove = 0
        minimprove = obj
        improveavg = 0
        print("\nREFINEMENTS FOR THIS LEVEL BELOW\n")
        while ct < 5:
            if method == 'pairwise':
                buildGainMap(fG, solution)
            if solver != 'hybrid':
                res = refine(fG, solution, spsize, obj, method, solver)                
                refinements += 1
                solution = res[0]
                new_obj = res[1]
            elif solver == "hybrid":
                os = solution.copy()
                ps = solution.copy()
                tS = S[:]
                tT = T[:]
                res = refine(fG, solution, spsize, obj, method, "qaoa")
                sp = res[2]
                solution = res[0]
                new_obj = res[1]
                tS2 = S[:]
                tT2 = T[:]
                S = tS
                T = tT
                res = refine(fG, os, spsize, obj, method, "sampling", sp)
                if res[1] > new_obj:
                        cSolves += 1
                        solution = res[0]
                        new_obj = res[1]
                else:
                    if res[1] == new_obj:
                        ties += 1                        
                    else:
                        qSolves += 1
                    S = tS2
                    T = tT2
                refinements += 1
            if new_obj == obj:
                ct += 1
            else:
                improvement = new_obj - obj
                improveavg += improvement
                improvect += 1
                if improvement > maximprove:
                    maximprove = improvement
                if improvement < minimprove:
                    minimprove = improvement
                ct = 0
            obj = new_obj
        if method != 'pairwise':
            buildGainMap(fG, solution)
            while True:
                if solver != 'hybrid':
                    res = refine(fG, solution, spsize, obj, method, solver)                
                    refinements += 1
                    solution = res[0]
                    new_obj = res[1]
                elif solver == "hybrid":
                    os = solution.copy()
                    tS = S[:]
                    tT = T[:]
                    res = refine(fG, solution, spsize, obj, method, "qaoa")
                    solution = res[0]
                    new_obj = res[1]
                    sp = res[2]
                    tS2 = S[:]
                    tT2 = T[:]
                    S = tS
                    T = tT
                    res = refine(fG, os, spsize, obj, method, "sampling", sp)
                    if res[1] > new_obj:
                        cSolves += 1
                        solution = res[0]
                        new_obj = res[1]
                    else:
                        if res[1] == new_obj:
                            ties += 1
                        else:
                            qSolves += 1
                        S = tS2
                        T = tT2
                    refinements += 1
                if new_obj == obj:
                    break
                else:
                    improvement = new_obj - obj
                    improveavg += improvement
                    improvect += 1
                    if improvement > maximprove:
                        maximprove = improvement
                        if improvement < minimprove:
                            minimprove = improvement
                obj = new_obj
        print("\nTOTAL REFINEMENTS: " + str(refinements))
        print("NUMBER OF IMPROVEMENTS: " + str(improvect))
        if improvect > 0:
            print("AVERAGE IMPROVEMENT: " + str(improveavg/improvect))
        print("MINIMUM IMPROVEMENT: " + str(minimprove))
        print("MAXIMUM IMPROVEMENT: " + str(maximprove))
        print("OBJECTIVE AFTER REFINEMENT: " + str(obj))
        print("IMBALANCE: " + str(calc_imbalance(solution, fG.numberOfNodes())))

    return calc_obj(problem_graph, solution)



if gformat == 'alist':
    G = readGraph()
elif gformat == 'elist':
    G = readGraphEList()



G = nw.components.ConnectedComponents.extractLargestConnectedComponent(G)

s = time.perf_counter()
if useml == False:
    obj = maxcut_solve(G)
elif useml == True:
    obj = no_ML(G)
e = time.perf_counter()
print("Found maximum value for " + str(gname) + " of " + str(obj) + " " + str(e-s) + "s")
print("Qct = " + str(qct))
if solver == 'hybrid':
    print("Quantum: " + str(qSolves) + " Classical: " + str(cSolves) + " Ties: " + str(ties))
