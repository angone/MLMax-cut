import networkit as nw
import random
from datetime import datetime
import argparse
import time
import numpy as np
import networkx as nx
import pyomo.environ as pyo
import scipy
from queue import SimpleQueue as Queue
import faulthandler
from scipy.optimize import minimize
from sklearn.neighbors import KDTree
from sortedcontainers import SortedKeyList
from node2vec import Node2Vec
import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)
faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument("-gname", type = str, default = "None", help = "graph file")
parser.add_argument("-spsize", type = int, default = 18, help = "size of subproblems")
parser.add_argument("-solver", type = str, default = "gurobi", help = "subproblem solver")
parser.add_argument("-optimizer", type = str, default = "COBYLA", help = "qaoa optimizer")
parser.add_argument("-gformat", type = str, default = "elist", help = "graph format")
parser.add_argument("-cycles", type = int, default = 1, help = "number of v-cycles")
parser.add_argument("-embed", type = str, default = 'cube', help = 'shape of embedding')
parser.add_argument("-coarseonly", type = int, default = 0)
parser.add_argument("-sfile", type = str, default = None)
parser.add_argument("-sparse", type = str, default = None)
parser.add_argument("-testsubprob", type = int, default = 0)
parser.add_argument("-eratio", type = float, default = 0.1)
args = parser.parse_args()
eratio = args.eratio
sparsification = args.sparse
testsubprob = args.testsubprob
embed = args.embed
gname = args.gname
spsize = args.spsize
solver = args.solver
optimizer = args.optimizer
gformat = args.gformat
cycles = args.cycles
coarseonly = args.coarseonly
sfile = args.sfile
fiedler_list =[]
gainmap = []
gainlist = None
change = set()
n2vm = None
subset = None
last_idx = 0
sortedEdges = []
lastembed = None
threshold = {}
nodes = []

if solver == 'qaoa':
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
    median, kde = get_median_pre_trained_kde(3)




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
    f = open("graphs/" + gname, "r")
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
        opt.options['TimeLimit'] = 1
    elif solver == "ipopt":
        opt = pyo.SolverFactory('ipopt')
        opt.options['max_cpu_time'] = 1
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
        if solution[i] < 0.5:
            solution[i] = 0
        else:
            solution[i] = 1
    return solution        
        


def buildEmbedObj(G, u, d, space):
    def obj(pos):
        o = 0
        for x in G.iterNeighbors(u):
            temp = 0
            for i in range(d):
                temp += (pos[i] - space[x][i])**2
            o += (np.sqrt(temp)*G.weight(u,x))
        return -1 * o
    return obj




def embedNodes(G, d):
    n = G.numberOfNodes()
    space = []
    for i in range(n):
        space.append([random.random() for _ in range(d)])
    for _ in range(3):
        for i in range(n):
            b = buildEmbedObj(G, i, d, space)
            bnds = ((0,1) for _ in range(d))
            p = [random.random() for _ in range(d)]
        def sphere(x):
            return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - 1
        cons = [{'type': 'ineq', 'fun': sphere}] if embed == 'sphere' else None
        res = minimize(b, p, bounds=bnds, tol=0.00001, constraints=cons)
        space[i] = list(res.x)
    return space



def embedDist(u, v, space, d):
    a = 0
    for i in range(d):
        a += (space[u][i] - space[v][i])**2
    return np.sqrt(a)

def embeddingMatching(G, d):
    global sortedEdges
    n = G.numberOfNodes()
    space = np.array(embedNodes(G, d))
    sortedEdges = []
    for u, v, w in G.iterEdgesWeights():
        sortedEdges.append((embedDist(u,v,space,d),(u,v,w)))
    sortedEdges.sort()
    tree = KDTree(space)
    ind = tree.query_radius(space, 0)
    matching = set()
    used = set()
    clusters = []
    singletons = []
    t = 0
    for x in ind:
        if x[0] in used:
            continue
        elif len(x) == 1:
            singletons.append(x[0])
        else:
            clusters.append(x)
            for y in x:
                used.add(y)
            t += len(x)
    
    for c in clusters:
        k = len(c)
        if k % 2 == 1:
            k = k - 1
            singletons.append(c[k])
            if c[k] in used:
                used.remove(c[k])
        for i in range(int(k/2)):
            matching.add((c[2*i], c[2*i + 1]))

    indices = []
    newspace = []
    R = -1
    k = len(singletons)
    if k % 2 == 1:
        k = k-1
        R = singletons[k]
        used.add(R)
    if k == 0:
        return matching, R
    for i in range(k):
        x = singletons[i]
        indices.append(x)
        newspace.append(space[x])
    tree = KDTree(newspace)
    ind = tree.query(newspace,k=min(40,k),return_distance=False)
    
    unused = []
    for i in range(len(ind)):
        idx = indices[i]
        if idx not in used:
            for j in ind[i]:
                jdx = indices[j]
                if jdx not in used and idx != jdx:
                   matching.add((idx, jdx))
                   used.add(idx)
                   used.add(jdx)
                   break
    for i in range(n):
        if i not in used:
            unused.append(i)
    m = len(unused)
    
    for i in range(int(m/2)):
        matching.add((unused[2*i], unused[2*i + 1]))
   
    return matching, R
    
    



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




def key(v):
    return -1*gainmap[v]

def updateGain(G, new_sol, old_sol):
    global gainmap
    global gainlist
    global change
    for u in change:
        if new_sol[u] != old_sol[u]:
            gainlist.remove(u)
            gainmap[u] = 0
            for v, w in G.iterNeighborsWeights(u):
                if new_sol[u] == new_sol[v]:
                    gainmap[u] += w
                else:
                    gainmap[u] -= w
                if v not in change:
                    gainlist.remove(v)
                    if new_sol[u] == new_sol[v]:
                        gainmap[v] += w
                    else:
                        gainmap[v] -= w
                    gainlist.add(v)
            gainlist.add(u)
                    
def buildGain(G, sol):
    global gainmap
    global gainlist
    gainmap = []
    for i in range(G.numberOfNodes()):
        gainmap.append(0)
    for u,v,w in G.iterEdgesWeights():
        if sol[u] == sol[v]:
            gainmap[u] += w
            gainmap[v] += w
        else:
            gainmap[u] -= w
            gainmap[v] -= w

    gainlist = SortedKeyList([i for i in range(G.numberOfNodes())],key=key)



def randomWalk(E, p, u):
    near = E.most_similar(str(u))
    v = int(near[random.randint(0,len(near)-1)][0])
    if random.random() < p:
        return randomWalk(E, p/2, v)
    return u
    
def n2v(G, sol, sp_size, S):
    global gainlist
    global n2vm
    global last_idx
    n = G.numberOfNodes()

    E = n2vm
    V = set()
    spnodes = []
    
        
    while len(V) < sp_size:        
        u = gainlist[last_idx]
        if u not in V and u in S:
            V.add(u)
            near = E.most_similar(str(u))
            for x in near:
                X = randomWalk(E, 0.5, int(x[0]))
                if len(V) > sp_size:
                    break
                
                elif X not in V and x in S:
                    V.add(int(X))
        last_idx = (last_idx + 1) % len(gainlist)
    for u in V:
        spnodes.append(u)

    return spnodes



def SOCSubProb(G, sol, sp_size):
    global gainmap
    global nodes
    if len(nodes) != G.numberOfNodes():
        nodes = [i for i in range(G.numberOfNodes())]
    sandpile = {}
    spnodes = []
    used = set()
    for x in range(G.numberOfNodes()):
        sandpile[x] = random.randint(0, max(1,int(G.weightedDegree(x))-1))
        
    while len(spnodes) < sp_size:
        i = random.choices(nodes, weights=gainmap, k=1)[0] 
        sandpile[i] += 1
        k = sandpile[i]
        d = int(G.weightedDegree(i))
        q = Queue()
        if k > d:
            q.put(i)
            while not q.empty():
                u = q.get()
                if u not in used:
                    spnodes.append(u)
                    used.add(u)
                    if len(spnodes) > sp_size:
                        break
                sandpile[u] = sandpile[u] - int(G.weightedDegree(u))
                for v in G.iterNeighbors(u):
                    if random.random() < 0.05:
                        continue
                    sandpile[v] += 1
                    if sandpile[v] > int(G.weightedDegree(v)):
                        q.put(v)
    subprob = nw.graph.Graph(n=sp_size+2, weighted = True, directed = False)
    mapProbToSubProb = {}
    ct =0
    i = 0
    idx =0
    change = set()
    while i < sp_size:
        u = spnodes[i]
        change.add(u)
        mapProbToSubProb[u] = idx
        idx += 1
        i += 1



    keys = mapProbToSubProb.keys()
    total = 0
    j = 0
    while j < sp_size:
        u = spnodes[j]
        spu = mapProbToSubProb[u]
        for v, w in G.iterNeighborsWeights(u):
            if v not in keys:
                if sol[v] == 0:
                    spv = idx
                else:
                    spv = idx + 1
                subprob.increaseWeight(spu, spv, w)
            else:
                spv = mapProbToSubProb[v]
                if u < v:
                    subprob.increaseWeight(spu, spv, w)
            total += w
        j += 1

    subprob.increaseWeight(idx, idx+1, G.totalEdgeWeight() - total)
    return (subprob, mapProbToSubProb, idx)
            
            




def randGainSubProb(G, sol, sp_size):
    global gainmap
    global gainlist
    global change
    subprob = nw.graph.Graph(n=sp_size+2, weighted = True, directed = False)
    
    l = [i for i in range(G.numberOfNodes())]
    S = l

    gain = gainlist[:sp_size]

    mapProbToSubProb = {}
    ct =0
    i = 0
    idx =0
    change = set()
    while i < sp_size:
        u = gain[i]
        change.add(u)
        mapProbToSubProb[u] = idx
        idx += 1
        i += 1



    keys = mapProbToSubProb.keys()
    total = 0
    j = 0
    while j < sp_size:
        u = gain[j]
        spu = mapProbToSubProb[u]
        for v, w in G.iterNeighborsWeights(u):
            if v not in keys:
                if sol[v] == 0:
                    spv = idx
                else:
                    spv = idx + 1
                subprob.increaseWeight(spu, spv, w)
            else:
                spv = mapProbToSubProb[v]
                if u < v:
                    subprob.increaseWeight(spu, spv, w)
            total += w
        j += 1

    subprob.increaseWeight(idx, idx+1, G.totalEdgeWeight() - total)
    return (subprob, mapProbToSubProb, idx)




def embedGainSubProb(G, sol, sp_size):
    global gainmap
    global gainlist
    global change
    subprob = nw.graph.Graph(n=sp_size+2, weighted = True, directed = False)
    
    l = [i for i in range(G.numberOfNodes())]
#    S = random.sample(l,min(sp_size*10,G.numberOfNodes()))
    S = l

    gain = n2v(G,sol,sp_size, S)

    mapProbToSubProb = {}
    ct =0
    i = 0
    idx =0
    change = set()
    while i < sp_size:
        u = gain[i]
        change.add(u)
        mapProbToSubProb[u] = idx
        idx += 1
        i += 1



    keys = mapProbToSubProb.keys()
    total = 0
    j = 0
    while j < sp_size:
        u = gain[j]
        spu = mapProbToSubProb[u]
        for v, w in G.iterNeighborsWeights(u):
            if v not in keys:
                if sol[v] == 0:
                    spv = idx
                else:
                    spv = idx + 1
                subprob.increaseWeight(spu, spv, w)
            else:
                spv = mapProbToSubProb[v]
                if u < v:
                    subprob.increaseWeight(spu, spv, w)
            total += w
        j += 1

    subprob.increaseWeight(idx, idx+1, G.totalEdgeWeight() - total)
    return (subprob, mapProbToSubProb, idx)



def qaoa(G, p=3):
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


def refine(G, sol, sp_size, obj, spsolver, sp=None, m=0):
    if sp != None:
        subprob = sp
    elif m == 1:
        subprob = randGainSubProb(G, sol, sp_size)
    else:
        subprob = SOCSubProb(G, sol, sp_size)

    idx = subprob[2]
    mapProbToSubProb = subprob[1]
    old_sol = {}
    old_sol[idx] = 0
    old_sol[idx+1] = 1
    if spsolver == "qbsolv":
        Q = build_qubo(subprob[0])
        response = QBSolv().sample_qubo(Q)
        solution = response.samples()[0]
    elif spsolver == "qaoa":
        solution = qaoa(subprob[0])
    elif spsolver == "gurobi":
        solution = pyomo(subprob[0], spsolver)
    elif spsolver == "ipopt":
        solution = pyomo(subprob[0], spsolver)
    elif spsolver == "sampling":
        solution = randSampleSolve(subprob[0])

    
    n = G.numberOfNodes()
    new_sol = {}
    
    keys = mapProbToSubProb.keys()
    for i in range(n):
        if i in keys:
            old_sol[mapProbToSubProb[i]] = sol[i]
            new_sol[i] = solution[mapProbToSubProb[i]]
        else:
            if sol[i] == 0:
                new_sol[i] = solution[idx]
            else:
                new_sol[i] = solution[idx+1]
    new_obj = calc_obj(G, new_sol)
    changes = []
    for i in range(n):
        if sol[i] != new_sol[i]:
            changes.append(i)
    if new_obj > obj:
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


def informedMatching(G, GVs, tol=0):
    n = G.numberOfNodes()
    nxG = nw.nxadapter.nk2nx(G)
    eigvec = nx.linalg.algebraicconnectivity.fiedler_vector(nxG, tol=0.0001,method='lobpcg')
    F = []
    for i in range(len(eigvec)):
        F.append((eigvec[i],i))
    F.sort()
    F.reverse()
    used = set()
    matching = set()
    optionCt = 1 + tol*2
    for i in range(n):
        u = F[i][1]
        prob = GVs[u][0] / GVs[u][1]
        upart = 1 if random.random() < prob else 0
        if u in used:
            continue
        used.add(u)
        options = [(i + int(n/2) + j) % n for j in range(0-tol, 0+tol+1)]
        c = random.randint(0,optionCt-1)
        j = options[c]
        v = F[j][1]
        prob = GVs[v][0] / GVs[v][1]
        vpart = 1 if random.random() < prob else 0
        ct = 0
        retry = False
        while v in used and vpart != upart:
            j = (j+1) % n
            v = F[j][1]
            ct += 1
            if ct > 2*optionCt:
                retry = True
                break
        if retry:
            while v in used:
                j = (j+1) % n
                v = F[j][1]
        used.add(v)
        matching.add((u,v))
    remaining = -1
    if n % 2 == 1:
        for x in F:
            if x[1] not in used:
                remaining = x[1]
        

    return matching, remaining, F



def getFiedler(G):
    nxG = nw.nxadapter.nk2nx(G)
    eigvec = nx.linalg.algebraicconnectivity.fiedler_vector(nxG, tol=0.0001,method='lobpcg')
    F = []
    for i in range(len(eigvec)):
        F.append((eigvec[i],i))
    F.sort()
    F.reverse()
    return F

def spectralMatching(G, tol=0):
    n = G.numberOfNodes()
    F = getFiedler(G)
    used = set()
    matching = set()
    optionCt = 1 + tol*2
    for i in range(n):
        u = F[i][1]
        if u in used:
            continue
        used.add(u)
        options = [(i + int(n/2) + j) % n for j in range(0-tol, 0+tol+1)]
        c = random.randint(0,optionCt-1)
        j = options[c]
        v = F[j][1]
        ct = 0
        while v in used:
            j = (j+1) % n
            v = F[j][1]
            ct += 1
            if ct > n:
                break
            
        used.add(v)
        matching.add((u,v))
    remaining = -1
    if n % 2 == 1:
        for x in F:
            if x[1] not in used:
                remaining = x[1]
        

    return matching, remaining, F


def randomMatching(G):
    n = G.numberOfNodes()
    matching = set()
    used = set()
    R = -1
    for i in range(n):
        if i in used:
            continue
        used.add(i)
        j = random.randint(i,n-1)
        while j in used:
            j = j+1
        used.add(j)
        matching.add((i,j))
    for i in range(n):
        if i not in used:
            R = i
            break
    return matching, R


def matchingCoarsening(G,C,GVs=None):
    global sortedEdges
    edgesDropped = 0
    edgesAggregated = 0
    n = G.numberOfNodes()
    i = 0
    j = int(n/2)
    mapCoarseToFine = {}
    mapFineToCoarse = {}
    idx = 0
    F = None
    newGVs = {} if GVs != None else None
    if C == 0:
        M, R, F = spectralMatching(G, 2)
    elif C == 1:
        M, R = randomMatching(G)
    elif C == 2:
        M, R = embeddingMatching(G, 3)
    elif GVs != None:
        M, R, F = informedMatching(G, 2, GVs)
    for u, v in M:
        mapCoarseToFine[idx] = [u, v]
        mapFineToCoarse[u] = idx
        mapFineToCoarse[v] = idx
        if GVs != None:
            newGVs[idx] = (GVs[u][0] + GVs[v][0], GVs[u][1] + GVs[v][1])
        idx += 1
    if n % 2 == 1:
        mapCoarseToFine[idx] = [R]
        mapFineToCoarse[R] = idx
        if GVs != None:
            newGVs[idx] = GVs[R]
        idx += 1
    cG = nw.graph.Graph(n=idx, weighted=True, directed=False)
    for u,v in G.iterEdges():
        cu = mapFineToCoarse[u]
        cv = mapFineToCoarse[v]
        cG.increaseWeight(cu, cv, G.weight(u, v))
    cG.removeSelfLoops()
    cG.indexEdges()
    C, _, _ = getComponents(cG)
    nC = len(C)
    if nC > 1:
        for i in range(1, nC):
            cG.addEdge(C[0][0], C[i][0], w=0.001)
    newS = []
    if GVs != None:
        for i in range(idx):
            prob = newGVs[i][0] / newGVs[i][1]
            if prob < random.random():
                newS.append(1)
            else:
                newS.append(0)
    d = 3
    space = np.array(embedNodes(cG, d))
    sortedEdges = []
    for u, v, w in cG.iterEdgesWeights():
        sortedEdges.append((embedDist(u,v,space,d),(u,v,w)))
    sortedEdges.sort()
    return (cG, mapCoarseToFine, F, newGVs, newS)



def randInitialSolution(G):
    sol = {}
    for x in G.iterNodes():
        sol[x] = random.randint(0,1)
    return sol
       

def getComponents(G):
    q = Queue()
    idx = 0
    i = 0
    componentMap = {}
    components = {}
    mns = 0
    while mns < G.numberOfNodes():
        q.put(mns)
        components[idx] = []
        while not q.empty():
            u = q.get()
            if componentMap.get(u) == None:
                componentMap[u] = idx
                components[idx].append(u)
                for v in G.iterNeighbors(u):
                    q.put(v)
        for i in range(mns, G.numberOfNodes()+1):
            if componentMap.get(i) == None:
                mns = i
                break
        idx += 1
    return components, componentMap, idx
            
def sparsifyEmbedding(G, ratio):
    global sortedEdges
    n = G.numberOfNodes()
    m = G.numberOfEdges()
    e_goal = int(ratio*m)
    e_del = 0
    deletedEdges = set()
    nG = nw.graph.Graph(n=G.numberOfNodes(), directed=False,weighted=True)
    for u,v,w in G.iterEdgesWeights():
        nG.setWeight(u,v,w)

    for i in range(len(sortedEdges)):
        if e_del < e_goal:
            e = sortedEdges[i][1]
            nG.removeEdge(e[0], e[1])
            e_del += 1
        else:
            break

    
    C = getComponents(nG)

    if C[2] != 1:
        for e in deletedEdges:
            u = e[0]
            v = e[1]
            w = e[2]
            cu = C[1][u]
            cv = C[1][v]
            c = C[0]
            if cu != cv:
                nG.setWeight(u,v,w)
                for i in c[cv]:
                    c[cu].append(i)
                    C[1][i] = cu
                c[cv] = None
            elif random.randint(0,100) < 3:
                nG.setWeight(u,v,w)
    return nG


def sparsifyRandom(G, ratio):
    n = G.numberOfNodes()
    m = G.numberOfEdges()

    deletedEdges = set()
    e_goal = int(ratio*m)
    e_del = 0
    nodeQueue = Queue(maxsize = 0)
    nodeQueue.put(random.randint(0, n-1))
    rc = 0
    while(e_del < e_goal):
        u = nodeQueue.get()
        for v in G.iterNeighbors(u):
            w = G.weight(u,v)
            e = (u, v, w) if u < v else (v, u, w)
            if w != 0 and e not in deletedEdges:
                if random.randint(0, 100) / 100 < ratio:
                    deletedEdges.add(e)
                    e_del += 1
                    if e_del == e_goal:
                        break
                    nodeQueue.put(v)
        if nodeQueue.empty():
            nodeQueue.put(random.randint(0,n-1))
            rc += 1
        if rc > 30:
            break
    nG = nw.graph.Graph(n=G.numberOfNodes(), directed=False,weighted=True)
    for u,v,w in G.iterEdgesWeights():
        nG.setWeight(u,v,w)
        
    for e in deletedEdges:
        if e[2] != 0:
            nG.removeEdge(e[0],e[1])
    C = getComponents(nG)
    if C[2] != 1:
        for e in deletedEdges:
            u = e[0]
            v = e[1]
            w = e[2]
            cu = C[1][u]
            cv = C[1][v]
            c = C[0]
            if cu != cv:
                nG.setWeight(u,v,w)
                for i in c[cv]:
                    c[cu].append(i)
                    C[1][i] = cu
                c[cv] = None
            elif random.randint(0,100) < 3:
                nG.setWeight(u,v,w)
    return nG




def sparsify(G, ratio):
    if sparsification == 'algebraic':
        ADS = nw.sparsification.AlgebraicDistanceSparsifier()
        s = ADS.scores(G)
        H = ADS.getSparsifiedGraphOfSize(G, 1 - ratio, s)
    elif sparsification == 'embedding':
        H = sparsifyEmbedding(G,ratio)
    else:
        print('random')
        H = sparsifyRandom(G, ratio)
    return H

    
def calc_density(G):
    e = G.numberOfEdges()
    n = G.numberOfNodes()
    return (2*e) / (n*(n-1))




def maxcut_solve(G, C, obj=None, S=None):
    global fiedler_list
    global last_idx
    global n2vm
    global eratio
    refinements = 0
    print(gname)
    print(str(G))
    fiedler_list = []
    GVs = None
    if S != None:
        GVs = {}
        for i in range(G.numberOfNodes()):
            GVs[i] = (1,1) if S[i] == 1 else (0,1)
    start = time.perf_counter()
    problem_graph = G
    n = G.numberOfNodes()   
    if sparsification != None:
        sG = sparsify(G, eratio)
    else:
        sG = G
    hierarchy = [(G,sG)]
    hierarchy_map = []
    old = G.numberOfNodes()
    new = G.numberOfNodes()
    fiedler_list = []
    while(True):
        old = G.numberOfNodes()
        if new < spsize:
            break
        coarse= matchingCoarsening(sG, C, GVs)    
        G = coarse[0]
        GVs = coarse[3]
        fiedler_list.append(coarse[2])
        if sparsification != None:
            sG = sparsify(G, eratio)
        else:
            sG = G

        hierarchy.append((G,sG))
        hierarchy_map.append(coarse[1])
        new = G.numberOfNodes()
    end = time.perf_counter()
    fiedler_list.append(getFiedler(G))
    fiedler_list.reverse()
    hierarchy_map.reverse()
    hierarchy.reverse()
    solution = randInitialSolution(G)
    if obj == None:
        obj = 0
    for i in range(len(hierarchy_map)):
        fG = hierarchy[i+1][0]
        cG = hierarchy[i][0]
        sG = hierarchy[i+1][1]
        fMap = hierarchy_map[i]
        print("\n\nLEVEL " + str(i))
        new_solution = {}
                
        for j in range(cG.numberOfNodes()):
            for x in fMap[j]:
                new_solution[x] = solution[j]
        solution = new_solution
        obj = calc_obj(fG, solution)
        buildGain(fG, solution)
        ct = 0
        print(str(fG))
        nxG = nw.nxadapter.nk2nx(sG)
        n2v = Node2Vec(nxG, dimensions=3,walk_length=16,num_walks=10, quiet=True)
        n2vm = n2v.fit(window=10,min_count=1).wv
        while ct < 3:
            res = refine(sG, solution, spsize, obj, solver)
            refinements += 1
            updateGain(fG, res[0], solution)
            solution = res[0]
            new_obj = calc_obj(fG, solution)
            
            if new_obj <= obj:
                ct += 1
            else:
                ct = 0
                last_idx = 0
                obj = new_obj
        ct = 0
        nxG = nw.nxadapter.nk2nx(fG)
        n2v = Node2Vec(nxG, dimensions=3,walk_length=16,num_walks=10,quiet=True)
        n2vm = n2v.fit(window=10,min_count=1).wv
        while ct < 5:
            res = refine(fG, solution, spsize, obj, solver)                
            refinements += 1
            updateGain(fG, res[0], solution)
            solution = res[0]
            new_obj = res[1]
            if new_obj <= obj:
                ct += 1
            else:
                ct = 0
                last_idx = 0
                obj = new_obj
        print("\nTOTAL REFINEMENTS: " + str(refinements))
        print(gname + " OBJECTIVE AFTER REFINEMENT: " + str(obj))
        print("IMBALANCE: " + str(calc_imbalance(solution, fG.numberOfNodes())))
        if coarseonly == 1:
            exit()
    return calc_obj(problem_graph, solution), solution



def testSPSelection(G):
    global n2vm
    solution = randInitialSolution(G)
    obj = calc_obj(G, solution)
    buildGain(G, solution)
    print(gname)
    ct = 0
    print("random gain")
    for i in range(251):
        if ct > 20:
            print('no more improvement')
            break
        res = refine(G, solution, spsize, obj, solver, m=1)                
        updateGain(G, res[0], solution)
        solution = res[0]
        new_obj = res[1]
        if new_obj > obj:
            obj = new_obj
            ct = 0
        else:
            ct += 1
        if i % 50 == 0:
            print(str(i) + ": " + str(new_obj))
    print("FINAL: " + str(obj))
    obj = 0
    ct = 0
    solution = randInitialSolution(G)
    print("embed gain")
    nxG = nw.nxadapter.nk2nx(G)
    n2v = Node2Vec(nxG, dimensions=3,walk_length=16,num_walks=10, quiet=True)
    n2vm = n2v.fit(window=10,min_count=1).wv
    for i in range(251):
        if ct > 20:
            print('no more improvement')
            break
        res = refine(G, solution, spsize, obj, solver, m=0)                
        updateGain(G, res[0], solution)
        solution = res[0]
        new_obj = res[1]
        if new_obj > obj:
            obj = new_obj
            ct = 0
        else:
            ct += 1
        if i % 50 == 0:
            print(str(i) + ": " + str(new_obj))
    print("FINAL: " + str(obj))
    print('mixed gain')
    obj = 0
    solution = randInitialSolution(G)
    ct = 0
    M = 1
    for i in range(251):
        if ct > 5:
            M = 0
            ct = 0
        if ct > 20:
            print('no more improvement')
            break
        res = refine(G, solution, spsize, obj, solver, m=M)
        updateGain(G, res[0], solution)
        solution = res[0]
        new_obj = res[1]
        if new_obj > obj:
            obj = new_obj
            ct = 0
        else:
            ct += 1
        if i % 50 == 0:
            print(str(i) + ": " + str(new_obj))
    print("FINAL: " + str(obj))

if gformat == 'alist':
    G = readGraph()
elif gformat == 'elist':
    G = readGraphEList()

if testsubprob == 1:
    testSPSelection(G)
    exit()





G = nw.components.ConnectedComponents.extractLargestConnectedComponent(G)
print(str(G))
s = time.perf_counter()
max_obj = 0
S = None
for i in range(cycles):
    obj, S = maxcut_solve(G, 1, obj=max_obj, S=S)
    if obj > max_obj:
        max_obj = obj
e = time.perf_counter()
f = open(gname + '.sol', 'w')
f.write(str(S))
f.close()
print("Found maximum value for " + str(solver) + " " + str(gname) + " of " + str(max_obj) + " " + str(e-s) + "s")
