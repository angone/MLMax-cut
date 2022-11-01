import networkit as nw
from dwave_qbsolv import QBSolv
import random
from datetime import datetime
import argparse
import time
import numpy as np
import networkx as nx
import pyomo.environ as pyo
import scipy
from queue import Queue
import faulthandler
faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument("-gname", type = str, default = "None", help = "graph file")
parser.add_argument("-method", type = str, default = "None", help = "refinement method")
parser.add_argument("-spsize", type = int, default = 18, help = "size of subproblems")
parser.add_argument("-solver", type = str, default = "qbsolv", help = "subproblem solver")
parser.add_argument("-optimizer", type = str, default = "COBYLA", help = "qaoa optimizer")
parser.add_argument("-gformat", type = str, default = "elist", help = "graph format")


args = parser.parse_args()


method = args.method
gname = args.gname
spsize = args.spsize
solver = args.solver
optimizer = args.optimizer
gformat = args.gformat




if solver == 'qaoa':
    median, kde = get_median_pre_trained_kde(3)
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
        opt.options['TimeLimit'] = 5
    elif solver == "ipopt":
        opt = pyo.SolverFactory('ipopt')
        opt.options['max_wall_time'] = 5
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


def nodeGain(G, sol, u):
    gain = 0
    for x in G.iterNeighbors(u):
        if sol[x] == sol[u]:
            gain += G.weight(u, x)
        else:
            gain -= G.weight(u, x)
    return gain



def buildParts(G, sol):
    S.clear()
    T.clear()
    for x in range(G.numberOfNodes()):
        if sol[x] == 0:
            S.append(x)
        elif sol[x] == 1:
            T.append(x)
    return

    


def calcDensity(G):
    e = G.numberOfEdges()
    n = G.numberOfNodes()
    return (2*e) / (n *(n-1))
            


def trivial(sp, sol):
    idx = sp.numberOfNodes() - 2
    count = 0
    res = True
    for u in sp.iterNodes():
        
        for v in sp.iterNeighbors(u):
            if u == idx or u == idx + 1 or v == idx or v == idx+1:
                continue
            if sol[u] == sol[v]:
                count += 1
                res = False
    print(count)
    return res


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


def randGainSubProb(G, sol, sp_size):
    subprob = nw.graph.Graph(n=sp_size+2, weighted = True, directed = False)
    sampleSize = 10 * sp_size
    sample = [x for x in range(G.numberOfNodes())]
    random.shuffle(sample)
    sample = sample[:sampleSize]
    gain = []
    for x in sample:
        gain.append((nodeGain(G, sol, x), x))
        
    gain = sorted(gain)
    gain.reverse()

    mapProbToSubProb = {}
    totalGain = 0
    ct =0
    i = 0
    idx =0
    while i < sp_size:
        u = gain[i][1]
        mapProbToSubProb[u] = idx
        idx += 1
        i += 1



    keys = mapProbToSubProb.keys()
    total = 0
    j = 0
    while j < sp_size:
        u = gain[j][1]
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
    if sp != None:
        subprob = sp
    else:
        subprob = randGainSubProb(G, sol, sp_size)

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
    if checktrivial == 1:
        if trivial(subprob[0], old_sol):
            trivialsp += 1

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
    orderedNodes = []
    for i in range(len(eigvec)):
        orderedNodes.append((eigvec[i],i))
    orderedNodes.sort()
    orderedNodes.reverse()
    used = set()
    matching = set()
    optionCt = 1 + tol*2
    for i in range(n):
        u = orderedNodes[i][1]
        prob = GVs[u][0] / GVs[u][1]
        upart = 1 if random.random() < prob else 0
        if u in used:
            continue
        used.add(u)
        options = [(i + int(n/2) + j) % n for j in range(0-tol, 0+tol+1)]
        c = random.randint(0,optionCt-1)
        j = options[c]
        v = orderedNodes[j][1]
        prob = GVs[v][0] / GVs[v][1]
        vpart = 1 if random.random() < prob else 0
        ct = 0
        retry = False
        while v in used and vpart != upart:
            j = (j+1) % n
            v = orderedNodes[j][1]
            ct += 1
            if ct > 2*optionCt:
                retry = True
                break
        if retry:
            while v in used:
                j = (j+1) % n
                v = orderedNodes[j][1]
        used.add(v)
        matching.add((u,v))
    remaining = -1
    if n % 2 == 1:
        for x in orderedNodes:
            if x[1] not in used:
                remaining = x[1]
        

    return matching, remaining



def spectralMatching(G, tol=0):
    n = G.numberOfNodes()
    nxG = nw.nxadapter.nk2nx(G)
    eigvec = nx.linalg.algebraicconnectivity.fiedler_vector(nxG, tol=0.0001,method='lobpcg')
    orderedNodes = []
    for i in range(len(eigvec)):
        orderedNodes.append((eigvec[i],i))
    orderedNodes.sort()
    orderedNodes.reverse()
    used = set()
    matching = set()
    optionCt = 1 + tol*2
    for i in range(n):
        u = orderedNodes[i][1]
        if u in used:
            continue
        used.add(u)
        options = [(i + int(n/2) + j) % n for j in range(0-tol, 0+tol+1)]
        c = random.randint(0,optionCt-1)
        j = options[c]
        v = orderedNodes[j][1]
        ct = 0
        while v in used:
            j = (j+1) % n
            v = orderedNodes[j][1]
            ct += 1
            
        used.add(v)
        matching.add((u,v))
    remaining = -1
    if n % 2 == 1:
        for x in orderedNodes:
            if x[1] not in used:
                remaining = x[1]
        
        

    return matching, remaining


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


def matchingCoarsening(G,C):
    edgesDropped = 0
    edgesAggregated = 0

    
    nxG = nw.nxadapter.nk2nx(G)
    
    n = G.numberOfNodes()
    i = 0
    j = int(n/2)
    mapCoarseToFine = {}
    mapFineToCoarse = {}
    idx = 0
    if C == 0:
        M, R = spectralMatching(G, 2)
    elif C == 1:
        M, R = randomMatching(G)
    elif C == 2:
        M, R = informedMatching(G, 2, GVs)
    print(n)
    print(len(M))
    for u, v in M:
        mapCoarseToFine[idx] = [u, v]
        mapFineToCoarse[u] = idx
        mapFineToCoarse[v] = idx
        idx += 1
    if n % 2 == 1:
        mapCoarseToFine[idx] = [R]
        mapFineToCoarse[R] = idx
    cG = nw.graph.Graph(n=idx, weighted=True, directed=False)
    for u,v in G.iterEdges():
        cu = mapFineToCoarse[u]
        cv = mapFineToCoarse[v]
        cG.increaseWeight(cu, cv, G.weight(u, v))
    cG.removeSelfLoops()
    cG.indexEdges()
    return (cG, mapCoarseToFine)



def randInitialSolution(G):
    sol = {}
    for x in G.iterNodes():
        sol[x] = random.randint(0,1)
    return sol
       

def getComponents(G):
    q = Queue(maxsize = 0)
    idx = 0
    i = 0
    componentMap = {}
    components = {}
    mns = 0
    while mns < G.numberOfNodes():
        q.put(mns)
        components[idx] = set()
        while not q.empty():
            u = q.get()
            if componentMap.get(u) == None:
                componentMap[u] = idx
                components[idx].add(u)
                for v in G.iterNeighbors(u):
                    q.put(v)
        for i in range(mns, G.numberOfNodes()+1):
            if componentMap.get(i) == None:
                mns = i
                break
        idx += 1
    return components, componentMap, idx
            


def sparsify(G, ratio):
    n = G.numberOfNodes()
    m = G.numberOfEdges()

    deletedEdges = set()
    total = (n*(n-1))
    e_del = 0
    e_goal = m - int(ratio*total)
    if e_goal <= 0:
        return G
    nodeQueue = Queue(maxsize = 0)
    nodeQueue.put(random.randint(0, n-1))

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
                    c[cu].add(i)
                    C[1][i] = cu
                c[cv] = None
            elif random.randint(0,100) < 3:
                nG.setWeight(u,v,w)

        

    return nG


    
def calc_density(G):
    e = G.numberOfEdges()
    n = G.numberOfNodes()
    return (2*e) / (n*(n-1))




def maxcut_solve(G, C):

    refinements = 0
    print(gname)
    print(str(G))

    start = time.perf_counter()
    problem_graph = G

    density_cutoff = calc_density(G)
    density = density_cutoff
    if density > 0.4:
        sG = sparsify(G, 0.4)
        density_cutoff = calc_density(G)
        density = density_cutoff
    else:
        sG = G
    hierarchy = [(G,sG)]
    hierarchy_map = []
    old = G.numberOfNodes()
    new = 0
    while(abs(new - old) > 2*spsize):
        old = G.numberOfNodes()
        if old <= 2*(1+spsize):
            break
        coarse= matchingCoarsening(G,C)    
        G = coarse[0]
        if calc_density(G) > density_cutoff:
            sG = sparsify(G, density_cutoff)
        else:
            sG = G

        hierarchy.append((G,sG))
        hierarchy_map.append(coarse[1])
        new = G.numberOfNodes()
    end = time.perf_counter()
    
    hierarchy_map.reverse()
    hierarchy.reverse()
    solution = randInitialSolution(G)
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
        ct = 0

        while ct < 3:
            res = refine(sG, solution, spsize, obj, method, solver)
            refinements += 1
            solution = res[0]
            new_obj = res[1]
            if new_obj <= obj:
                ct += 1
            else:
                ct = 0
                obj = new_obj
        ct = 0
        while ct < 5:
            res = refine(fG, solution, spsize, obj, method, solver)                
            refinements += 1
            solution = res[0]
            new_obj = res[1]
            if new_obj <= obj:
                ct += 1
            else:
                ct = 0
                obj = new_obj
        print("\nTOTAL REFINEMENTS: " + str(refinements))
        print("OBJECTIVE AFTER REFINEMENT: " + str(obj))
        print("IMBALANCE: " + str(calc_imbalance(solution, fG.numberOfNodes())))

    return calc_obj(problem_graph, solution)



if gformat == 'alist':
    G = readGraph()
elif gformat == 'elist':
    G = readGraphEList()




G = nw.components.ConnectedComponents.extractLargestConnectedComponent(G)
print(str(G))
s = time.perf_counter()
obj = maxcut_solve(G, 0)
e = time.perf_counter()
print("Found maximum value for " + str(gname) + " of " + str(obj) + " " + str(e-s) + "s")
