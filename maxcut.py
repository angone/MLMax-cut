import networkit as nw
import random
import argparse
import time
import numpy as np
import networkx as nx
import scipy
from queue import SimpleQueue as Queue
import faulthandler
from scipy.optimize import minimize, Bounds
from sklearn.neighbors import KDTree
from sortedcontainers import SortedKeyList
import logging
import MQLib as mq
import multiprocessing
import cProfile
import resource
from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.optimizers import COBYLA
from qiskit import BasicAer
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import pstats
import math
import warnings
T = 0
warnings.filterwarnings("ignore")

random.seed(int(time.perf_counter()))
np.random.seed(int(time.perf_counter()))
faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument("-g", type = str, default = "None", help = "graph file")
parser.add_argument("-sp", type = int, default = 18, help = "size of subproblems")
parser.add_argument("-S", type = str, default = "mqlib", help = "subproblem solver")
parser.add_argument("-f", type = str, default = "elist", help = "graph format")
parser.add_argument("-e", type = str, default = 'cube', help = 'shape of embedding')
parser.add_argument("-c", type = int, default = 0, help = 'coarse only')
parser.add_argument("-sparse", type = float, default = 0, help='ratio to sparsify')
args = parser.parse_args()
sptime = 0
flag = True

def parallel(ref):
    s = int(ref[2])
    random.seed(s)
    np.random.seed(s)
    R = Refinement(ref[0], args.sp, args.S, ref[1])
    R.refineLevel()
    return R.solution, R.obj

class EmbeddingCoarsening:
    def __init__(self, G, d, shape, ratio):
        self.G = G
        self.sG = nw.graphtools.toWeighted(G)
        self.d = d
        self.n = G.numberOfNodes()
        self.space = np.random.rand(self.n, d)
        self.shape = shape
        self.M = set()
        self.R = -1
        self.ratio = ratio

    def sparsify(self):
        if self.ratio == 0:
            return
        removeCount = int(self.ratio * self.sG.numberOfEdges())
        edgeDist = []
        edgeMap = {}
        for u,v in self.sG.iterEdges():
            w = self.sG.weight(u,v)
            d = 0
            for i in range(self.d):
                d += (self.space[u][i] - self.space[v][i])**2
            d = w*np.sqrt(d)
            edgeDist.append((d, u, v))
            edgeMap[(u,v)] = d
            edgeMap[(v,u)] = d
        edgeDist.sort()
        for i in range(removeCount):
            u = edgeDist[i][1]
            v = edgeDist[i][2]
            minE_u = None
            minE_v = None
            for x in self.sG.iterNeighbors(u):
                if v != x:
                    if minE_u == None or edgeMap[(u,x)] < edgeMap[minE_u]:
                        minE_u = (u, x)
            for x in self.sG.iterNeighbors(v):
                if u != x:
                    if minE_v == None or edgeMap[(v,x)] < edgeMap[minE_v]:
                        minE_v = (v, x)
            w = self.sG.weight(u,v)
            if minE_u != None and (minE_v == None or edgeMap[minE_u] < edgeMap[minE_v]):
                u1 = minE_u[0]
                u2 = minE_u[1]
                if self.sG.weight(u1, u2) != 0:
                    self.sG.increaseWeight(u1, u2, w)
            elif minE_v != None and (minE_u == None or edgeMap[minE_v] < edgeMap[minE_u]):
                v1 = minE_v[0]
                v2 = minE_v[1]
                if self.sG.weight(v1, v2) != 0:
                    self.sG.increaseWeight(v1, v2, w)
            self.sG.removeEdge(u, v)

        

    def nodeObj(self, p, c):
        obj = 0
        for x in c:
            for i in range(self.d):
                obj += x[self.d]*(p[i]-x[i])**2
        return obj

    def randPoint(self):
        theta = random.uniform(0,2*math.pi)
        phi = random.uniform(0,math.pi)
        x = math.cos(theta) * math.sin(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(phi)
        return [x, y, z]    

    def optimal(self, u):
        k = 2*self.sG.weightedDegree(u)
        a = 1
        b = -2*k
        c = k**2
        X = self.space[u]
        temp = [0 for _ in range(self.d)]
        for v in self.sG.iterNeighbors(u):
            w = self.sG.weight(u, v)
            for i in range(self.d):
                temp[i] += 2*w*self.space[v][i]
        for i in range(self.d):
            c -= temp[i]**2
        lambda1 = (-b + np.sqrt(b**2 - 4*a*c))/2*a
        lambda2 = (-b - np.sqrt(b**2 - 4*a*c))/2*a
        p = k - lambda1
        q = k - lambda2
        if p == 0 and q == 0:
            return X, 0
        if p != 0:
            p1 = [temp[i] / p for i in range(self.d)]
        if q != 0:
            p2 = [temp[i] / q for i in range(self.d)]
        if p == 0:
            return p2
        if q == 0:
            return p1
        p1d = 0
        p2d = 0
        for v in self.sG.iterNeighbors(u):
            t1 = 0
            t2 = 0
            x = self.space[v]
            for i in range(self.d):
                t1 += (p1[i] - x[i])**2
                t2 += (p2[i] - x[i])**2
            p1d += np.sqrt(t1)
            p2d += np.sqrt(t2)
        if p1d > p2d:
            d = 0
            for i in range(self.d):
                d += (p1[i] - X[i])**2
            return p1, np.sqrt(d)
        if p2d > p1d:
            d = 0
            for i in range(self.d):
                d += (p2[i] - X[i])**2
            return p2, np.sqrt(d)

    def coarseObj(self):
        o = 0
        for u, v, w in self.sG.iterEdgesWeights():
            for i in range(self.d):
                o += w * (self.space[u][i] - self.space[v][i])**2
        print('Current Obj (to be minimized):',o)

    def embed(self, nodes):
        n = self.sG.numberOfNodes()
        change = 0
        for i in nodes:
            res, c = self.optimal(i)
            self.space[i] = res
            change += c
        return change/n
    
    def match(self):
        n = self.sG.numberOfNodes()
        tree = KDTree(self.space)
        ind = tree.query_radius(self.space, 0)
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
        used = set()
        for c in clusters:
            k = len(c)
            if k % 2 == 1:
                singletons.append(c[k-1])
            for i in range(int(k/2)):
                self.M.add((c[2*i], c[2*i + 1]))
                used.add(c[2*i])
                used.add(c[2*i+1])
        indices = []
        newspace = []
        k = len(singletons)
        if k % 2 == 1:
            k = k-1
            self.R = singletons[k]
            used.add(self.R)
        if k == 0:
            return
        for i in range(k):
            x = singletons[i]
            indices.append(x)
            newspace.append(self.space[x])
        newspace = np.array(newspace)
        tree = KDTree(newspace)
        ind = tree.query(newspace,k=min(40,k),return_distance=False)
 
        unused = []
        for i in range(len(ind)):
            idx = indices[i]
            ct = 0
            if idx not in used:
                for j in ind[i]:
                    jdx = indices[j]
                    if jdx not in used and idx != jdx and (ct >=10 or not self.sG.hasEdge(idx, jdx)):
                        self.M.add((idx, jdx))
                        used.add(idx)
                        used.add(jdx)
                        break
                    ct += 1
        for i in range(n):
            if i not in used:
                unused.append(i)
        m = len(unused)
        for i in range(int(m/2)):
            self.M.add((unused[2*i], unused[2*i + 1]))

    def coarsen(self):
        n = self.sG.numberOfNodes()
        i = 0
        j = int(n/2)
        self.mapCoarseToFine = {}
        self.mapFineToCoarse = {}
        idx = 0
        count = 1
        nodes = [i for i in range(n)]
        random.shuffle(nodes)
        change = self.embed(nodes)
        while change > 0.01 and count < 31:
            change = self.embed(nodes)
            count += 1
        print(count, 'iterations until embedding convergence')
        self.sparsify()
        self.match()
        for u, v in self.M:
            self.mapCoarseToFine[idx] = [u, v]
            self.mapFineToCoarse[u] = idx
            self.mapFineToCoarse[v] = idx
            idx += 1
        if n % 2 == 1:
            self.mapCoarseToFine[idx] = [self.R]
            self.mapFineToCoarse[self.R] = idx
            idx += 1
        self.cG = nw.graph.Graph(n=idx, weighted=True, directed=False)
        for u,v in self.sG.iterEdges():
            cu = self.mapFineToCoarse[u]
            cv = self.mapFineToCoarse[v]
            self.cG.increaseWeight(cu, cv, self.G.weight(u, v))
        self.cG.removeSelfLoops()
        self.cG.indexEdges()
    
class Refinement:
    def __init__(self, G, spsize, solver, solution):
        self.G = G
        self.n = G.numberOfNodes()
        self.gainmap = [0 for _ in range(self.n)]
        self.passes = 0
        self.spsize = spsize
        self.solver = solver
        self.solution = solution
        self.buildGain()
        self.obj = self.calc_obj(G, solution)
        self.last_subprob = None
        self.unused = SortedKeyList([i for i in range(self.n)])
        self.locked_nodes = set()
        self.alpha = 0.2
        self.randomness = 1.5
        self.bound = 3
        self.increase = -1
        self.done = False
        
    def refine_coarse(self):
        self.solution, obj = self.mqlibSolve(5, G=self.G)
        self.obj = self.calc_obj(self.G, self.solution)
        return self.obj

    def calc_obj(self, G, solution):
        obj = 0
        n = G.numberOfNodes()
        for u, v in G.iterEdges():
            obj += G.weight(u, v)*(2*solution[u]*solution[v] - solution[u] - solution[v])
        return -1 * obj
    
    def mqlibSolve(self, t=0.1, G=None):
        if G == None:
            G = self.G
            n = self.G.numberOfNodes()
        else:
            n = G.numberOfNodes()
        Q = np.zeros((n,n))
        for u, v, w in G.iterEdgesWeights():
            Q[u][u] -= w
            Q[v][v] -= w
            if u < v:
              Q[u][v] = 2*w
            else:
                Q[v][u] = 2*w
        Q = Q/2
        i = mq.Instance('M',Q)
        def f(s):
            return 1
        res = mq.runHeuristic("BURER2002", i, t, f, 100)

        return (res['solution']+1)/2, res['objval']

    def qaoa(self, p=3, G=None):
        n = G.numberOfNodes()
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
        qaoa = QAOA(optimizer=cobyla, reps=p, quantum_instance=backend)
        algorithm=MinimumEigenOptimizer(qaoa)
        result = algorithm.solve(problem)
        L = result.x
        i = 0
        res = {}
        for x in L:
            res[i] = x
            i += 1
        return res

    def buildGain(self):
        for u,v,w in self.G.iterEdgesWeights():
            if self.solution[u] == self.solution[v]:
                self.gainmap[u] += w
                self.gainmap[v] += w
            else:
                self.gainmap[u] -= w
                self.gainmap[v] -= w
        self.gainlist = SortedKeyList([i for i in range(self.n)], key=lambda x: self.gainmap[x]+0.01*x)
           
    def updateGain(self, S, changed):
        for u in changed:
            for v in self.G.iterNeighbors(u):
                if v not in changed:
                    w = 2*self.G.weight(u,v)*(1+self.alpha)
                    if S[u] == S[v]:
                        self.gainmap[v] += w
                    else:
                        self.gainmap[v] -= w       
         
    def randGainSubProb(self):
            sample_size = min(5*self.spsize, self.n)
            sample = random.sample(range(self.n), sample_size)
            nodes = [i for i in sample]
            nodes.sort(reverse=True, key=lambda x: self.gainmap[x])
            spnodes = nodes[:self.spsize]

            subprob = nw.graph.Graph(n=len(spnodes)+2, weighted = True, directed = False)
            mapProbToSubProb = {}
            i = 0
            idx =0
            change = set()
            while i < len(spnodes):
                u = spnodes[i]
                change.add(u)
                mapProbToSubProb[u] = idx
                idx += 1
                i += 1
            self.last_subprob = spnodes


            keys = mapProbToSubProb.keys()
            j = 0
            while j < len(spnodes):
                u = spnodes[j]
                spu = mapProbToSubProb[u]
                for v in self.G.iterNeighbors(u):
                    w = self.G.weight(u,v)
                    if v not in keys:
                        if self.solution[v] == 0:
                            spv = idx
                        else:
                            spv = idx + 1
                        subprob.increaseWeight(spu, spv, w)
                    else:
                        spv = mapProbToSubProb[v]
                        if u < v:
                            subprob.increaseWeight(spu, spv, w)
                j += 1
            total = subprob.totalEdgeWeight()
            subprob.increaseWeight(idx, idx+1, self.G.totalEdgeWeight() - total)
            return (subprob, mapProbToSubProb, idx)


    def lockGainSubProb(self, spnodes=None):
        if spnodes != None:
            spsize = len(spnodes)
        elif len(self.gainlist) >= self.spsize:
            if self.randomness <= 0:
                spnodes = self.gainlist[:self.spsize]
            else:
                if self.randomness >= 1:
                    randomnodes = self.spsize
                else:
                    randomnodes = int(self.randomness * self.spsize)
                spsize = self.spsize - randomnodes
                spnodes = self.gainlist[:spsize]
                used = set(spnodes)
                c = 0
                while c < randomnodes:
                    k = random.randint(0, self.G.numberOfNodes()-1)
                    if k not in used:
                        spnodes.append(k)
                        used.add(k)
                        c += 1
        else:
            self.done = True
            self.randomness += self.increase
            spsize = self.spsize
            spnodes = self.gainlist[:len(self.gainlist)]
            used = set(spnodes)
            while len(spnodes) < self.spsize:
                k = random.randint(0, self.G.numberOfNodes()-1)
                if k not in used:
                    spnodes.append(k)
                    used.add(k)

        subprob = nw.graph.Graph(n=len(spnodes)+2, weighted = True, directed = False)
        mapProbToSubProb = {}
        ct =0
        i = 0
        idx =0
        change = set()
        while i < len(spnodes):
            u = spnodes[i]
            change.add(u)
            if u in self.unused:
                self.unused.remove(u)
            mapProbToSubProb[u] = idx
            idx += 1
            i += 1
        self.last_subprob = spnodes


        keys = mapProbToSubProb.keys()
        total = 0
        j = 0
        while j < len(spnodes):
            u = spnodes[j]
            if u in self.gainlist:
                self.gainlist.remove(u)
            spu = mapProbToSubProb[u]
            for v in self.G.iterNeighbors(u):
                w = self.G.weight(u,v)
                if v not in keys:
                    if self.solution[v] == 0:
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

        subprob.increaseWeight(idx, idx+1, self.G.totalEdgeWeight() - total)

        return (subprob, mapProbToSubProb, idx)


    def refine(self):
        count = 0
        while count < 3:
            subprob = self.randGainSubProb()
            mapProbToSubProb = subprob[1]
            if self.solver == 'qaoa':
                S =self.qaoa(p=3, G=subprob[0])
            else:
                S, new_obj = self.mqlibSolve(G=subprob[0])
            new_sol = self.solution.copy()
            
            keys = mapProbToSubProb.keys()
            for i in keys:
                new_sol[i] = S[mapProbToSubProb[i]]
            changed = set()
            for u in self.last_subprob:
                if self.solution[u] != new_sol[u]:
                    changed.add(u)
            new_obj = self.obj
            for u in changed:
                for v in self.G.iterNeighbors(u):
                    if v not in changed:
                        w = self.G.weight(u,v)
                        if new_sol[u] == new_sol[v]:
                            new_obj -= w
                        else:
                            new_obj += w
            count += 1
            if new_obj >= self.obj:
                self.updateGain(new_sol, changed)
                self.solution = new_sol.copy()
                if new_obj > self.obj:
                    count = 0
                    self.obj = new_obj
            
    def refineLevel(self):
        self.refine()
        #print('global:', self.calc_obj(self.G, self.mqlibSolve(t=10, G=self.G)))

            


class MaxcutSolver:
    def __init__(self, fname, sp, solver, ratio):
        self.problem_graph = nw.readGraph("./graphs/"+fname, nw.Format.EdgeListSpaceOne)
        self.hierarchy = []
        self.hierarchy_map = []
        self.spsize = sp
        self.solver = solver
        self.solution = None
        self.obj = 0
        self.start = time.perf_counter()
        self.ratio = ratio
    
    def noisySolution(self, ratio):
        S = self.solution.copy()
        for i in range(int(len(S)*ratio)):
            k = random.randint(0, len(S)-1)
            S[k] = 1 - S[k]
        return S
    
    def solve(self):
        global sptime
        G = nw.graphtools.toWeighted(self.problem_graph)
        print(G)
        s = time.perf_counter()
        while G.numberOfNodes() > self.spsize:
            E = EmbeddingCoarsening(G, 3,'cube', self.ratio)
            E.coarsen()
            print(E.cG)
            self.hierarchy.append(E)
            G = E.cG
        t = time.perf_counter()
        print(t-s, 'sec coarsening')
        self.hierarchy.reverse()
        R = Refinement(G, self.spsize, 'mqlib', [random.randint(0, 1) for _ in range(G.numberOfNodes())])
        self.coarse_obj = R.refine_coarse()
        self.obj = R.obj
        self.solution = R.solution
        starts = 40
        for i in range(len(self.hierarchy)):
            E = self.hierarchy[i]
            if i != len(self.hierarchy) -1:
                G = E.G
            else:
                G = self.problem_graph
            fineToCoarse = E.mapFineToCoarse
            print('Level',i+1,'Nodes:',G.numberOfNodes(),'Edges:',G.numberOfEdges())
            S = [0 for _ in range(G.numberOfNodes())]
            for j in range(len(S)):
                S[j] = self.solution[fineToCoarse[j]]
            self.solution = S
            if False:
                sptime -= time.perf_counter()
                R = Refinement(G, self.spsize, self.solver, self.solution)
                R.refineLevel()
                sptime += time.perf_counter()
                self.solution = R.solution
                self.obj = R.obj
            else:
                inputs = [(G, self.solution.copy(), j, self.solver) for j in range(starts)]
                max_obj = self.obj
                max_sol = self.solution
                pool = multiprocessing.Pool()
                sptime -= time.perf_counter()
                outputs = pool.map(parallel, inputs)
                sptime += time.perf_counter()
                for O in outputs:
                    if O[1] > max_obj:
                        max_obj = O[1]
                        max_sol = O[0]
                self.solution = max_sol
                self.obj = max_obj
                print('Objective:',self.obj)
                starts = max(2, int(starts/2))
        print(R.calc_obj(self.problem_graph, self.solution))
       # mqobj = R.calc_obj(self.problem_graph, R.mqlibSolve(t=sptime,G=self.problem_graph))
       # print('mqlib ratio:',self.obj / mqobj)
        #print('coarse ratio:', self.coarse_obj/self.obj)


s = time.perf_counter()
M = MaxcutSolver(fname=args.g, sp=args.sp, solver=args.S, ratio = args.sparse)
M.solve()
t = time.perf_counter()
print('Found obj for',args.g,'of', M.obj, 'in', t-s, 's')


