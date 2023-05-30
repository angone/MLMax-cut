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

random.seed(0)
np.random.seed(0)
faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument("-g", type = str, default = "None", help = "graph file")
parser.add_argument("-sp", type = int, default = 18, help = "size of subproblems")
parser.add_argument("-S", type = str, default = "mqlib", help = "subproblem solver")
parser.add_argument("-f", type = str, default = "elist", help = "graph format")
parser.add_argument("-e", type = str, default = 'cube', help = 'shape of embedding')
parser.add_argument("-c", type = int, default = 0, help = 'coarse only')
args = parser.parse_args()


class EmbeddingCoarsening:
    def __init__(self, G, d, shape):
        self.G = G
        self.d = d
        self.n = G.numberOfNodes()
        self.space = np.random.rand(self.n, d)
        self.shape = shape
        self.M = set()

    def buildObj(self, u):
        def obj(pos):
            o = 0
            for x in self.G.iterNeighbors(u):
                temp = 0
                for i in range(self.d):
                    temp += (pos[i] - self.space[x][i])**2
                o += ((temp)*self.G.weight(u,x))
            return -1 * o
        return obj
    
    def dist(self, u, v):
        a = 0
        for i in range(self.d):
            a += (self.space[u][i] - self.space[v][i])**2
        return np.sqrt(a)
    
    def embed(self):
        n = self.G.numberOfNodes()
        embeddings = []
        for i in range(n):
            b = self.buildObj(i)
            bnds = [(0,1) for _ in range(self.d)]
            p = [random.random() for _ in range(self.d)]
            def sphere(x):
                return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - 1
            cons = [{'type': 'ineq', 'fun': sphere}] if self.shape == 'sphere' else None
            res = minimize(b, p, bounds=bnds, tol=0.0001, constraints=cons)
            self.space[i] = res.x 
    
    def match(self):
        n = self.G.numberOfNodes()
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
        R = -1
        k = len(singletons)
        if k % 2 == 1:
            k = k-1
            R = singletons[k]
            used.add(R)
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
            if idx not in used:
                for j in ind[i]:
                    jdx = indices[j]
                    if jdx not in used and idx != jdx:
                        self.M.add((idx, jdx))
                        used.add(idx)
                        used.add(jdx)
                        break
        for i in range(n):
            if i not in used:
                unused.append(i)
        m = len(unused)
        
        for i in range(int(m/2)):
            self.M.add((unused[2*i], unused[2*i + 1]))
        self.R = R

    def coarsen(self):
        n = self.G.numberOfNodes()
        i = 0
        j = int(n/2)
        self.mapCoarseToFine = {}
        self.mapFineToCoarse = {}
        idx = 0
        self.embed()
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
        for u,v in self.G.iterEdges():
            cu = self.mapFineToCoarse[u]
            cv = self.mapFineToCoarse[v]
            self.cG.increaseWeight(cu, cv, self.G.weight(u, v))
        self.cG.removeSelfLoops()
        self.cG.indexEdges()
    
class Refinement:
    def __init__(self, G, spsize, solver, solution):
        self.G = G
        self.n = G.numberOfNodes()
        self.gainmap = np.zeros((G.numberOfNodes(),1))
        self.passes = 0
        self.spsize = spsize
        self.solver = solver
        self.solution = solution
        self.buildGain()
        self.obj = self.calc_obj(G, solution)
        self.last_subprob = None
        self.unused = SortedKeyList([i for i in range(self.n)])
        self.locked_nodes = set()
        self.alpha = 0.5

    def refine_coarse(self):
        self.solution = self.mqlibSolve(5, G=self.G)
        self.obj = self.calc_obj(self.G, self.solution)
    
    def terminate(self):
        for i in range(self.n):
            if self.gainmap[i] > 0 or self.uses[i] < 2:
                return False
        return True

    def calc_obj(self, G, solution):
        obj = 0
        n = G.numberOfNodes()
        for u, v in G.iterEdges():
            obj += G.weight(u, v)*(2*solution[u]*solution[v] - solution[u] - solution[v])
        return -1 * obj
    
    def mqlibSolve(self, t, G=None):
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
        return (res['solution']+1)/2 

    def buildGain(self):
        for u,v,w in self.G.iterEdgesWeights():
            if self.solution[u] == self.solution[v]:
                self.gainmap[u] += w
                self.gainmap[v] += w
            else:
                self.gainmap[u] -= w
                self.gainmap[v] -= w
            self.gainlist = SortedKeyList([i for i in range(self.n)], key=lambda x: self.gainmap[x])
    
    def updateGain(self):
        used = set()
        if self.last_subprob == None:
            return
        for u in self.last_subprob:
            self.gainmap[u] = 0
            self.gainlist.remove(u)
            self.locked_nodes.add(u)
            for v, w in self.G.iterNeighborsWeights(u):
                if self.solution[u] == self.solution[v]:
                    self.gainmap[u] += w
                else:
                    self.gainmap[u] -= w
                if v not in used:
                    self.gainlist.remove(v)
                    used.add(v)
                    self.gainmap[v] = 0
                    for x, y in self.G.iterNeighborsWeights(v):
                        if x in self.locked_nodes:
                            y = y*(1+self.alpha)
                        if self.solution[v] == self.solution[x]:
                            self.gainmap[v] += y
                        else:
                            self.gainmap[v] -= y
                    self.gainlist.add(v)
        

    def lockGainSubProb(self):
        if len(self.gainlist) >= self.spsize:
            spnodes = self.gainlist[:self.spsize]
        else:
            self.passes += 1
            self.gainlist = SortedKeyList([i for i in range(self.n)], key=lambda x: self.gainmap[x])
            spnodes = self.gainlist[:self.spsize]
        subprob = nw.graph.Graph(n=self.spsize+2, weighted = True, directed = False)
        mapProbToSubProb = {}
        ct =0
        i = 0
        idx =0
        change = set()
        while i < self.spsize:
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
        while j < self.spsize:
            u = spnodes[j]
            spu = mapProbToSubProb[u]
            for v, w in self.G.iterNeighborsWeights(u):
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

    def SOCSubProb(self):
        spnodes = []
        used = set()
        sandpile = {}
        nodeq = Queue()
        for x in range(self.G.numberOfNodes()):
            sandpile[x] = random.randint(0, max(1,int(self.G.weightedDegree(x))-1))
        while len(spnodes) < self.spsize:
            j = -1
            if len(self.posgain) > 0:
                j = random.randint(0, len(self.posgain)-1)
                i = self.posgain[j]
            elif len(self.unused) > 0:
                j = random.randint(0, len(self.unused)-1)
                i = self.unused[j]
            else:
                i = random.randint(0, self.G.numberOfNodes()-1)
            sandpile[i] += 1
            k = sandpile[i]
            d = int(self.G.weightedDegree(i))
            if k > d:
                nodeq.put(i)
                while not nodeq.empty():
                    u = nodeq.get()
                    if u not in used:
                        spnodes.append(u)
                        used.add(u)
                        if len(spnodes) > self.spsize:
                            break
                    sandpile[u] = sandpile[u] - int(self.G.weightedDegree(u))
                    for v in self.G.iterNeighbors(u):
                        if random.random() < 0.1:
                            continue
                        sandpile[v] += 1
                        if sandpile[v] > int(self.G.weightedDegree(v)):
                                nodeq.put(v)
        subprob = nw.graph.Graph(n=self.spsize+2, weighted = True, directed = False)
        mapProbToSubProb = {}
        ct =0
        i = 0
        idx =0
        change = set()
        while i < self.spsize:
            u = spnodes[i]
            change.add(u)
            self.uses[u] += 1
            if u in self.unused:
                self.unused.remove(u)
            mapProbToSubProb[u] = idx
            idx += 1
            i += 1
        self.last_subprob = spnodes


        keys = mapProbToSubProb.keys()
        total = 0
        j = 0
        while j < self.spsize:
            u = spnodes[j]
            spu = mapProbToSubProb[u]
            for v, w in self.G.iterNeighborsWeights(u):
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
        subprob = self.lockGainSubProb()
        mapProbToSubProb = subprob[1]
        S = self.mqlibSolve(0.25, subprob[0])
        new_sol = self.solution.copy()
        
        keys = mapProbToSubProb.keys()
        for i in keys:
            new_sol[i] = S[mapProbToSubProb[i]]
        new_obj = self.calc_obj(self.G, new_sol)
        if new_obj >= self.obj:
            self.obj = new_obj
            self.solution = new_sol.copy()
        self.updateGain()

    def refineLevel(self):
        ct = 0
        obj = 0
        while self.passes < 4:
            self.refine()

class MaxcutSolver:
    def __init__(self, fname, sp, solver):
        self.problem_graph = nw.readGraph("./graphs/"+fname, nw.Format.EdgeListSpaceOne)
        self.problem_graph = nw.components.ConnectedComponents.extractLargestConnectedComponent(self.problem_graph)
        self.hierarchy = []
        self.hierarchy_map = []
        self.spsize = sp
        self.solver = solver
        self.solution = None
        self.obj = 0
    
    def solve(self):
        G = self.problem_graph
        while G.numberOfNodes() > 2*self.spsize:
            E = EmbeddingCoarsening(G, 3,'cube')
            E.coarsen()
            print(E.cG)
            self.hierarchy.append(E)
            G = E.cG
        self.hierarchy.reverse()
        R = Refinement(G, 18, 'mqlib', [random.randint(0, 1) for _ in range(G.numberOfNodes())])
        R.refine_coarse()
        self.obj = R.obj
        self.solution = R.solution
        for i in range(len(self.hierarchy)):
            E = self.hierarchy[i]
            G = E.G
            fineToCoarse = E.mapFineToCoarse
            print(str(G), len(fineToCoarse))
            S = [0 for _ in range(G.numberOfNodes())]
            for i in range(len(S)):
                S[i] = self.solution[fineToCoarse[i]]
            self.solution = S
            R = Refinement(E.G, 18, 'mqlib', self.solution)
            R.refineLevel()
            self.solution = R.solution
            self.obj = R.obj
    
s = time.perf_counter()
M = MaxcutSolver(args.g, args.sp, args.S)
M.solve()
t = time.perf_counter()
print('Found obj', M.obj, 'in', t-s, 's')
