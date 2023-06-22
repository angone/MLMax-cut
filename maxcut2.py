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
T = 0
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

def parallel(ref):
    pr = cProfile.Profile()
    pr.enable()
    s = int(ref[2] * time.perf_counter())
    random.seed(s)
    np.random.seed(s)
    R = Refinement(ref[0], args.sp, 'mqlib', ref[1])
    R.refineLevel()
    pr.disable()
    pr.dump_stats(str(ref[0].numberOfNodes())+'_'+str(ref[2])+".process")
    return R.solution, R.obj

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
        self.alpha = 0.25
        self.randomness = 0
        self.bound = 20
        self.increase = 0
        
    def refine_coarse(self):
        self.solution = self.mqlibSolve(5, G=self.G)
        self.obj = self.calc_obj(self.G, self.solution)
        print('Coarse Level:',self.obj)
    
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
        self.gainlist = SortedKeyList([i for i in range(self.n)], key=lambda x: self.gainmap[x]+0.01*x)
        
         
    def updateGain(self, S):
        used = set()
        if self.last_subprob == None:
            return
        changed = set()
        to_update = set()
        for u in self.last_subprob:
            if u not in self.locked_nodes:
                self.locked_nodes.add(u)
            if S[u] != self.solution[u]:
                changed.add(u)
        for u in changed:
            for v in self.G.iterNeighbors(u):
                if v not in self.locked_nodes:
                    if v not in to_update:
                        to_update.add(v)
                    if v in self.gainlist:
                        self.gainlist.remove(v)
                    w = 2*self.G.weight(u,v)*(1+self.alpha)
                    if S[u] == S[v]:
                        self.gainmap[v] += w
                    else:
                        self.gainmap[v] -= w       
        for u in to_update:
            self.gainlist.add(u)
        
        
    def lockGainSubProb(self, spnodes=None):
        if spnodes != None:
            spsize = len(spnodes)
        elif len(self.gainlist) >= self.spsize:
            if self.randomness <= 0 and self.randomness > -1:
                spnodes = self.gainlist[:self.spsize]
            else:
                if self.randomness <= -1:
                    randomnodes = self.spsize
                else:
                    randomnodes = int(self.randomness * self.spsize)
                spsize = self.spsize - randomnodes
                spnodes = self.gainlist[:spsize]
                used = set(spnodes)
                c = 0
                while c < randomnodes:
                    k = random.randint(0, len(self.gainlist)-1)
                    if self.gainlist[k] not in used:
                        spnodes.append(self.gainlist[k])
                        used.add(self.gainlist[k])
                        c += 1
        else:
            self.passes += 1
            self.randomness += self.increase
            spsize = self.spsize
            spnodes = self.gainlist[:len(self.gainlist)-1]
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

    def fixSolution(self):
        done = False
        while not done:
            done =  True
            for i in range(self.n):
                if self.gainmap[i] > 0:
                    done = False
                    self.solution[i] = 1 - self.solution[i]
                    self.gainmap[i] = 0
                    for v in self.G.iterNeighbors(i):
                        w = self.G.weight(v, i)
                        if self.solution[v] == self.solution[i]:
                            self.gainmap[i] += w
                            self.gainmap[v] += 2*w
                        else:
                            self.gainmap[i] -= w
                            self.gainmap[v] -= 2*w
        self.obj = self.calc_obj(self.G, self.solution)
        return


    def refine(self):
        calls = 0
        while len(self.gainlist) > 0:
            subprob = self.lockGainSubProb()
            mapProbToSubProb = subprob[1]
            S = self.mqlibSolve(0.1, subprob[0])
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

            if new_obj >= self.obj:
                self.obj = new_obj
                self.updateGain(new_sol)
                self.solution = new_sol.copy()
            calls += 1
        print(calls)



    def refineLevel(self):
        ct = 0
        obj = 0
        while self.passes < self.bound:
            self.refine()
            self.locked_nodes = set()
            self.buildGain()
        self.fixSolution()

    def test(self):
        S = self.mqlibSolve(5, G=self.G)
        O = self.calc_obj(self.G, S)
        print("MQLib:",O)
        print("MLM:", self.obj)
        spnodes = []
        ct = 0
        for i in range(self.n):
            if S[i] == self.solution[i]:
                spnodes.append(i)
                ct += 1
        print("diff:", ct)
        if ct > 2:
            subprob = self.lockGainSubProb(spnodes)
            mapProbToSubProb = subprob[1]
            S2 = self.mqlibSolve(5, subprob[0])
            new_sol = self.solution.copy()
        
            keys = mapProbToSubProb.keys()
            for i in keys:
                new_sol[i] = S2[mapProbToSubProb[i]]
            new_obj = self.calc_obj(self.G, new_sol)
            print("after sp:",new_obj)
        print(subprob)
        print(self.gainmap)

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
    
    def noisySolution(self, ratio):
        S = self.solution.copy()
        for i in range(int(len(S)*ratio)):
            k = random.randint(0, len(S)-1)
            S[k] = 1 - S[k]
        return S
    
    def solve(self):
        global T
        G = self.problem_graph
        print(G)
        while G.numberOfNodes() > 2*self.spsize:
            E = EmbeddingCoarsening(G, 3,'cube')
            E.coarsen()
            print(E.cG)
            self.hierarchy.append(E)
            G = E.cG
        self.hierarchy.reverse()
        R = Refinement(G, self.spsize, 'mqlib', [random.randint(0, 1) for _ in range(G.numberOfNodes())])
        R.refine_coarse()
        self.obj = R.obj
        self.solution = R.solution
        starts = 40
        for i in range(len(self.hierarchy)):
            E = self.hierarchy[i]
            G = E.G
            fineToCoarse = E.mapFineToCoarse
            print('Level',i+1,'Nodes:',G.numberOfNodes(),'Edges:',G.numberOfEdges())
            S = [0 for _ in range(G.numberOfNodes())]
            for i in range(len(S)):
                S[i] = self.solution[fineToCoarse[i]]
            self.solution = S
            if True:
                R = Refinement(E.G, self.spsize, 'mqlib', self.solution)
                R.refineLevel()
                #R.test()
                self.solution = R.solution
                self.obj = R.obj
            else:
                if False:
                    inputs = [(E.G, self.noisySolution(0.2), j) for j in range(starts)]
                else:
                    inputs = [(E.G, self.solution.copy(), j) for j in range(starts)]
                a = time.perf_counter()
                pool = multiprocessing.Pool(processes=starts)
                outputs = pool.map(parallel, inputs)
                b = time.perf_counter()
                T += (b-a)
                print([outputs[i][1] for i in range(len(outputs))])
                max_obj = outputs[0][1]
                max_sol = outputs[0][0]
                for O in outputs:
                    if O[1] > max_obj:
                        max_obj = O[1]
                        max_sol = O[0]
                self.solution = max_sol
                self.obj = max_obj
                print('MLM:',self.obj)
            starts = max(2, int(starts/2))

def get_max_memory_usage():
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Convert from kilobytes to megabytes
    max_memory_mb = max_memory / 1024
    return max_memory_mb
    
s = time.perf_counter()
M = MaxcutSolver(args.g, args.sp, args.S)
M.solve()
t = time.perf_counter()
print('Found obj', M.obj, 'in', t-s, 's')
print(T)



max_memory_usage = get_max_memory_usage()
print(f"Maximum memory usage: {max_memory_usage} MB")