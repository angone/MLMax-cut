import networkit as nw
from gurobipy import Model, GRB
import sys

def calc_obj(G, solution):
    obj = 0
    for u, v in G.iterEdges():
         obj += G.weight(u, v)*(2*solution[u]*solution[v] - solution[u] - solution[v])
    return -1 * obj

graph = nw.readGraph("./graphs/"+sys.argv[1], nw.Format.EdgeListSpaceOne)
model = Model("Max-Cut")
model.setParam(GRB.Param.TimeLimit, sys.argv[2])
model.setParam('OutputFlag', False) 
variables = {}
for node in graph.iterNodes():
    variables[node] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}")

objective = 0
for u,v,w in graph.iterEdgesWeights():
    objective += w*((2*variables[v]*variables[u]) - (variables[v] + variables[u]))
model.setObjective(objective, GRB.MAXIMIZE)
model.optimize()
solution = [variables[node].x for node in graph.iterNodes()]