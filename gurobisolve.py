import networkit as nk
from gurobipy import Model, GRB
graph = nk.graph.Graph()
model = Model("Max-Cut")
model.setParam(GRB.Param.TimeLimit, 60)
model.setParam('OutputFlag', False) 
variables = {}
for node in graph.iterNodes():
    variables[node] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}")

objective = 0
for u,v,w in graph.iterEdgesWeights():
    objective += (2*w*variables[v]*variables[u]) - w*(variables[v] + variables[u])
model.setObjective(objective, GRB.MAXIMIZE)
model.optimize()