import networkit as nw
import pyomo.environ as pyo
import sys

def pyomo(G, time):

    opt = pyo.SolverFactory('gurobi_direct')
    opt.options['TimeLimit'] = time

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
    print(results)
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

print(sys.argv)
G = nw.readGraph("./graphs/"+sys.argv[1], nw.Format.EdgeListSpaceOne)
pyomo(G, int(sys.argv[2]))