
inf = open("soc-gemsec-HR.edges", "r")
of = open("soc-gemsec-HR.edges2", "w")

line = inf.readline().split()
while line != []:
    u = int(line[0]) + 1
    v = int(line[1]) + 1
    of.write(str(u) + " " + str(v)+"\n")
    line = inf.readline().split()
of.close()
