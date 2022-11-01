import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str, default="")
args = parser.parse_args()
f = open(args.f, "r")
l = f.readline()
l = l.split()
while l != []:
    a = int(l[0]) + 1
    b = int(l[1]) + 1
    print(str(a) + " " + str(b))
    l = f.readline()
    l = l.split()
