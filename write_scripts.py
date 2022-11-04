def write_script(G, C, S):
    f = open(f'{G[:len(G)-6]}_{C}_{S}.pbs', 'w')

    f.write('#!/bin/bash\n')
    f.write('#PBS -l select=1:ncpus=16:mem=30gb:interconnect=10ge,walltime=168:00:00\n')

    f.write('module add gnu-parallel/20220722\n')
    f.write('module load gnu-parallel/20220722\n')

    f.write('source /home/aangone/modules.sh\n')
    f.write('export PATH="/home/aangone/anaconda3/bin:$PATH"\n')
    f.write('source activate quantum\n')
    f.write('cd /home/aangone/MLMax-cut\n')
    f.write('module load gurobi\n')
    f.write(f'python maxcut.py -gname graphs/bigset/{G} -spsize 98 -solver {S} -gformat elist -cycles {C}\n')
    f.write(f'python maxcut.py -gname graphs/bigset/{G} -spsize 98 -solver {S} -gformat elist -cycles {C}\n')
    f.write(f'python maxcut.py -gname graphs/bigset/{G} -spsize 98 -solver {S} -gformat elist -cycles {C}\n')

g = ['fb-pages-artist.edges', 'soc-brightkite.mtx', 'soc-gemsec-HR.edges', 'soc-slashdot.mtx', 'soc-themarker.edges', 'soc-LiveMocha.mtx',
     'soc-epinions.mtx', 'soc-buzznet.mtx']
S = ['gurobi', 'sampling', 'ipopt']

for G in g:
    for s in S:
        for i in range(1,4):
            write_script(G, i, s)




