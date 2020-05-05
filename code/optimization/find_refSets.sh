#/bin/bash

SCENARIO=$1
PARAM=$2

dir=../../data/optimization_output/${SCENARIO}
ndv=2
nobj=2
eps1=0.075
eps2=0.225

module load python/3.6.9

python3 ../misc/pareto.py ${dir}/sets${PARAM}/*.set -o $((ndv))-$((ndv+nobj-1)) -e $eps1 $eps2 --output ${dir}/param${PARAM}_borg.resultfile --delimiter=" " --comment="#"
cut -d ' ' -f $((ndv+1))-$((ndv+nobj)) ${dir}/param${PARAM}_borg.resultfile >${dir}/param${PARAM}_borg.reference



