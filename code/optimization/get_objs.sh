#!/bin/bash
SCENARIO=$1
PARAM=$2
NSEEDS=$3

dir=../../data/optimization_output/${SCENARIO}
ndv=2
nobj=2
seeds=$(seq 1 ${NSEEDS})
for seed in ${seeds}
do
	awk -v ndv="$ndv" 'BEGIN {FS=" "}; /^#/ {print $0}; /^[^#/]/ {printf("%s %s\n",$(ndv+1),$(ndv+2))}' ${dir}/runtime${PARAM}/param${PARAM}_seedS1_seedB${seed}.runtime >${dir}/objs/param${PARAM}_seedS1_seedB${seed}.obj
done

