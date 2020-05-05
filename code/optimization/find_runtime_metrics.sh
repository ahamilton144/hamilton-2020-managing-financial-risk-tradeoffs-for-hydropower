#!/bin/bash
SCENARIO=$1
PARAM=$2
NSEEDS=$3

dir=../../data/optimization_output/${SCENARIO}
ndv=2
nobj=2
seeds=$(seq 1 ${NSEEDS})

JAVA_ARGS="-cp ../misc/MOEAFramework-2.12-Demo.jar"

for seed in ${seeds}
do
	objfil=${dir}/objs/param${PARAM}_seedS1_seedB${seed}.obj
	metricfil=${dir}/metrics/param${PARAM}_seedS1_seedB${seed}.metrics
	reffil=${dir}/param${PARAM}_borg.reference
	sbatch -n 1 -t 12:00:00 --wrap="java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.ResultFileEvaluator -d $nobj -i $objfil -r $reffil -o $metricfil"
done
