#!/bin/bash

NSAMPLES=150
METHOD=Latin
JAVA_ARGS="-cp ../misc/MOEAFramework-2.12-Demo.jar"
RFILE="../../data/generated_inputs/param_LHC_bounds.txt"
WFILE="../../data/generated_inputs/param_LHC_sample.txt"
# Generate the parameter samples
java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.SampleGenerator --method ${METHOD} --n ${NSAMPLES} --p ${RFILE} --o ${WFILE}
# add baseline parameter estimates to end of file
echo '0.914 0.40 -1.73 1.00 0.25' >> ${WFILE}

