#/bin/bash
filr=$1
filw=$2

java -cp ../misc/MOEAFramework-2.12-Demo.jar HypervolumeEval ${filr} >> ${filw}
