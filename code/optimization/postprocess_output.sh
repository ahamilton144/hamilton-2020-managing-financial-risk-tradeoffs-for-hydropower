#!/bin/bash

scenario='baseline'
nseeds=50
dir='../../data/optimization_output/'${scenario}
param=150

mkdir ${dir}/'runtime'${param}
mkdir ${dir}/'sets'${param}
for seed in `seq 1 ${nseeds}`;
do
	mv ${dir}/'runtime/param'${param}'_seed1_seedB'${seed}'.runtime' ${dir}/'runtime/param'${param}'_seedS1_seedB'${seed}'.runtime'
	mv ${dir}/'runtime/param'${param}'_seedS1_seedB'${seed}'.runtime' ${dir}/'runtime'${param}'/param'${param}'_seedS1_seedB'${seed}'.runtime'
        mv ${dir}/'sets/param'${param}'_seedS1_seedB'${seed}'.set' ${dir}/'sets'${param}'/param'${param}'_seedS1_seedB'${seed}'.set'
done

sh find_refSets.sh $scenario $param

sh get_objs.sh $scenario $param $nseeds

sh find_runtime_metrics.sh $scenario $param $nseeds

sh find_hypervolume.sh ${dir}/'param'${param}'_borg.reference' ${dir}/'param'${param}'_borg.hypervolume'

sh run_retest_ref.sh ${dir}/'param'${param}'_borg.resultfile' ${dir}/'param'${param}'_borg_retest.resultfile' $param



scenario='sensitivity'
nseeds=10
dir='../../data/optimization_output/'${scenario}
for param in `seq 0 149`;
do
        mkdir ${dir}/'runtime'${param}
        mkdir ${dir}/'sets'${param}
        for seed in `seq 1 ${nseeds}`;
        do
		mv ${dir}/'runtime/param'${param}'_seed1_seedB'${seed}'.runtime' ${dir}/'runtime/param'${param}'_seedS1_seedB'${seed}'.runtime'
                mv ${dir}/'runtime/param'${param}'_seedS1_seedB'${seed}'.runtime' ${dir}/'runtime'${param}'/param'${param}'_seedS1_seedB'${seed}'.runtime'
                mv ${dir}/'sets/param'${param}'_seedS1_seedB'${seed}'.set' ${dir}/'sets'${param}'/param'${param}'_seedS1_seedB'${seed}'.set'
        done

        sh find_refSets.sh $scenario $param

        sh get_objs.sh $scenario $param $nseeds

        sh find_runtime_metrics.sh $scenario $param $nseeds

        sh find_hypervolume.sh ${dir}/'param'${param}'_borg.reference' ${dir}/'param'${param}'_borg.hypervolume'

        sh run_retest_ref.sh ${dir}/'param'${param}'_borg.resultfile' ${dir}/'param'${param}'_borg_retest.resultfile' $param
done

