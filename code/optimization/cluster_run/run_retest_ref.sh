#!/bin/bash

FILR=$1
FILW=$2
PARAM=$3

seed_sample=1
seed_sample_retest=2

sbatch -n 1 -t 00:05:00 --wrap="./dv3_retest $seed_sample_retest $PARAM $FILR $FILW"


