#!/bin/bash
borg='./../../misc/borg/'
misc='./../../misc/'
g++ -c ${borg}moeaframework.c -o moeaframework.o -I/${borg}
g++ -c ${borg}borg.c -o borg.o -I/${borg}
g++ -c ${borg}mt19937ar.c -o mt19937ar.o -I/${borg}
ar rs lib_moeaf_borg_mt.a moeaframework.o borg.o mt19937ar.o
g++ -g -Wall main.cpp lib_moeaf_borg_mt.a -o dv2_borg -I/${borg} -I/${misc}
echo "Finished build"
