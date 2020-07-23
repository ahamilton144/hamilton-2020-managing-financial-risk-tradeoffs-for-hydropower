# Makefile for 2dv version of SFPUC financial risk multi-objective optimization
CXX = mpiCC
CXXFLAGS = -c -Wall
INC = -I /opt/ohpc/pub/libs/gnu8/openmpi3/boost/1.70.0/include
LIBS = -lm
UNAME_S = $(shell uname -s)

ifneq (, $(findstring SunOS, $(UNAME_S)))
    LIBS += -lnsl -lsocket -lresolv
endif

all: problem_borgms problem_sensitivity_borgms problem_retest

problem_borgms: problem_borgms.o moeaframework.o borgms.o mt19937ar.o
	$(CXX) problem_borgms.o moeaframework.o borgms.o mt19937ar.o -o problem_borgms

problem_sensitivity_borgms: problem_sensitivity_borgms.o moeaframework.o borgms.o mt19937ar.o
	$(CXX) problem_sensitivity_borgms.o moeaframework.o borgms.o mt19937ar.o -o problem_sensitivity_borgms

problem_retest: problem_retest.o moeaframework.o borgms.o mt19937ar.o
	$(CXX) problem_retest.o moeaframework.o borgms.o mt19937ar.o -o problem_retest

problem_borgms.o: main.cpp ./../../misc/borg/moeaframework.h ./../../misc/boostutil.h ./../../misc/borg/borgms.h
	$(CXX) $(CXXFLAGS) $(INC) main.cpp -o problem_borgms.o

problem_sensitivity_borgms.o: main_sensitivity.cpp ./../../misc/borg/moeaframework.h ./../../misc/boostutil.h ./../../misc/borg/borgms.h
	$(CXX) $(CXXFLAGS) $(INC) main_sensitivity.cpp -o problem_sensitivity_borgms.o

problem_retest.o: main_retest.cpp ./../../misc/borg/moeaframework.h ./../../misc/boostutil.h ./../../misc/borg/borgms.h
	$(CXX) $(CXXFLAGS) $(INC) main_retest.cpp -o problem_retest.o

moeaframework.o: ./../../misc/borg/moeaframework.c ./../../misc/borg/moeaframework.h
	$(CXX) $(CXXFLAGS) ./../../misc/borg/moeaframework.c

borgms.o: ./../../misc/borg/borgms.c ./../../misc/borg/borgms.h ./../../misc/borg/borg.h
	$(CXX) $(CXXFLAGS) ./../../misc/borg/borgms.c $(LIBS)

mt19937ar.o: ./../../misc/borg/mt19937ar.c ./../../misc/borg/mt19937ar.h
	$(CXX) $(CXXFLAGS) ./../../misc/borg/mt19937ar.c $(LIBS)

clean:
	rm -rf *.o
	rm problem_borgms
	rm problem_sensitivity_borgms
	rm problem_retest
