
TestPCE : src/TestPCE.cpp $(wildcard src/pce/*.h)
	g++ -std=c++11 -Isrc -Iexternal/eigen src/TestPCE.cpp -o TestPCE

TestMCMC : src/TestMCMC.cpp $(wildcard src/mcmc/*.h)
	clang++-3.5 -std=c++11 -Isrc -Iexternal/eigen src/TestMCMC.cpp -o TestMCMC