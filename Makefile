
TestPCE : src/TestPCE.cpp $(wildcard src/pce/*.h)
	c++ -std=c++11 -Isrc -Iexternal/eigen src/TestPCE.cpp -o TestPCE