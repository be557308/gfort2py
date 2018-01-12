
FC = gfortran

ifdef $(TRAVIS)
	FC = $(TRAVIS_GFORTRAN)
endif

all:
	$(FC) -fdump-tree-all -fPIC -shared -c test2.f90
	$(FC) -fdump-tree-all -fPIC -shared -o tester.so test_mod.f90 test2.f90
	cp tester.mod tester.gz
	gunzip -f tester.gz
