# Use g++ to compile .cpp files
CPP  = g++
SDIR = .

# Eigen Headers
# EIG_INC = /usr/local/Cellar/eigen/3.3.4/include/eigen3/
EIG_INC = /usr/local/Eigen

# Include standard optimization flags
CPPFLAGS = -O2 -g -c -I$(EIG_INC) -I$(SDIR)

# List of all the objects you need
OBJECTS  = ar1.o kGrid.o vfInit.o binaryVal.o vfStep.o binaryMax.o timer.o parameters.o

# Rule that tells make how to make the program from the objects
main :	main.o $(OBJECTS)
	$(CPP) -o main main.o $(OBJECTS) $(LFLAGS) 

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f main
