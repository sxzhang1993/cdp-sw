C = nvcc
NVCCFLAGS = -arch=sm_70 
CFLAGS = -std=c++11 -rdc=true -lcudadevrt

all: align

align: align.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o align align.cu 

clean:
	rm -f align *.dat

