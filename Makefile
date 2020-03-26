
CXX = g++

ROOTTRT=/home/redo/Tensort/TensorRT-7.0.0.11
ROOTCUDA=/usr/local/cuda

CXXFLAGS = -std=c++11 -I$(ROOTCUDA)/include \
	   -I$(ROOTTRT)/include \
	   -I$(ROOTTRT)/samples/common 

LIB =  -L $(ROOTCUDA)/lib64/ -L $(ROOTTRT)/lib -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn \
       -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group -lhdf5_cpp -lhdf5

OBJ = Trk_Doublets.o

Trk_Doublets : $(OBJ)
	g++ $(OBJ) -o $@ $(LIB)

clean:
	rm -f $(OBJ) Trk_Doublets
