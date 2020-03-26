
CXX = g++

ROOTTRT=/home/redo/Tensort/TensorRT-7.0.0.11

CXXFLAGS = -std=c++11 -I/usr/local/cuda/include \
	   -I$(ROOTTRT)/include \
	   -I$(ROOTTRT)/samples/common 

LIB =  -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn \
       -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group

OBJ = Trk_Doublets.o

Trk_Doublets : $(OBJ)
	g++ $(OBJ) -o $@ $(LIB)

clean:
	rm -f $(OBJ) Trk_Doublets
