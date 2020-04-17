CMGGPU = no

ifeq ($(CMGGPU),yes)
  CXX=h5c++
  ROOTTRT=/afs/cern.ch/user/b/borghtom/TensorRT-7.0.0.11
  ROOTCUDA=/usr/local/cuda
  ROOTH5 =/data/patatrack/opt/anaconda3
  ROOTCUDNN =/afs/cern.ch/user/b/borghtom/cuda

  CXXFLAGS = -std=c++11 -I$(ROOTCUDA)/include \
	     -I$(ROOTTRT)/include \
	     -I$(ROOTTRT)/samples/common -O2 \
	     -I$(ROOTH5)/include \
	     -I$(ROOTCUDNN)/include
  
  LIB =  -L /usr/lib64  -L $(ROOTCUDA)/lib64/ -L $(ROOTCUDNN)/lib64  -L $(ROOTTRT)/lib \
	 -L$(ROOTH5)/lib -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn \
	 -lmyelin -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group \
	 -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lhdf5
else
  CXX =g++

  ROOTTRT=/home/redo/Tensort/TensorRT-7.0.0.11
  ROOTCUDA=/usr/local/cuda

  CXXFLAGS = -std=c++11 -I$(ROOTCUDA)/include \
 	     -I$(ROOTTRT)/include \
	     -I$(ROOTTRT)/samples/common -O2

  LIB = -L $(ROOTCUDA)/lib64/ -L $(ROOTTRT)/lib -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn \
        -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group -lhdf5_cpp -lhdf5
endif

OBJ = Trk_Doublets.o

Trk_Doublets : $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LIB)

clean:
	rm -f $(OBJ) Trk_Doublets
