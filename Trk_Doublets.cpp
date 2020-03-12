#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <ctime>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "common.h"
#include "logger.h"
#include "buffers.h"
#include "H5Cpp.h"

using namespace nvuffparser;  // there might be some ambiguity on some functions in TRT 6.0, so this statement was not removed 
using namespace nvinfer1; // there might be some ambiguity on some functions in TRT 6.0, so this statement was not removed 

using namespace H5;

#define MAX_WORKSPACE (1 << 20)
#define maxBatchSize 30000  // it's intended per context, so it can be lower than batchSize if more contexts are created
                            // for a multistream version -- just be sure that maxBatchSize < batchSize/n_contexts


int main(){

    //define the data info
    const H5std_string FILE_NAME("pixel_only_data_test.h5");
    const H5std_string DATASET_NAME_DATA("data/block0_values");
    const H5std_string DATASET_NAME_LABELS("labels/block0_values");

    const std::string dir{"./"} // directory where the .uff model is located
    const std::string uffFileName{dir + "pixel_only_final.uff"}; // name of the .pb trained model converted into .uff

    // Attributes of the model
    const int input_c{20};
    const int input_h{16};
    const int input_w{16};
    int img_size = input_c * input_h * input_w;
    const int output_size{2};
    const std::string inputTensorName = "hit_shape_input";
    const std::string outputTensorName = "output/Softmax";

    std::cout << "*** MODEL TO IMPORT: " << fileName << "\n";
    std::cout << "*** DATASET FILE: " << FILE_NAME << "\n";
    std::cout << "*** MAX WORKSPACE: " << MAX_WORKSPACE << "\n";
    std::cout << "*** MAX BATCHSIZE: " << maxBatchSize << std::endl;

    int batchSize = 30000;
    std::cout << "*** Number of images to process (batchSize): " << batchSize << std::endl;

    float ms;
    Logger gLogger; // object for warning and error reports

    // *** IMPORTING THE MODEL *** 
    std::cout << "*** IMPORTING THE UFF MODEL ***" << std::endl;

    // Create the builder and the network    
    nvinfer1::IBuilder* builder(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    nvinfer1::INetworkDefinition* network(builder->createNetwork());
    nvinfer1::IBuilderConfig* config(builder->createBuilderConfig());

    // Create the UFF parser
    nvuffparser::IUffParser* parser(nvuffparser::createUffParser());
    assert(parser);

    // Declare the network inputs and outputs of the model to the parser
    parser->registerInput(inputTensorName.c_str(), nvinfer1::Dims3(input_c, input_h, input_w), nvuffparser::UffInputOrder::kNCHW);
    // in TRT 4.0 I was using DimsCHW instead of Dims3, but that one might be deprecated in TRT 6, so I used this one
    parser->registerOutput(outputTensorName.c_str());
    
    // Parse the imported model to populate the network
    parser->parse(uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    std::cout << "*** IMPORTING DONE ***" << std::endl; 

    // Build the engine
    std::cout << "*** BUILDING THE ENGINE ***" << std::endl;

    //Build the engine using the builder object
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);
    config->setFlag(BuilderFlag::kFP16); //16-bit kernels are permitted -- this is useful with GPUs that support full FP16 operations
    nvinfer1::ICudaEngine* mEngine(builder->buildEngineWithConfig(*network, *config));
    assert(mEngine);
    std::cout << "*** BUILDING DONE ***" << std::endl; 

    // Destroy network, builder, config and parser -- we don't need them anymore
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    // Start inference
    std::cout << "*** PERFORMING INFERENCE ***" << std::endl;

    // Create the input and the output buffers on Host
    output = std::array<float, batchSize * output_size>;

    // Open the file and the dataset
    H5File file( FILE_NAME, H5F_ACC_RDONLY );
    DataSet dataset = file.openDataSet( DATASET_NAME_DATA );

    // Get dataspace of the dataset
    DataSpace dataspace = dataset.getSpace();

    // Get the number of dimensions in the dataspace
    int rank = dataspace.getSimpleExtentNdims();
    hsize_t dims[2];
    int status_n = dataspace.getSimpleExtentDims(dims, NULL);
    std::cout << "Rank: " << rank << "\n";
    std::cout << "Dimensions: " << dims[0] << " " << dims[1] << std::endl;

    // Define the memory space to read dataset
    DataSpace memspace(rank,dims);
    std::cout << "MEMSPACE CREATED" << std::endl;

    // Read dataset back and display
    float *input_data = new float[dims[0] * dims[1]]; // might be replaced with std_array<float, dims[0]*dims[1]>, but need to check
                                                      // if compatible with H5 libraries

    dataset.read(input_data, PredType::NATIVE_FLOAT, memspace, dataspace);
    std::cout << "DATASET READ" << std::endl;

    // Engine requires exactly IEngine::getNbBindings() number of buffers  
    int nbBindings = mEngine->getNbBindings();
    assert(nbBindings == 2); // 1 input and 1 output
    std::cout << nbBindings << std::endl;
    
    // Create a context to store intermediate activation values
    nvinfer1::IExecutionContext* context(mEngine->createExecutionContext());
    void* buffers[nbBindings];

    const int inputIndex = mEngine->getBindingIndex(inputTensorName.c_str());
    const int outputIndex = mEngine->getBindingIndex(outputTensorName.c_str());

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * img_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * output_size * sizeof(float)));

    const auto t_start = std::chrono::high_resolution_clock::now();
    // Copy data to HtD
    CHECK(cudaMemcpy(buffers[inputIndex], input_data, batchSize * img_size * sizeof(float), cudaMemcpyHostToDevice));

    // Synchronous kernel execution
    context->execute(batchSize, buffers);

    // Copy data to HtD
    CHECK(cudaMemcpy(output, buffers[outputIndex], batchSize * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    const auto t_end = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference on " << batchSize << " doublets performed in " << ms << " milliseconds." << std::endl;

    // Free GPU memory
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    context->destroy();
    mEngine->destroy();

    for(i = 0; i <  batchSize; i++){
        std::cout << "Image n: " << i << "\n";
        std::cout << "y_pred = (" << output[2*i] << "," << output[2*i+1] << ")";
    }

    delete[] input_data;

    nvuffparser::shutdownProtobufLibrary();
    return 0;

}