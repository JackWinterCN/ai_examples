/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.

 * The sample can work in single, double, half precision, but it
 * assumes the data in files is stored in single precision
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include <FreeImage.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "gemv.h"
#include "error_util.h"

// #define DEBUG_OUTPUT

#define IMAGE_H 28
#define IMAGE_W 28

const char *first_image = "one_28x28.pgm";
const char *second_image = "five_28x28.pgm";
const char *third_image = "five_28x28.pgm";
const char *test_image = "five_28x28.pgm";

const char *conv1_bin = "conv2d_kernel.bin";
const char *conv1_bias_bin = "conv2d_bias.bin";
const char *conv2_bin = "conv2d_1_kernel.bin";
const char *conv2_bias_bin = "conv2d_1_bias.bin";
const char *ip1_bin = "dense_kernel.bin";
const char *ip1_bias_bin = "dense_bias.bin";
const char *ip2_bin = "dense_1_kernel.bin";
const char *ip2_bias_bin = "dense_1_bias.bin";

/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("keras_mnist/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};
template <> struct ScaleFactorTypeMap<half1>  { typedef float Type;};

// Conversion from FP64
template <typename T> inline T Convert(double x)
{
    return T(x);
}

template<> inline half1 Convert<half1>(double x)
{
    return cpu_float2half_rn(float(x));
}

// Conversion from FP32
template <typename T> inline T Convert(float x) 
{
    return T(x);
}

template<> inline half1 Convert<half1>(float x) 
{
    return cpu_float2half_rn(x);
}

// Conversion from FP16
template <typename T> inline T Convert(half1 x) 
{
    return T(cpu_half2float(x));
}

template<> inline half1 Convert<half1>(half1 x) 
{
    return x;
}

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::cout << "size = " << size << std::endl;
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }

    std::cout << "Loading binary file " << fname << std::endl;

    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    std::cout << "size_b = " << size_b << std::endl;
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }

    // conversion
    for (int i = 0; i < size; i++)
    {
        data_h[i] = Convert<value_type>(data_tmp[i]);
    }

    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b, cudaMemcpyHostToDevice) );
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    FatalError(zMessage);
}
template <class value_type>
void readImage(const char* fname, value_type* imgData_h)
{
    FILE  *fp_image = NULL;
    fp_image = fopen(fname, "rb");
    for (int r = 0; r < IMAGE_H; ++r)
	{
		for (int c = 0; c < IMAGE_W; ++c)
		{
			unsigned char temp = 0;
			temp = fgetc(fp_image);
			imgData_h[r * IMAGE_W + c] = (float)temp/255.0;
			printf("%3d ", temp);
		}
		printf("\n");
	}
	fclose(fp_image);
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << Convert<real_type>(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2
 } fp16Import_t;

template <class value_type>
struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;

    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;  // store weight param of this layer
    value_type *bias_h, *bias_d;  // store bias param of this layer

    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST)
    {}

    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        fp16Import = _fp16Import;
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
#ifdef DEBUG_OUTPUT
        checkCudaErrors (cudaDeviceSynchronize()); 
        // if(strcmp(fname_weights, ip1_bin) == 0) {
        if(false && strcmp(fname_weights, ip1_bin) == 0) {
            printf("================================%s weight\n", pname);    
            for(int i = 0; i < outputs; i++) {
                printf("===============================%s output[%d]\n", pname, i);
                for (int j = 0; j < inputs; j++) {
                    // printf("%s kernel: [%d]", pname, j);
                    for (int r = 0; r < kernel_dim; r++) {
                      for (int c = 0; c < kernel_dim; c++) {
                        int idx = i * inputs * kernel_dim * kernel_dim +
                                  j * kernel_dim * kernel_dim + r * kernel_dim +
                                  c;
                        if (idx % 32 == 0) {
                          printf("\n");
                        }
                        printf("%7.4f ", (float)data_h[idx]);
                        
                      }
                    }
                }
            }
        }
#endif          
        
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);

#ifdef DEBUG_OUTPUT
        checkCudaErrors (cudaDeviceSynchronize()); 
        if(false && strcmp(fname_bias, ip1_bias_bin) == 0) {
            printf("\n================================%s bias\n", fname_bias);    
            for(int i = 0; i < outputs; i++) {
                 printf("%7.4f ", (float)bias_h[i]);
            }
            printf("\n");
        }
#endif

    }

    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }

    private:

    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }
};

template <>
void Layer_t<half1>::readAllocInit(const char* fname, int size, half1** data_h, half1** data_d)
{
    *data_h = new half1[size];
    int size_b = size*sizeof(half1);
    checkCudaErrors( cudaMalloc(data_d, size_b) );    
    float *data_tmp_h, *data_tmp_d;

    switch(fp16Import)
    {
        case FP16_HOST :
        {
            readBinaryFile<half1>(fname, size, *data_h);
            checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b,
                                cudaMemcpyHostToDevice) );
            break;
        }
        case FP16_CUDA :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);

            gpu_float2half_rn<float>(size, data_tmp_d, *data_d);

            delete [] data_tmp_h;
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
        case FP16_CUDNN :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
            delete [] data_tmp_h;
            cudnnHandle_t cudnnHandle;
            cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
            checkCUDNN( cudnnCreate(&cudnnHandle) );
            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
                                                CUDNN_DATA_FLOAT,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
                                                CUDNN_DATA_HALF,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            float alpha = 1.0f;
            float beta = 0.0f;
            checkCUDNN( cudnnTransformTensor(cudnnHandle, &alpha,
                                             srcTensorDesc,
                                             data_tmp_d, &beta,
                                             dstTensorDesc,
                                             *data_d) );
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
            checkCUDNN( cudnnDestroy(cudnnHandle) );
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
    }
}

// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}

template <class value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;

    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }

    void destroyHandles()
    {
        checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCublasErrors( cublasDestroy(cublasHandle) );
    }

    public:

    network_t()
    {
        convAlgorithm = -1;
        switch (sizeof(value_type))
        {
            case 2 : dataType = CUDNN_DATA_HALF; break;
            case 4 : dataType = CUDNN_DATA_FLOAT; break;
            case 8 : dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };

    ~network_t()
    {
        destroyHandles();
    }

    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }

    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }

    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(1);
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }
    //(ip1, n = 1, c = 64, h = 12, w = 12, srcData[9216], &dstData);
    void fullyConnectedForward(const Layer_t<value_type>& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;  // 9216
        int dim_y = ip.outputs;  // 128
        resize(dim_y, dstData);

        scaling_type alpha = scaling_type(1), beta = scaling_type(1);

        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        
        // gemv(cublasHandle, dim_x = 9216, dim_y = 128, alpha = 1,
        //         ip.data_d[128][9216], srcData[9216], beta,*dstData);
        gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta,*dstData);

        h = 1; w = 1; c = dim_y;
    }
    //   convoluteForward(conv1, 1, 1, 28, 28, srcData, &dstData);
    void convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA) );
 
        const int convDims = 2;
        int padA[convDims] = {0,0};
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;

        // Math are done in FP32 when tensor are in FP16.
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT;
        }

        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType) );

        // outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;
        //           = 1 + ( 28 + 2*0 - (((3 - 1)*1) + 1) )/1
        //           = 1 + 28 - 3 
        //           = 26 
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        if (convAlgorithm < 0)
        {
            int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT; 
            int returnedAlgoCount = -1;
            cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

            // Choose the best according to the preference
            std::cout << "Testing cudnnGetConvolutionForwardAlgorithm_v7 ...\n";
            checkCUDNN( cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
                                                              srcTensorDesc,
                                                              filterDesc,
                                                              convDesc,
                                                              dstTensorDesc,
                                                              requestedAlgoCount,
                                                              &returnedAlgoCount,
                                                              results));
            for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
                printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
                    cudnnGetErrorString(results[algoIndex].status), 
                    results[algoIndex].algo, results[algoIndex].time, 
                    (unsigned long long)results[algoIndex].memory);
            }

            // New way of finding the fastest config
            // Setup for findFastest call
            std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
            checkCUDNN( cudnnFindConvolutionForwardAlgorithm(cudnnHandle, 
                                                            srcTensorDesc,
                                                            filterDesc,
                                                            convDesc,
                                                            dstTensorDesc,
                                                            requestedAlgoCount,
                                                            &returnedAlgoCount,
                                                            results));
            for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
                printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
                    cudnnGetErrorString(results[algoIndex].status), 
                    results[algoIndex].algo, results[algoIndex].time, 
                    (unsigned long long)results[algoIndex].memory);
            }
            
            algo = results[0].algo;            
        } else {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
        }

        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        std::cout << "GPU workspace size for conv: " << sizeInBytes << std::endl; // 0

        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData)
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                CUDNN_PROPAGATE_NAN,
                                                poolDims,
                                                windowDimA,
                                                paddingA,
                                                strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                    srcTensorDesc,
                                                    tensorDims,
                                                    tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
     
        resize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }

    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }

    void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }

    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }

    int classify_example(const char* fname, const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& ip1,
                          const Layer_t<value_type>& ip2)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        readImage(fname, imgData_h);

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        convoluteForward(conv1, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);
#ifdef DEBUG_OUTPUT        
        int i_max = 32;
        int j_max = 26;
        int k_mak = 26;
        int my_data_len = i_max*j_max*k_mak*sizeof(value_type);
        float * myData = (float*)malloc(my_data_len);
        cudaMemcpy(myData, srcData, my_data_len, cudaMemcpyDeviceToHost);

        for (int i = 0; i < i_max; i++){
            printf("\n----------After conv2d test_input_image_conv2ded[%d]:\n", i);
            for (int j = 0; j < j_max; j++) {
                for(int k = 0; k < k_mak; k++) {
                    float val_tmp = myData[i*j_max*k_mak + j * k_mak + k];
                    if(val_tmp != 0) {
                        printf("%5.1f ", val_tmp);
                    } else {
                        printf("    . ");
                    }
                }
                printf("\n");
            }
        }
        printf("before conv_1: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
#endif
        convoluteForward(conv2, n, c, h, w, srcData, &dstData);

#ifdef DEBUG_OUTPUT      
        checkCudaErrors (cudaDeviceSynchronize());
        printf("after conv_1: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
#endif
        activationForward(n, c, h, w, dstData, &srcData);


#ifdef DEBUG_OUTPUT  
        checkCudaErrors (cudaDeviceSynchronize());
        i_max = 64;
        j_max = 24;
        k_mak = 24;
        my_data_len = i_max*j_max*k_mak*sizeof(value_type);
        free(myData);
        myData = (float*)malloc(my_data_len);
        cudaMemcpy(myData, srcData, my_data_len, cudaMemcpyDeviceToHost);

        for (int i = 0; i < i_max; i++){
            printf("\nAfter conv2d_1 test_input_image_conv2d_1ed[%d]:\n", i);
            for (int j = 0; j < j_max; j++) {
                for(int k = 0; k < k_mak; k++) {
                    float val_tmp = myData[i*j_max*k_mak + j * k_mak + k];
                    if(val_tmp != 0) {
                        printf("%5.1f ", val_tmp);
                    } else {
                        printf("    . ");
                    }
                }
                printf("\n");
            }
        }
        printf("before max_poll: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
#endif
        poolForward(n, c, h, w, srcData, &dstData);

#ifdef DEBUG_OUTPUT
        printf("after max_poll: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
        checkCudaErrors (cudaDeviceSynchronize());
        i_max = 64;
        j_max = 12;
        k_mak = 12;
        my_data_len = i_max*j_max*k_mak*sizeof(value_type);
        free(myData);
        myData = (float*)malloc(my_data_len);
        cudaMemcpy(myData, dstData, my_data_len, cudaMemcpyDeviceToHost);

        for (int i = 0; i < i_max; i++){
            printf("\nAfter max_pooling [%d]:\n", i);
            for (int j = 0; j < j_max; j++) {
                for(int k = 0; k < k_mak; k++) {
                    float val_tmp = myData[i*j_max*k_mak + j * k_mak + k];
                    if(val_tmp != 0) {
                        printf("%5.1f ", val_tmp);
                    } else {
                        printf("    . ");
                    }
                }
                printf("\n");
            }
        }
#endif
        int data_len_tmp = 64*12*12*sizeof(value_type);
        float * myDataTmp_1 = (float*)malloc(data_len_tmp);
        float * myDataTmp_2 = (float*)malloc(data_len_tmp);
        cudaMemcpy(myDataTmp_1, dstData, data_len_tmp, cudaMemcpyDeviceToHost);

        for (int r = 0; r < 12; r++) {
          for (int c = 0; c < 12; c++) {
            for (int out_num = 0; out_num < 64; out_num++) {
              myDataTmp_2[r * 12 * 64 + c * 64 + out_num] =
                  myDataTmp_1[out_num * 12 * 12 + r * 12 + c];
            }
          }
        }
        cudaMemcpy(dstData, myDataTmp_2, data_len_tmp, cudaMemcpyHostToDevice);

#ifdef DEBUG_OUTPUT       
        checkCudaErrors (cudaDeviceSynchronize());
        printf("before fully connect: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
#endif
        //fullyConnectedForward(ip1, n = 1, c = 64, h = 12, w = 12, dstData[9216], &srcData);
        fullyConnectedForward(ip1, n, c, h, w, dstData, &srcData);

#ifdef DEBUG_OUTPUT
        printf("After fully connect: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
        checkCudaErrors (cudaDeviceSynchronize());
        i_max = 1;
        j_max = 1;
        k_mak = 128;
        my_data_len = i_max*j_max*k_mak*sizeof(value_type);
        free(myData);
        myData = (float*)malloc(my_data_len);
        cudaMemcpy(myData, srcData, my_data_len, cudaMemcpyDeviceToHost);

        for (int i = 0; i < i_max; i++){
            printf("\nAfter dense [%d]:\n", i);
            for (int j = 0; j < j_max; j++) {
                for(int k = 0; k < k_mak; k++) {
                    float val_tmp = myData[i*j_max*k_mak + j * k_mak + k];
                    if(val_tmp != 0) {
                        printf("%5.1f ", val_tmp);
                    } else {
                        printf("    . ");
                    }
                }
                printf("\n");
            }
        }
#endif
        activationForward(n, c, h, w, srcData, &dstData);

#ifdef DEBUG_OUTPUT
        checkCudaErrors (cudaDeviceSynchronize());
        i_max = 1;
        j_max = 1;
        k_mak = 128;
        my_data_len = i_max*j_max*k_mak*sizeof(value_type);
        free(myData);
        myData = (float*)malloc(my_data_len);
        cudaMemcpy(myData, dstData, my_data_len, cudaMemcpyDeviceToHost);

        for (int i = 0; i < i_max; i++){
            printf("\nAfter activation [%d]:\n", i);
            for (int j = 0; j < j_max; j++) {
                for(int k = 0; k < k_mak; k++) {
                    float val_tmp = myData[i*j_max*k_mak + j * k_mak + k];
                    if(val_tmp != 0) {
                        printf("%5.1f ", val_tmp);
                    } else {
                        printf("    . ");
                    }
                }
                printf("\n");
            }
        }
        printf("before fully connect 1: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
#endif
        fullyConnectedForward(ip2, n, c, h, w, dstData, &srcData);

#ifdef DEBUG_OUTPUT
        printf("After fully connect 1: n = %d; c = %d; h = %d; w = %d\n", n, c, h, w);
        checkCudaErrors (cudaDeviceSynchronize());
        i_max = 1;
        j_max = 1;
        k_mak = 10;
        my_data_len = i_max*j_max*k_mak*sizeof(value_type);
        free(myData);
        myData = (float*)malloc(my_data_len);
        cudaMemcpy(myData, srcData, my_data_len, cudaMemcpyDeviceToHost);

        for (int i = 0; i < i_max; i++){
            printf("\nAfter fully connect_1 [%d]:\n", i);
            for (int j = 0; j < j_max; j++) {
                for(int k = 0; k < k_mak; k++) {
                    float val_tmp = myData[i*j_max*k_mak + j * k_mak + k];
                    if(val_tmp != 0) {
                        printf("%5.1f ", val_tmp);
                    } else {
                        printf("    . ");
                    }
                }
                printf("\n");
            }
        }
#endif
        softmaxForward(n, c, h, w, srcData, &dstData);

        //cuDNN and cuBLAS library calls are asynchronous w.r.t. the host.
        // Need a device sync here before copying back the results.
        checkCudaErrors (cudaDeviceSynchronize());
        const int max_digits = 10;

        // Take care of half precision
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (Convert<scaling_type>(result[id]) < Convert<scaling_type>(result[i])) {
                id = i;
            }
        }

        std::cout << "Resulting weights from Softmax:" << std::endl;
        printDeviceVector(n*c*h*w, dstData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }
};

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
                          int& n, int& c, int& h, int& w,
                          half1* srcData, half1** dstData)
{
    c = c*h*w; h = 1; w = 1;
    network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData);
    c = ip.outputs;
}
#endif

static char * baseFile(char *fname) 
{
    char *base;
    for (base = fname; *fname != '\0'; fname++) {
        if (*fname == '/' || *fname == '\\') {
            base = fname + 1;
        }
    }
    return base;
}

static void displayUsage()
{
    printf( "mnistCUDNN {<options>}\n");
    printf( "help                   : display this help\n");
    printf( "device=<int>           : set the device to run the sample\n");
    printf( "image=<name>           : classify specific image\n");
}

int main(int argc, char *argv[])
{   
    printf("Executing: %s", baseFile(argv[0]));
    for (int i = 1; i < argc; i++) {
        printf(" %s", argv[i]);
    }
    printf("\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayUsage();
        exit(-1); 
    }

    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\n", COMPILER_NAME, COMPILER_VER);
    showDevices();

    int device = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        checkCudaErrors( cudaSetDevice(device) );
    }
    std::cout << "Using device " << device << std::endl;
    
    if (checkCmdLineFlag(argc, (const char **)argv, "image"))
    {
        getCmdLineArgumentString(argc, (const char **)argv,
                                 "image", (char **) &test_image);        
    }
    std::cout << "Testing image: " << test_image << std::endl;

    // check available memory
    struct cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
    double globalMem = prop.totalGlobalMem/double(1024*1024);
    bool low_memory = false;
    if (globalMem < 1536) 
    {
        // takes care of 1x1 convolution workaround for fully connected layers
        // when CUDNN_CONVOLUTION_FWD_ALGO_FFT is used
#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
        low_memory = true;
#endif
    }
    {
        std::cout << "\nTesting single precision\n";
        network_t<float> mnist;
        Layer_t<float> conv1(1,32,3,conv1_bin,conv1_bias_bin,argv[0]);
        Layer_t<float> conv2(32,64,3,conv2_bin,conv2_bias_bin,argv[0]);
        Layer_t<float>   ip1(9216,128,1,ip1_bin,ip1_bias_bin,argv[0]);
        Layer_t<float>   ip2(128,10,1,ip2_bin,ip2_bias_bin,argv[0]);
        std::string image_path;    
        get_path(image_path, test_image, argv[0]);
        int res = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        std::cout << "\nResult of classification: " << res << std::endl;
    }

    // {
    //     std::cout << "\nTesting half precision (math in single precision)\n";
    //     network_t<half1> mnist;

    //     // Conversion of input weights to half precision is done
    //     // on host using tools from fp16_emu.cpp
    //     Layer_t<half1> conv1(1,32,3,conv1_bin,conv1_bias_bin,argv[0],FP16_HOST);
    //     Layer_t<half1> conv2(32,64,3,conv2_bin,conv2_bias_bin,argv[0],FP16_HOST);

    //     // Conversion of input weights to half precision is done
    //     // on device using cudnnTransformTensor
    //     Layer_t<half1>   ip1(9216,128,1,ip1_bin,ip1_bias_bin,argv[0], FP16_CUDNN);

    //     // Conversion of input weights to half precision is done
    //     // on device using CUDA kernel from fp16_dev.cu
    //     Layer_t<half1>   ip2(128,10,1,ip2_bin,ip2_bias_bin,argv[0], FP16_CUDA);
    //     // get_path(image_path, first_image, argv[0]);
    //     // i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        
    //     get_path(image_path, second_image, argv[0]);
    //     i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        
    //     // get_path(image_path, third_image, argv[0]);

    //     // New feature in cuDNN v3: FFT for convolution
    //     // if (!low_memory)
    //     // {
    //     //     mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
    //     // }
    //     // i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

    //     std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
    //     if (i1 != 1 || i2 != 3 || i3 != 5)
    //     {
    //         std::cout << "\nTest failed!\n";
    //         FatalError("Prediction mismatch");
    //     }
    //     else
    //     {
    //         std::cout << "\nTest passed!\n";
    //     }
    // }

    cudaDeviceReset();
    exit(0);
}

