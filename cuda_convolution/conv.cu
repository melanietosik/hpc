#include <cudnn.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#define H 4096
#define W 4096
#define C 3
#define FH 3
#define FW 3
#define K 10
#define P 1   // Padding
#define N 16  // Block size

#define I_size (C * W * H)
#define F_size (K * C * FH * FW)
#define O_size (K * W * H)

#define idx3(c,x,y) ((c)*(H*W) + (x)*W + (y))
#define idx4(k,c,i,j) (k*(C*FH*FW) + c*(FH*FW) + (i)*FW + (j))  // Brackets are life savers

#define CHECK(expression)                                  \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

/*
  Usage (on HPC):

    module purge
    module load cuda/9.0.176
    module load cudnn/9.0v7.0.5
    make && ./conv
*/

__global__ void convolutionKernel(double* I, double* F, double* O) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z;

  // Skip computation if thread is outside boundaries
  if (k >= K || col >= W || row >= H) return;
    
  int x = col - P; // x and y correspond to I0 dimensions
  int y = row - P;

  double pixel = 0.0;

  for (int c=0; c<C; c++) {
    for (int j=0; j<FH; j++) {
      for (int i=0; i<FW; i++) {

        int cur_row = x + i;
        int cur_col = y + j;

        // Implicit handling of I0 padding
        double I0_val;
        if (cur_row == -1 || cur_row == W || cur_col == -1 || cur_col == H)
          I0_val = 0;
        else
          I0_val = I[idx3(c, cur_row, cur_col)];  // Index lookup

        // Compute and update output value
        pixel += (F[idx4(k, c, FW - 1 - i, FH - 1 - j)] * I0_val);
      }
    }
  }
  // Add computed pixel value to output
  O[idx3(k,col,row)] = pixel;
}

double* init_I(double* I) {
  /*
    Initialize input tensor I
    :: I[c, x, y] = c * (x + y)
  */
  int c, x, y;
  for (c=0; c<C; c++) {
    for (x=0; x<H; x++) {
      for (y=0; y<W; y++) {
        I[idx3(c, x, y)] = c * (x + y); 
  }}}
  return I;
}

double* init_F(double* F) {
  /*
    Initialize filter tensor I
    :: F[k, c, i, j] = (c + k) * (i + j);
  */
  int k, c, i, j;
  for (k=0; k<K; k++) {
    for (c=0; c<C; c++) {
      for (i=0; i<FW; i++) {
        for (j=0; j<FH; j++) {
          F[idx4(k,c,i,j)] = (c + k) * (i + j);
  }}}}
  return F;
}

void cuDNNConvolution(double* d_I, double* d_F, double* d_O) {
  /*
    Convolution in cuDNN
    cf. http://www.goldsborough.me/cuda/ml/cudnn/c++/
        2017/10/01/14-37-23-convolutions_with_cudnn/
  */
  cudnnHandle_t cudnn;
  CHECK(cudnnCreate(&cudnn));

  // I
  cudnnTensorDescriptor_t I_descriptor;
  CHECK(cudnnCreateTensorDescriptor(&I_descriptor));
  CHECK(cudnnSetTensor4dDescriptor(I_descriptor,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

  // F
  cudnnFilterDescriptor_t F_descriptor;
  CHECK(cudnnCreateFilterDescriptor(&F_descriptor));
  CHECK(cudnnSetFilter4dDescriptor(F_descriptor,
    CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));

  // O
  cudnnTensorDescriptor_t O_descriptor;
  CHECK(cudnnCreateTensorDescriptor(&O_descriptor));
  CHECK(cudnnSetTensor4dDescriptor(O_descriptor,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

  // C
  cudnnConvolutionDescriptor_t CONV_descriptor;
  CHECK(cudnnCreateConvolutionDescriptor(&CONV_descriptor));
  CHECK(cudnnSetConvolution2dDescriptor(CONV_descriptor,
    P,  // Padding height
    P,  // Padding width
    1,  // Vertical stride
    1,  // Horizontal stride
    1,  // Dilation height
    1,  // Dilation width
    // CUDNN_CROSS_CORRELATION,
    CUDNN_CONVOLUTION,
    CUDNN_DATA_DOUBLE));

  // Convolution algorithm
  cudnnConvolutionFwdAlgo_t CONV_algorithm;
  CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn,
    I_descriptor, F_descriptor, CONV_descriptor, O_descriptor,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0, &CONV_algorithm));

  // Workspace specification
  size_t workspace_bytes = 0;
  CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
    I_descriptor, F_descriptor, CONV_descriptor, O_descriptor,
    CONV_algorithm, &workspace_bytes));

  void* d_w;
  cudaMalloc(&d_w, sizeof(double)*workspace_bytes);

  double alpha = 1.;
  double beta = 0.;  // Not ResNet

  // Compute convolution
  CHECK(cudnnConvolutionForward(cudnn,
    &alpha,
    I_descriptor, d_I,
    F_descriptor, d_F,
    CONV_descriptor,
    CONV_algorithm,
    d_w,
    workspace_bytes,
    &beta,
    O_descriptor, d_O));

  cudaFree(d_w);
  CHECK(cudnnDestroyTensorDescriptor(I_descriptor));
  CHECK(cudnnDestroyTensorDescriptor(O_descriptor));
  CHECK(cudnnDestroyFilterDescriptor(F_descriptor));
  CHECK(cudnnDestroyConvolutionDescriptor(CONV_descriptor));
  CHECK(cudnnDestroy(cudnn));
}

int main(void) {
  /*
    Main function
  */
  double* I = (double*) malloc(I_size * sizeof(double));
  double* F = (double*) malloc(F_size * sizeof(double));
  double* O = (double*) malloc(O_size * sizeof(double));
  init_I(I);
  init_F(F);
  double* d_I;  // Device I
  double* d_F;  // Device F
  double* d_O;  // Device O

  cudaError_t malloc_I = cudaMalloc(&d_I, I_size * sizeof(double)); 
  cudaError_t malloc_F = cudaMalloc(&d_F, F_size * sizeof(double));
  cudaError_t malloc_O = cudaMalloc(&d_O, O_size * sizeof(double));
  if (malloc_I != cudaSuccess) throw std::runtime_error("Failed to allocate memory: I");
  if (malloc_F != cudaSuccess) throw std::runtime_error("Failed to allocate memory: F");
  if (malloc_O != cudaSuccess) throw std::runtime_error("Failed to allocate memory: O");

  cudaError_t cp_I = cudaMemcpy(d_I, I, I_size * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t cp_F = cudaMemcpy(d_F, F, F_size * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t cp_O = cudaMemcpy(d_O, O, O_size * sizeof(double), cudaMemcpyHostToDevice);
  if (cp_I != cudaSuccess) throw std::runtime_error("Failed to copy to host memory: I");
  if (cp_F != cudaSuccess) throw std::runtime_error("Failed to copy to host memory: F");
  if (cp_O != cudaSuccess) throw std::runtime_error("Failed to copy to host memory: O");

  /* Accounting
     cf. https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
  */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double checksum;
  float milliseconds;

  /* CUDA simple convolution kernel */
  dim3 threadsPerBlock(N, N);
  dim3 numBlocks(W/threadsPerBlock.x, H/threadsPerBlock.y, K);

  cudaEventRecord(start);  // Start time
  convolutionKernel<<<numBlocks, threadsPerBlock>>>(d_I, d_F, d_O);
  cudaEventRecord(stop);  // End time

  cudaMemcpy(O, d_O, O_size * sizeof(double), cudaMemcpyDeviceToHost);

  // Get execution time
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Compute checksum
  checksum = 0;
  for (int i=0; i<O_size; i++) {
    checksum += O[i];
  }
  printf("%.2f,%4.3lf\n", checksum, milliseconds);

  /* cuDNN convolution */
  memset(O, 0, O_size * sizeof(double));  // Zero-out O

  cp_O = cudaMemcpy(d_O, O, O_size * sizeof(double), cudaMemcpyHostToDevice);
  if (malloc_O != cudaSuccess) throw std::runtime_error("Failed to allocate memory: O");
  if (cp_O != cudaSuccess) throw std::runtime_error("Failed to copy to host memory: O");

  cudaEventRecord(start);  // Start time
  cuDNNConvolution(d_I, d_F, d_O);
  cudaEventRecord(stop);  // End time

  cudaMemcpy(O, d_O, O_size * sizeof(double), cudaMemcpyDeviceToHost);

  // Get execution time
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Compute checksum
  checksum = 0;
  for (int i=0; i<O_size; i++) {
    checksum += O[i];
  }
  printf("%.2f,%4.3lf\n", checksum, milliseconds);

  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_O);
}

/*
228686907676500.00,175.743
228686907676500.00,849.415
*/
