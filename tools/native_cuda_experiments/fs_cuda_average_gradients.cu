#include <cuda_runtime.h>

#include <cstddef>

// ABI of MRISaverageGradients_Control and MRISaverageGradients_Data in
// FreeSurfer d932c45. Keep this interface C-only: the CUDA and FreeSurfer
// translation units can then use different host C++ compilers.
struct Control {
  int num_neighbors;
  int first_neighbor;
};
struct Data {
  float dx, dy, dz;
};
static_assert(sizeof(Control) == 2 * sizeof(int), "unexpected control layout");
static_assert(sizeof(Data) == 3 * sizeof(float), "unexpected gradient layout");

__global__ void average_gradients(const Control* controls, const int* neighbors,
                                  const Data* input, Data* output, int count) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= count) return;

  const Control c = controls[vertex];
  float dx = input[vertex].dx;
  float dy = input[vertex].dy;
  float dz = input[vertex].dz;
  for (int n = 0; n < c.num_neighbors; ++n) {
    const Data next = input[neighbors[c.first_neighbor + n]];
    dx += next.dx;
    dy += next.dy;
    dz += next.dz;
  }
  const float inv_num = 1.0f / (c.num_neighbors + 1);
  output[vertex].dx = dx * inv_num;
  output[vertex].dy = dy * inv_num;
  output[vertex].dz = dz * inv_num;
}

// Return 0 only after a complete device-to-host copy to output. A failed
// attempt leaves input intact so the caller can run FreeSurfer's CPU loop.
extern "C" int fs_cuda_average_gradients(int count, int neighbor_count,
                                         int iterations, const void* controls,
                                         const int* neighbors, const void* input,
                                         void* output) {
  if (count <= 0 || neighbor_count <= 0 || iterations <= 0 ||
      !controls || !neighbors || !input || !output) return -1;

  Control* device_controls = nullptr;
  int* device_neighbors = nullptr;
  Data* device_input = nullptr;
  Data* device_output = nullptr;
  cudaError_t error = cudaSuccess;
  const size_t control_bytes = size_t(count) * sizeof(Control);
  const size_t neighbor_bytes = size_t(neighbor_count) * sizeof(int);
  const size_t data_bytes = size_t(count) * sizeof(Data);

#define TRY_CUDA(call) do { error = (call); if (error != cudaSuccess) goto done; } while (0)
  TRY_CUDA(cudaMalloc(&device_controls, control_bytes));
  TRY_CUDA(cudaMalloc(&device_neighbors, neighbor_bytes));
  TRY_CUDA(cudaMalloc(&device_input, data_bytes));
  TRY_CUDA(cudaMalloc(&device_output, data_bytes));
  TRY_CUDA(cudaMemcpy(device_controls, controls, control_bytes, cudaMemcpyHostToDevice));
  TRY_CUDA(cudaMemcpy(device_neighbors, neighbors, neighbor_bytes, cudaMemcpyHostToDevice));
  TRY_CUDA(cudaMemcpy(device_input, input, data_bytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < iterations; ++i) {
    average_gradients<<<(count + 255) / 256, 256>>>(
        device_controls, device_neighbors, device_input, device_output, count);
    Data* swap = device_input;
    device_input = device_output;
    device_output = swap;
  }
  TRY_CUDA(cudaGetLastError());
  TRY_CUDA(cudaMemcpy(output, device_input, data_bytes, cudaMemcpyDeviceToHost));

done:
  cudaFree(device_controls);
  cudaFree(device_neighbors);
  cudaFree(device_input);
  cudaFree(device_output);
#undef TRY_CUDA
  return int(error);
}
