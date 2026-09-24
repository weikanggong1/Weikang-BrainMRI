#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

struct Header {
  char magic[8];
  uint32_t vertices, neighbors, iterations, original_vertices;
};
static_assert(sizeof(Header) == 24, "unexpected binary header layout");

#define CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) \
  throw std::runtime_error(cudaGetErrorString(err)); } while (0)

template <typename T>
void read_array(std::ifstream& file, std::vector<T>& values) {
  if (!file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T)))
    throw std::runtime_error("truncated case file");
}

struct Case {
  Header header;
  std::vector<uint32_t> vertex_ids, offsets, indices;
  std::vector<float> values;
};

Case read_case(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open case file");
  Case input;
  if (!file.read(reinterpret_cast<char*>(&input.header), sizeof(Header)) ||
      std::memcmp(input.header.magic, "FSGRAD1\0", 8) != 0)
    throw std::runtime_error("invalid case header");
  const auto n = input.header.vertices;
  const auto m = input.header.neighbors;
  if (n == 0 || n > input.header.original_vertices || m > n * 128ULL)
    throw std::runtime_error("invalid case dimensions");
  input.vertex_ids.resize(n);
  input.offsets.resize(n + 1);
  input.indices.resize(m);
  input.values.resize(3ULL * n);
  read_array(file, input.vertex_ids);
  read_array(file, input.offsets);
  read_array(file, input.indices);
  read_array(file, input.values);
  if (input.offsets[0] != 0 || input.offsets[n] != m)
    throw std::runtime_error("invalid CSR offsets");
  for (uint32_t i = 0; i < n; ++i)
    if (input.offsets[i] > input.offsets[i + 1])
      throw std::runtime_error("unordered CSR offsets");
  for (auto index : input.indices)
    if (index >= n) throw std::runtime_error("invalid CSR index");
  return input;
}

std::vector<float> smooth_cpu(const Case& input, int threads) {
  auto current = input.values;
  std::vector<float> next(current.size());
  for (uint32_t step = 0; step < input.header.iterations; ++step) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t v = 0; v < input.header.vertices; ++v) {
      const uint32_t vertex = static_cast<uint32_t>(v);
      float x = current[3ULL * vertex], y = current[3ULL * vertex + 1], z = current[3ULL * vertex + 2];
      for (uint32_t edge = input.offsets[vertex]; edge < input.offsets[vertex + 1]; ++edge) {
        const uint32_t adjacent = input.indices[edge];
        x += current[3ULL * adjacent];
        y += current[3ULL * adjacent + 1];
        z += current[3ULL * adjacent + 2];
      }
      const float scale = 1.0f / float(input.offsets[vertex + 1] - input.offsets[vertex] + 1);
      next[3ULL * vertex] = x * scale;
      next[3ULL * vertex + 1] = y * scale;
      next[3ULL * vertex + 2] = z * scale;
    }
    current.swap(next);
  }
  return current;
}

__global__ void smooth_kernel(const uint32_t* offsets, const uint32_t* indices,
                              const float* current, float* next, uint32_t n) {
  const uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= n) return;
  float x = current[3ULL * vertex], y = current[3ULL * vertex + 1], z = current[3ULL * vertex + 2];
  for (uint32_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
    const uint32_t adjacent = indices[edge];
    x += current[3ULL * adjacent];
    y += current[3ULL * adjacent + 1];
    z += current[3ULL * adjacent + 2];
  }
  const float scale = 1.0f / float(offsets[vertex + 1] - offsets[vertex] + 1);
  next[3ULL * vertex] = x * scale;
  next[3ULL * vertex + 1] = y * scale;
  next[3ULL * vertex + 2] = z * scale;
}

struct GpuResult { std::vector<float> values; double total_ms, kernel_ms; };

GpuResult smooth_gpu(const Case& input) {
  CUDA(cudaFree(nullptr));  // initialize CUDA before wall timing
  const auto start = std::chrono::steady_clock::now();
  uint32_t *offsets = nullptr, *indices = nullptr;
  float *current = nullptr, *next = nullptr;
  const size_t value_bytes = input.values.size() * sizeof(float);
  CUDA(cudaMalloc(&offsets, input.offsets.size() * sizeof(uint32_t)));
  CUDA(cudaMalloc(&indices, input.indices.size() * sizeof(uint32_t)));
  CUDA(cudaMalloc(&current, value_bytes));
  CUDA(cudaMalloc(&next, value_bytes));
  CUDA(cudaMemcpy(offsets, input.offsets.data(), input.offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(indices, input.indices.data(), input.indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(current, input.values.data(), value_bytes, cudaMemcpyHostToDevice));
  cudaEvent_t first, last;
  CUDA(cudaEventCreate(&first));
  CUDA(cudaEventCreate(&last));
  CUDA(cudaEventRecord(first));
  for (uint32_t step = 0; step < input.header.iterations; ++step) {
    smooth_kernel<<<(input.header.vertices + 255) / 256, 256>>>(
        offsets, indices, current, next, input.header.vertices);
    std::swap(current, next);
  }
  CUDA(cudaGetLastError());
  CUDA(cudaEventRecord(last));
  CUDA(cudaEventSynchronize(last));
  float kernel_ms = 0;
  CUDA(cudaEventElapsedTime(&kernel_ms, first, last));
  std::vector<float> output(input.values.size());
  CUDA(cudaMemcpy(output.data(), current, value_bytes, cudaMemcpyDeviceToHost));
  const auto stop = std::chrono::steady_clock::now();
  CUDA(cudaEventDestroy(first));
  CUDA(cudaEventDestroy(last));
  CUDA(cudaFree(offsets)); CUDA(cudaFree(indices)); CUDA(cudaFree(current)); CUDA(cudaFree(next));
  return {std::move(output), std::chrono::duration<double, std::milli>(stop - start).count(), kernel_ms};
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: smooth CASE.bin");
    const Case input = read_case(argv[1]);
    const auto start1 = std::chrono::steady_clock::now();
    const auto cpu1 = smooth_cpu(input, 1);
    const auto stop1 = std::chrono::steady_clock::now();
    const auto cpu4 = smooth_cpu(input, 4);
    const auto stop4 = std::chrono::steady_clock::now();
    const double cpu1_ms = std::chrono::duration<double, std::milli>(stop1 - start1).count();
    const double cpu4_ms = std::chrono::duration<double, std::milli>(stop4 - stop1).count();
    const auto gpu = smooth_gpu(input);
    double max_abs = 0, mean_abs = 0;
    uint64_t different = 0, cpu1_cpu4_different = 0;
    size_t worst = 0;
    for (size_t i = 0; i < cpu4.size(); ++i) {
      const double diff = std::abs(double(cpu4[i]) - double(gpu.values[i]));
      mean_abs += diff;
      if (diff > max_abs) { max_abs = diff; worst = i; }
      if (std::memcmp(&cpu4[i], &gpu.values[i], sizeof(float)) != 0) ++different;
      if (std::memcmp(&cpu1[i], &cpu4[i], sizeof(float)) != 0) ++cpu1_cpu4_different;
    }
    mean_abs /= cpu4.size();
    std::cout << "{\"vertices\":" << input.header.vertices
              << ",\"neighbors\":" << input.header.neighbors
              << ",\"iterations\":" << input.header.iterations
              << ",\"cpu1_ms\":" << cpu1_ms
              << ",\"cpu4_ms\":" << cpu4_ms
              << ",\"cuda_total_ms\":" << gpu.total_ms
              << ",\"cuda_kernel_ms\":" << gpu.kernel_ms
              << ",\"speedup_vs_cpu4_including_transfer\":" << cpu4_ms / gpu.total_ms
              << ",\"max_abs\":" << max_abs
              << ",\"mean_abs\":" << mean_abs
              << ",\"different_float32_values\":" << different
              << ",\"cpu1_cpu4_different\":" << cpu1_cpu4_different
              << ",\"worst_original_vertex\":" << input.vertex_ids[worst / 3]
              << ",\"worst_component\":" << worst % 3 << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
