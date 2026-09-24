#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Must match fs_cuda_average_gradients.cu and FreeSurfer's Control/Data layout.
struct Control { int num_neighbors, first_neighbor; };
struct Data { float dx, dy, dz; };
static_assert(sizeof(Control) == 8 && sizeof(Data) == 12, "unexpected ABI layout");
extern "C" int fs_cuda_average_gradients(int count, int neighbor_count,
                                           int iterations, const void* controls,
                                           const int* neighbors, const void* input,
                                           void* output);

struct Header {
  char magic[8];
  uint32_t vertices, neighbors, iterations, original_vertices;
};
static_assert(sizeof(Header) == 24, "unexpected case header layout");

template <typename T>
void read_array(std::ifstream& file, std::vector<T>& values) {
  if (!file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T)))
    throw std::runtime_error("truncated case file");
}

struct Case {
  Header header;
  std::vector<uint32_t> vertex_ids, offsets, indices;
  std::vector<Data> values;
};

Case read_case(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open case file");
  Case input;
  if (!file.read(reinterpret_cast<char*>(&input.header), sizeof(Header)) ||
      std::memcmp(input.header.magic, "FSGRAD1\0", 8) != 0)
    throw std::runtime_error("invalid case header");
  const uint32_t n = input.header.vertices, m = input.header.neighbors;
  if (n == 0 || n > INT_MAX || n > input.header.original_vertices ||
      m == 0 || m > INT_MAX || m > uint64_t(n) * 128 ||
      input.header.iterations == 0 || input.header.iterations > INT_MAX)
    throw std::runtime_error("invalid case dimensions");
  input.vertex_ids.resize(n);
  input.offsets.resize(uint64_t(n) + 1);
  input.indices.resize(m);
  input.values.resize(n);
  read_array(file, input.vertex_ids);
  read_array(file, input.offsets);
  read_array(file, input.indices);
  read_array(file, input.values);
  if (input.offsets[0] != 0 || input.offsets[n] != m)
    throw std::runtime_error("invalid CSR offsets");
  for (uint32_t i = 0; i < n; ++i)
    if (input.offsets[i] > input.offsets[i + 1])
      throw std::runtime_error("unordered CSR offsets");
  for (uint32_t index : input.indices)
    if (index >= n) throw std::runtime_error("invalid CSR index");
  return input;
}

std::vector<Data> average_cpu(const Case& input, const std::vector<Control>& controls,
                              const std::vector<int>& neighbors) {
  std::vector<Data> current = input.values, next(current.size());
  for (uint32_t step = 0; step < input.header.iterations; ++step) {
    for (size_t vertex = 0; vertex < current.size(); ++vertex) {
      const Control c = controls[vertex];
      float dx = current[vertex].dx, dy = current[vertex].dy, dz = current[vertex].dz;
      for (int k = 0; k < c.num_neighbors; ++k) {
        const Data d = current[neighbors[c.first_neighbor + k]];
        dx += d.dx; dy += d.dy; dz += d.dz;
      }
      const float inv_num = 1.0f / (c.num_neighbors + 1);
      next[vertex] = {dx * inv_num, dy * inv_num, dz * inv_num};
    }
    current.swap(next);
  }
  return current;
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: bridge_fixture CASE.bin");
    const Case input = read_case(argv[1]);
    std::vector<Control> controls(input.header.vertices);
    std::vector<int> neighbors(input.indices.begin(), input.indices.end());
    for (uint32_t vertex = 0; vertex < input.header.vertices; ++vertex) {
      controls[vertex] = {int(input.offsets[vertex + 1] - input.offsets[vertex]),
                          int(input.offsets[vertex])};
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    const auto expected = average_cpu(input, controls, neighbors);
    const auto cpu_stop = std::chrono::steady_clock::now();
    std::vector<Data> observed(expected.size());
    const auto context_start = std::chrono::steady_clock::now();
    if (cudaFree(nullptr) != cudaSuccess)
      throw std::runtime_error("CUDA context initialization failed");
    const auto context_stop = std::chrono::steady_clock::now();
    const auto cuda_start = std::chrono::steady_clock::now();
    const int status = fs_cuda_average_gradients(
        int(input.header.vertices), int(input.header.neighbors),
        int(input.header.iterations), controls.data(), neighbors.data(),
        input.values.data(), observed.data());
    const auto cuda_stop = std::chrono::steady_clock::now();
    if (status != 0) throw std::runtime_error("CUDA bridge returned status " + std::to_string(status));

    uint64_t mismatches = 0;
    double max_abs = 0, mean_abs = 0;
    size_t worst = 0;
    for (size_t vertex = 0; vertex < expected.size(); ++vertex) {
      const float actual[3] = {observed[vertex].dx, observed[vertex].dy, observed[vertex].dz};
      const float reference[3] = {expected[vertex].dx, expected[vertex].dy, expected[vertex].dz};
      for (size_t component = 0; component < 3; ++component) {
        if (!std::isfinite(actual[component]) || !std::isfinite(reference[component]))
          throw std::runtime_error("nonfinite output");
        const double difference = std::abs(double(actual[component]) - double(reference[component]));
        mean_abs += difference;
        if (difference > max_abs) { max_abs = difference; worst = 3 * vertex + component; }
        if (std::memcmp(&actual[component], &reference[component], sizeof(float)) != 0) ++mismatches;
      }
    }
    mean_abs /= expected.size() * 3;
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
    const double context_ms = std::chrono::duration<double, std::milli>(context_stop - context_start).count();
    const double cuda_ms = std::chrono::duration<double, std::milli>(cuda_stop - cuda_start).count();
    std::cout << "{\"vertices\":" << input.header.vertices
              << ",\"neighbors\":" << input.header.neighbors
              << ",\"iterations\":" << input.header.iterations
              << ",\"cpu_reference_ms\":" << cpu_ms
              << ",\"cuda_context_init_ms\":" << context_ms
              << ",\"cuda_call_total_ms\":" << cuda_ms
              << ",\"cuda_first_call_total_ms\":" << context_ms + cuda_ms
              << ",\"speedup_vs_cpu_reference\":" << cpu_ms / cuda_ms
              << ",\"max_abs\":" << max_abs
              << ",\"mean_abs\":" << mean_abs
              << ",\"different_float32_values\":" << mismatches
              << ",\"worst_original_vertex\":" << input.vertex_ids[worst / 3]
              << ",\"worst_component\":" << worst % 3 << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
