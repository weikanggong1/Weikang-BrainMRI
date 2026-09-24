// Isolated kernel experiment for FreeSurfer d932c45 mrisComputeIntensityTerm.
// The fixture contains a real sub-01 MRI and cortical mesh, with synthetic
// target intensities. This is not a drop-in replacement for mris_place_surface.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Input { float x, y, z, nx, ny, nz, target, sigma; };
struct Grad { float dx, dy, dz; };
struct Header { char magic[8]; uint32_t count, volume_bytes; };

static void check(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); }
}

// Same interpolation order and float source type as MRIsampleVolume(MRI_UCHAR).
__host__ __device__ static double sample(const uint8_t* vol, double x, double y, double z) {
  if (x < 0 || y < 0 || z < 0 || x >= 256 || y >= 256 || z >= 256) return 0;
  int xm = (int)x, ym = (int)y, zm = (int)z;
  int xp = xm < 255 ? xm + 1 : 255;
  int yp = ym < 255 ? ym + 1 : 255;
  int zp = zm < 255 ? zm + 1 : 255;
  double xmd = x - (float)xm, ymd = y - (float)ym, zmd = z - (float)zm;
  double xpd = 1.0f - xmd, ypd = 1.0f - ymd, zpd = 1.0f - zmd;
#define V(a,b,c) ((double)vol[((a)*256+(b))*256+(c)])
  double value = xpd*ypd*zpd*V(xm,ym,zm) + xpd*ypd*zmd*V(xm,ym,zp)
               + xpd*ymd*zpd*V(xm,yp,zm) + xpd*ymd*zmd*V(xm,yp,zp)
               + xmd*ypd*zpd*V(xp,ym,zm) + xmd*ypd*zmd*V(xp,ym,zp)
               + xmd*ymd*zpd*V(xp,yp,zm) + xmd*ymd*zmd*V(xp,yp,zp);
#undef V
  return value;
}

__host__ __device__ static Grad one(const uint8_t* vol, Input v) {
  // The fixture's exact vox2ras_tkr maps (x,y,z) -> (128-x,128-z,128+y).
  double center = sample(vol, 128.0-v.x, 128.0-v.z, 128.0+v.y);
  double sigma = v.sigma;
  if (fabs(sigma) < 1.1920929e-7) sigma = .25;
  double step = fmin(sigma / 2.0, .5);
  double outside = 0, inside = 0, kout = 0, kin = 0;
  for (double dist = step; dist <= 2.0*sigma; dist += step) {
    double k = exp(-dist*dist/(2.0*sigma*sigma));
    outside += k * sample(vol, 128.0-(v.x+dist*v.nx),
                              128.0-(v.z+dist*v.nz), 128.0+(v.y+dist*v.ny));
    inside += k * sample(vol, 128.0-(v.x-dist*v.nx),
                             128.0-(v.z-dist*v.nz), 128.0+(v.y-dist*v.ny));
    kout += k; kin += k;
  }
  if (kin > 0) inside /= kin;
  if (kout > 0) outside /= kout;
  double delta = fmax(-5.0, fmin(5.0, (double)v.target-center));
  double direction = (outside-inside)/2.0;
  direction = fabs(direction) >= 1.1920929e-7 ? direction/fabs(direction) : -1.0;
  double push = .2*delta*direction; // mris_place_surface default l_intensity
  return {(float)(v.nx*push), (float)(v.ny*push), (float)(v.nz*push)};
}

__global__ static void kernel(const uint8_t* vol, const Input* in, Grad* out, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) out[i] = one(vol, in[i]);
}

static double ms(std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b-a).count();
}

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s fixture.bin cuda_device_index|cpu\n", argv[0]); return 2; }
  std::ifstream file(argv[1], std::ios::binary);
  Header h{}; file.read((char*)&h, sizeof h);
  if (!file || std::string(h.magic, 8) != std::string("FSITERM\0", 8) ||
      h.volume_bytes != 256u*256u*256u) { fprintf(stderr, "invalid fixture\n"); return 2; }
  std::vector<uint8_t> vol(h.volume_bytes);
  std::vector<Input> in(h.count);
  std::vector<Grad> cpu(h.count), cpu4(h.count), gpu(h.count);
  file.read((char*)vol.data(), vol.size());
  file.read((char*)in.data(), in.size()*sizeof(Input));
  if (!file || sizeof(Input) != 32 || sizeof(Grad) != 12) { fprintf(stderr, "truncated fixture\n"); return 2; }

  auto a = std::chrono::steady_clock::now();
  for (unsigned i=0; i<h.count; i++) cpu[i] = one(vol.data(), in[i]);
  auto b = std::chrono::steady_clock::now();
  double cpu1 = ms(a,b);
  a = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(4)
  for (unsigned i=0; i<h.count; i++) cpu4[i] = one(vol.data(), in[i]);
  b = std::chrono::steady_clock::now();
  double cpu4_ms = ms(a,b);
  if (!strcmp(argv[2], "cpu")) {
    unsigned different=0;
    for (unsigned i=0; i<h.count; i++) {
      const float* x=(const float*)&cpu[i]; const float* y=(const float*)&cpu4[i];
      for (int c=0;c<3;c++) different += x[c]!=y[c];
    }
    printf("{\"vertices\":%u,\"cpu1_ms\":%.6f,\"cpu4_ms\":%.6f,"
           "\"cpu4_different_values\":%u}\n",h.count,cpu1,cpu4_ms,different);
    return 0;
  }

  check(cudaSetDevice(atoi(argv[2])));
  a = std::chrono::steady_clock::now();
  check(cudaFree(nullptr));
  b = std::chrono::steady_clock::now();
  double context = ms(a,b);
  uint8_t* dv=nullptr; Input* di=nullptr; Grad* do_=nullptr;
  a = std::chrono::steady_clock::now();
  check(cudaMalloc(&dv,vol.size()));
  check(cudaMalloc(&di,in.size()*sizeof(Input)));
  check(cudaMalloc(&do_,gpu.size()*sizeof(Grad)));
  check(cudaMemcpy(dv,vol.data(),vol.size(),cudaMemcpyHostToDevice));
  check(cudaMemcpy(di,in.data(),in.size()*sizeof(Input),cudaMemcpyHostToDevice));
  cudaEvent_t start, stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
  check(cudaEventRecord(start));
  kernel<<<(h.count+255)/256,256>>>(dv,di,do_,h.count);
  check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
  float kernel_ms=0; check(cudaEventElapsedTime(&kernel_ms,start,stop));
  check(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(Grad),cudaMemcpyDeviceToHost));
  b = std::chrono::steady_clock::now();
  double end_to_end = ms(a,b);
  double max_abs=0, mean_abs=0; unsigned different=0, cpu4_different=0;
  for (unsigned i=0; i<h.count; i++) {
    const float* x=(const float*)&cpu[i]; const float* y=(const float*)&gpu[i];
    for (int c=0;c<3;c++) {
      double diff=fabs((double)x[c]-y[c]);
      max_abs=std::max(max_abs,diff); mean_abs+=diff;
      different += x[c]!=y[c];
      cpu4_different += x[c]!=((const float*)&cpu4[i])[c];
    }
  }
  mean_abs /= (3.0*h.count);
  printf("{\"vertices\":%u,\"cpu1_ms\":%.6f,\"cpu4_ms\":%.6f,\"cuda_context_ms\":%.6f,"
         "\"cuda_kernel_ms\":%.6f,\"cuda_end_to_end_ms\":%.6f,"
         "\"max_abs_gradient\":%.9g,\"mean_abs_gradient\":%.9g,"
         "\"different_float32_values\":%u,\"cpu4_different_values\":%u}\n",
         h.count,cpu1,cpu4_ms,context,kernel_ms,end_to_end,max_abs,mean_abs,different,cpu4_different);
  check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
  check(cudaFree(dv)); check(cudaFree(di)); check(cudaFree(do_));
  return 0;
}
