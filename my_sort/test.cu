#include <cuda.h>
#include "cuda_runtime_api.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cstdio>
#include "/home/zht/document/CProj/cudaC/CUDA_Freshman/include/freshman.h"


#define tile_size 256u
#define log_tile_size 8u
#define DBL_MAX 1.79769e+308


 template<typename type>__device__ void swap(type& v1, type& v2)
{
	type t = v1;
	v1 = v2;
	v2 = t;
}
template<typename type>__device__ unsigned int lower_bound(const type* data, unsigned int n, const type& value)
{
 unsigned int len = n, half;
 const type* start = data, * mid;
 while (len)
 {
  half = len >> 1;
  mid = data + half;
  if (*mid < value)
  {
   data = mid + 1;
   len = len - half - 1;
  }
  else
   len = half;
 }
 return data - start;
}
template<typename type>__device__ unsigned int upper_bound(const type* data, unsigned int n, const type& value)
{
 unsigned int len = n, half;
 const type* start = data, *mid;
 while (len)
 {
  half = len >> 1;
  mid = data + half;
  if (value < *mid)
   len = half;
  else
  {
   data = mid + 1;
   len = len - half - 1;
  }
 }
 return data - start;
}

__global__ void binoticsort_gpu(double* data, int n)
{
	__shared__  double buffer[tile_size << 1];

	unsigned int d = blockIdx.x << (log_tile_size + 1), i = threadIdx.x;
	data += d; n -= d;
	buffer[i] = i < n ? data[i] : DBL_MAX;
	buffer[i + blockDim.x] = i + blockDim.x < n ? data[i + blockDim.x] : DBL_MAX;
	__syncthreads();

	unsigned int s, t;
	for (unsigned int k = 2; k <= blockDim.x * 2; k <<= 1)
		for (unsigned int p = k >> 1; p; p >>= 1)
		{
			s = (i & -p) << 1 | i & p - 1;
			t = s | p;

			if (s & k ? buffer[s] < buffer[t] : buffer[s] > buffer[t])
				swap(buffer[s], buffer[t]);
			__syncthreads();
		}


	if (i < n)
		data[i] = buffer[i];
	if (i + blockDim.x < n)
		data[i + blockDim.x] = buffer[i + blockDim.x];
}
// merge directly
__global__ void mergedirect_gpu(const double* data, unsigned int n, unsigned int s, double* result)
{
 unsigned int d1 = (blockIdx.z * gridDim.y + blockIdx.y) * s << 1, d2 = d1 + s, i = d1 + blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n)
  result[d2 < n ? i + lower_bound(data + d2, min(s, n - d2), data[i]) : i] = data[i];
 if (s + i < n)
  result[i + upper_bound(data + d1, s, data[s + i])] = data[s + i];
}
double* sortdirect_gpu(double* data, unsigned int n)
{
 double* tmp;
 cudaMalloc(&tmp, sizeof(double) * n);
 unsigned int b = n + (tile_size << 1) - 1 >> log_tile_size + 1, s = tile_size << 1;
 binoticsort_gpu<<<b, s >> 1>>>(data, n);
 for (b = b + 1 >> 1; s < n; s <<= 1, b = b + 1 >> 1)
 {
  mergedirect_gpu<<<dim3(s >> log_tile_size, std::min(b, 32768u), b + 32767 >> 15), tile_size>>>(data, n, s, tmp);
  std::swap(data, tmp);
 }
 cudaDeviceSynchronize();
 cudaFree(tmp);
 
 return data;
}

int main(int argc, char** argv)
{
 cudaSetDevice(0);
 cudaFree(0);
 unsigned int test[] = { 1025u, 65537u, 1048577u, 16777217u, 134217729u, 1073741825u };
 const unsigned int runtime = 3;
 
 double iStart, iElaps;
 double elasped[3] = { 0 };
 
 for (unsigned int t = 0; t < sizeof(test) / sizeof(unsigned int); ++t)
 {
  unsigned int n = test[t];
  double* original = new double[n];
  double* data = new double[n];
 
  elasped[0] = elasped[1] = elasped[2] = 0;
  srand(time(nullptr));
  for (int i = 0; i < runtime; ++i)
  {
   for (unsigned int i = 0; i < n; ++i)
    original[i] = data[i] = (rand() * 2.0 / RAND_MAX - 1.0) * rand();
 
   // cpu sort 1
iStart=cpuSecond();
   qsort(data, n, sizeof(double), [](const void* left, const void* right)->int {double tmp = *(double*)left - *(double*)right; if (tmp > 0) return 1; else if (tmp < 0) return -1; else return 0; });
    iElaps=cpuSecond()-iStart;
   elasped[0] += iElaps;
 
   // cpu sort 2
   memcpy(data, original, sizeof(double) * n);
iStart=cpuSecond();
   std::sort(data, data + n);
    iElaps=cpuSecond()-iStart;
   elasped[1] += iElaps;
 
 
   // gpu sort
   double* pd_data;
   cudaMalloc(&pd_data, sizeof(double) * n);
   cudaMemcpy(pd_data, original, sizeof(double) * n, cudaMemcpyHostToDevice);

iStart=cpuSecond();
   pd_data = sortdirect_gpu(pd_data, n);
    iElaps=cpuSecond()-iStart;
   elasped[2] += iElaps;

   cudaMemcpy(data, pd_data, sizeof(double) * n, cudaMemcpyDeviceToHost);
   cudaFree(pd_data);
  }
 
  printf("data number: %u\n", n);
  printf("cpu merge sort cost %.5f sec\n", elasped[0]  / runtime);
  printf("cpu quick sort cost %.5f sec\n", elasped[1]  / runtime);
  printf("gpu sort cost %.5f sec\n", elasped[2]  / runtime);
  printf("-------------------------------------------\n");
 
  delete[] data;
  delete[] original;
 }
 
 return 0;
}