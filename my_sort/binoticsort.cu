#include <stdio.h>
#include <cstdio>
#include <iostream>
#include "cuda_runtime_api.h"
#include "freshman.h"
#include <algorithm>

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

__global__ void binoticsort_gpu(float* data, int n)
{
	__shared__  float buffer[tile_size << 1];

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
__global__ void mergedirect_gpu(const float* data, unsigned int n, unsigned int s, float* result)
{
 unsigned int d1 = (blockIdx.z * gridDim.y + blockIdx.y) * s << 1, d2 = d1 + s, i = d1 + blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n)
  result[d2 < n ? i + lower_bound(data + d2, min(s, n - d2), data[i]) : i] = data[i];
 if (s + i < n)
  result[i + upper_bound(data + d1, s, data[s + i])] = data[s + i];
}
float* sortdirect_gpu(float* data, unsigned int n)
{
 float* tmp;
 cudaMalloc(&tmp, sizeof(float) * n);
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

__global__ void _bitonic_sort(float* data, unsigned stride, unsigned inner_stride) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned s = (tid & -inner_stride) << 1 | tid & inner_stride - 1;
	unsigned t = s | inner_stride;

    if (s & stride ? data[s] < data[t] : data[s] > data[t])
		swap(data[s], data[t]);
    
}

void float_sort(float *arr, int len) {
    // 首先检查长度是否为 2 的幂
    unsigned twoUpper = 1;
    for (;twoUpper < len; twoUpper <<= 1) {
        if (twoUpper == len) {
            break;
        }
    }

    // 如果是 host 指针，返回
    cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, arr);
    if (attrs.type != cudaMemoryTypeDevice) {
        return;
    }

    unsigned input_arr_len = len;
    // if (twoUpper == len) {
    //     input_arr_len = len;
    //     d_input_arr = arr;
    // } else {
    //     // 需要 padding
    //     input_arr_len = twoUpper;
    //     cudaMalloc(&d_input_arr, sizeof(float) * input_arr_len);
    //     // 然后初始化
    //     cudaMemcpy(d_input_arr, arr, sizeof(float) * len, cudaMemcpyHostToDevice);
    //     cudaMemset(d_input_arr + len, 0x7f, sizeof(float) * (input_arr_len - len));
    // }

    dim3 grid_dim((input_arr_len / 512 == 0)? 1 : input_arr_len / 512);
    dim3 block_dim((input_arr_len / 512 == 0)? input_arr_len : 256);
    
    // 排序过程(重点)
    for (unsigned stride = 2; stride <= input_arr_len; stride <<= 1) {
        for (unsigned inner_stride = stride>>1; inner_stride > 0; inner_stride >>= 1) {
            _bitonic_sort<<<grid_dim, block_dim>>>(arr, stride, inner_stride);
        }
    }

    // 如果 padding 过，则此处还原
    // if (twoUpper != len) {
    //     cudaMemcpy(arr, d_input_arr, sizeof(float) * len, cudaMemcpyDeviceToDevice);
    //     cudaFree(d_input_arr);
    // }
}

void printf_data(float* data, const int n)
{
    for(int i=0;i<n;i++)
    {
        printf("%f ",data[i]);
    }
    printf("\n");
}

bool cmp(float i, float j)
{
    return i<j;
}



int main()
{
    const int num_items = 2<<24;
    double iStart, iElaps;
    float *h_data = NULL;
    float *d_data_1 = NULL;
    float *d_data_2 = NULL;
    //int *h_res = NULL;
    float *d_res_1 = NULL;
    float *d_res_2 = NULL;
     
    // 分配内存
    h_data = (float*)malloc(num_items * sizeof(float));
    d_res_1 = (float*)malloc(num_items * sizeof(float));
    d_res_2 = (float*)malloc(num_items * sizeof(float));
    //h_res =  = (int*)malloc(num_items * sizeof(int));
    CHECK(cudaMalloc( (void**)&d_data_1, num_items * sizeof(float)))
    CHECK(cudaMalloc( (void**)&d_data_2, num_items * sizeof(float)))

    // 初始化数据
    initialData(h_data, num_items);
    // printf_data(h_data, num_items);
    // 数据传输
    CHECK(cudaMemcpy(d_data_1, h_data, num_items * sizeof(float), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_data_2, h_data, num_items * sizeof(float), cudaMemcpyHostToDevice))

    // GPU sort
    iStart=cpuSecond();
    d_data_1 = sortdirect_gpu(d_data_1, num_items);
    iElaps=cpuSecond()-iStart;
    // 数据传输 
    CHECK(cudaMemcpy(d_res_1, d_data_1, num_items * sizeof(float), cudaMemcpyDeviceToHost))
    printf("Device sort Time elapsed %f sec\n",iElaps);

    // GPU sort 2
    iStart=cpuSecond();
    float_sort(d_data_2, num_items);
    cudaDeviceSynchronize();
    iElaps=cpuSecond()-iStart;
    // 数据传输 
    CHECK(cudaMemcpy(d_res_2, d_data_2, num_items * sizeof(float), cudaMemcpyDeviceToHost))
    printf("Device sort Time elapsed %f sec\n",iElaps);
    
    // printf_data(d_res, num_items);
    // CPU sort
    iStart=cpuSecond();
    std::sort(h_data, h_data + num_items,cmp);
    iElaps=cpuSecond()-iStart;
    printf("Host sort Time elapsed %f sec\n",iElaps);

    // check result
    checkResult(d_res_2, h_data, num_items);
    checkResult(d_res_1, h_data, num_items);

    free(h_data);
    free(d_res_1);
    free(d_res_2);
    CHECK(cudaFree(d_data_1));
    CHECK(cudaFree(d_data_2));

    return 0;
}