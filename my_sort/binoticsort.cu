#include <stdio.h>
#include <cstdio>
#include <iostream>
#include "cuda_runtime_api.h"
#include "E:\cproject\cuda_c\CUDA_C\include\freshman.h"
#include <algorithm>

#define tile_size 256u
#define log_tile_size 8u

template<typename type>__device__ void swap(type& v1, type& v2)
{
	type t = v1;
	v1 = v2;
	v2 = t;
}

__global__ void binoticsort_gpu(float* data, int n)
{
	__shared__  double buffer[tile_size << 1];

	unsigned int d = blockIdx.x << (log_tile_size + 1), i = threadIdx.x;
	data += d; n -= d;
	buffer[i] = i < n ? data[i] : DBL_MAX;
	buffer[i + blockDim.x] = i + blockDim.x < n ? data[i + blockDim.x] : DBL_MAX;
	__syncthreads();

	unsigned int s, t;
	for (unsigned int k = 2; k <= blockDim.x; k <<= 1)
		for (unsigned int p = k >> 1; p; p >>= 1)
		{
			s = (i & -p) << 1 | i & p - 1;
			t = s | p;

			if (s & k ? buffer[s] > buffer[t] : buffer[s] < buffer[t])
				swap(buffer[s], buffer[t]);
			__syncthreads();
		}
	for (unsigned int p = blockDim.x; p; p >>= 1)
	{
		s = (i & -p) << 1 | i & p - 1;
		t = s | p;
		if (buffer[s] > buffer[t])
			swap(buffer[s], buffer[t]);
		__syncthreads();
	}

	if (i < n)
		data[i] = buffer[i];
	if (i + blockDim.x < n)
		data[i + blockDim.x] = buffer[i + blockDim.x];
}

void binoticsort(float* data, const int n)
{
    binoticsort_gpu<<<(n-1) / tile_size + 1, tile_size>>>(data, n);

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
    const int num_items = 2>>8;
    double iStart, iElaps;
    float *h_data = NULL;
    float *d_data = NULL;
    //int *h_res = NULL;
    float *d_res = NULL;
    
    // 分配内存
    h_data = (float*)malloc(num_items * sizeof(float));
    d_res = (float*)malloc(num_items * sizeof(float));
    //h_res =  = (int*)malloc(num_items * sizeof(int));
    CHECK(cudaMalloc( (void**)&d_data, num_items * sizeof(float)))

    // 初始化数据
    initialData_float(h_data, num_items);
    printf_data(h_data, num_items);
    // 数据传输
    CHECK(cudaMemcpy(d_data, h_data, num_items * sizeof(float), cudaMemcpyHostToDevice))
    
    // GPU sort
    iStart=cpuSecond();
    binoticsort(d_data, num_items);
    // 数据传输 
    CHECK(cudaMemcpy(d_res, d_data, num_items * sizeof(float), cudaMemcpyDeviceToHost))
    iElaps=cpuSecond()-iStart;
    printf("Device sort Time elapsed %f sec\n",iElaps);

    // CPU sort
    iStart=cpuSecond();
    std::sort(h_data, h_data + num_items,cmp);
    iElaps=cpuSecond()-iStart;
    printf("Host sort Time elapsed %f sec\n",iElaps);

    

    
    // check result
    checkResult(d_res, h_data, num_items);

    free(h_data);
    free(d_res);
    CHECK(cudaFree(d_data));

    return 0;
}