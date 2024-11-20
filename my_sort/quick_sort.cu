#include <stdio.h>
#include <cstdio>
#include <iostream>
#include "cuda_runtime_api.h"
#include "E:\cproject\cuda_c\CUDA_C\include\freshman.h"
#include <algorithm>

#define MAX_DEPTH 16
#define INSERTION_SORT 32

//using namespace std;

__device__ void selection_sort( int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        
        int min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            int val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void cdp_simple_quicksort( int *data, int left, int right,
                                   int depth) {
    // 当递归的深度大于设定的MAX_DEPTH或者待排序的数组长度小于设定的阈值，直接调用简单选择排序
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

     int *left_ptr = data + left;
     int *right_ptr = data + right;
     int pivot = data[(left + right) / 2];
    // partition
    while (left_ptr <= right_ptr) {
         int left_val = *left_ptr;
         int right_val = *right_ptr;

        while (left_val < pivot) { // 找到第一个比pivot大的
            left_ptr++;
            left_val = *left_ptr;
        }

        while (right_val > pivot) { // 找到第一个比pivot小的
            right_ptr--;
            right_val = *right_ptr;
        }

        // do swap
        if (left_ptr <= right_ptr) {
            *left_ptr++ = right_val;
            *right_ptr-- = left_val;
        }
    }

    // recursive
    int n_right = right_ptr - data;
    int n_left = left_ptr - data;
    // Launch a new block to sort the the left part.
    if (left < (right_ptr - data)) {
        cudaStream_t l_stream;
        // 设置非阻塞流
        cudaStreamCreateWithFlags(&l_stream, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, l_stream>>>(data, left, n_right,
                                                    depth + 1);
        cudaStreamDestroy(l_stream);
    }

    // Launch a new block to sort the the right part.
    if ((left_ptr - data) < right) {
        cudaStream_t r_stream;
        // 设置非阻塞流
        cudaStreamCreateWithFlags(&r_stream, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, r_stream>>>(data, n_left, right,
                                                    depth + 1);
        cudaStreamDestroy(r_stream);
    }
}

// Call the quicksort kernel from the host.
void device_qsort(int *data, const int nitems) {
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    int left = 0;
    int right = nitems - 1;
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    CHECK(cudaDeviceSynchronize());
}




bool cmp(int i, int j)
{
    return i<j;
}

int main()
{
    const int num_items = 500000;
    double iStart, iElaps;
    int *h_data = NULL;
    int *d_data = NULL;
    //int *h_res = NULL;
    int *d_res = NULL;
    // 初始化数据
    h_data = (int*)malloc(num_items * sizeof(int));
    d_res = (int*)malloc(num_items * sizeof(int));
    //h_res =  = (int*)malloc(num_items * sizeof(int));

    initialData_int(h_data, num_items);

    // 分配设备内存
    CHECK(cudaMalloc( (void**)&d_data, num_items * sizeof(int)))
    CHECK(cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice))
    
    iStart=cpuSecond();
    device_qsort(d_data, num_items);
    iElaps=cpuSecond()-iStart;
    printf("Device sort Time elapsed %f sec\n",iElaps);

    iStart=cpuSecond();
    std::sort(h_data, h_data + num_items,cmp);
    iElaps=cpuSecond()-iStart;
    printf("Host sort Time elapsed %f sec\n",iElaps);

    CHECK(cudaMemcpy(d_res, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost))


    // check result
    checkResult(d_res, h_data, num_items);

    free(h_data);
    CHECK(cudaFree(d_data));

    return 0;
}