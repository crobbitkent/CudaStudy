#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define N 100

__global__ void printHelloCuda()
{
	printf("Hello, CUDA!\n");
}



__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main()
{


	float *A, *B, *C;
	 int num = N;
	cudaMalloc((void**)&A, num*num * sizeof(float));
	cudaMalloc((void**)&B, num*num * sizeof(float));
	cudaMalloc((void**)&C, num*num * sizeof(float));

	float *a = (float*)malloc(num*num * sizeof(float));
	float *b = (float*)malloc(num*num * sizeof(float));
	float *c = (float*)malloc(num*num * sizeof(float));

	cudaMemcpy(A, a, num*num * sizeof(*A), cudaMemcpyHostToDevice);
	cudaMemcpy(B, b, num*num * sizeof(*b), cudaMemcpyHostToDevice);

	// num * num * 1threads의 한 블록과 함께 실행되는 커널 인보케이션
	int numBlocks = 1;
	dim3 threadsPerBlock(num, num);
	MatAdd << <numBlocks, threadsPerBlock >> > ((float(*)[N])A, (float(*)[N])B, (float(*)[N])C);

	cudaMemcpy(c, C, num*num * sizeof(*C), cudaMemcpyHostToDevice);

	cudaFree(A); cudaFree(B); cudaFree(C);




	return 0;
}


