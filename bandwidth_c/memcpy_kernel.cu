#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

extern "C" __global__
void memcpy_kernel(unsigned char* __restrict__ output, const unsigned char* __restrict__ input){
    output += (blockIdx.x<<13)|(threadIdx.x<<2);
    input  += (blockIdx.x<<13)|(threadIdx.x<<2);
    *((float* )&output[0])       = *((float* )&input[0]);
    *((float* )&output[0x400])   = *((float* )&input[0x400]);
    *((float* )&output[0x800])   = *((float* )&input[0x800]);
    *((float* )&output[0xc00])   = *((float* )&input[0xc00]);
    *((float* )&output[0x1000])  = *((float* )&input[0x1000]);
    *((float* )&output[0x1400])  = *((float* )&input[0x1400]);
    *((float* )&output[0x1800])  = *((float* )&input[0x1800]);
    *((float* )&output[0x1c00])  = *((float* )&input[0x1c00]);
}

#define CALL(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        std::cout<<"'"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define WARMUP 2
#define LOOP 10

static inline void b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%luB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

static inline int env_get_int(const char * var_name, int def_v)
{
    char * v = getenv(var_name);
    int r = def_v;
    if(v)
        r = atoi(v);
    return r;
}

int main() {
	cudaSetDevice(0);
    unsigned char *A, *B;
    const int dwords = env_get_int("DWORDS",64*3*224*224);
    float * h_A = (float*)malloc(dwords*sizeof(float));
    float * h_B = (float*)malloc(dwords*sizeof(float));
	for (int i = 0; i < dwords; ++i)
		h_A[i] = i % 71;

    CALL(cudaMalloc(&A, dwords * sizeof(float)));
    CALL(cudaMalloc(&B, dwords * sizeof(float)));
    CALL(cudaMemcpy(A, h_A, dwords * sizeof(float), cudaMemcpyHostToDevice));

    // benchmark kernel
    int bx = 256;
    int gx = (dwords+255)>>11;
    assert(dwords/bx);

    cudaEvent_t start_ev, stop_ev;
    CALL(cudaEventCreate(&start_ev));
    CALL(cudaEventCreate(&stop_ev));

    for(int i=0;i<WARMUP;i++)
        memcpy_kernel<<<gx, bx>>>(B, A);

    CALL(cudaEventRecord(start_ev, 0));
    for(int i=0;i<LOOP;i++)
        memcpy_kernel<<<gx, bx>>>(B, A);
    CALL(cudaEventRecord( stop_ev, 0 ));
    CALL(cudaEventSynchronize(stop_ev));

    float ms;
    CALL(cudaEventElapsedTime(&ms,start_ev, stop_ev));
    ms/=LOOP;

    sleep(1);

    // benchmark memcpy api
    for(int i=0;i<WARMUP;i++)
        CALL(cudaMemcpy(B, A, dwords * sizeof(float), cudaMemcpyDeviceToDevice));
    CALL(cudaEventRecord( start_ev, 0));
    for(int i=0;i<LOOP;i++)
        CALL(cudaMemcpy(B, A, dwords * sizeof(float), cudaMemcpyDeviceToDevice));
    CALL(cudaEventRecord( stop_ev, 0 ));
    CALL(cudaEventSynchronize(stop_ev));

    float ms_api;
    CALL(cudaEventElapsedTime(&ms_api,start_ev, stop_ev));
    ms_api/=LOOP;

    char str[64];
    b2s(dwords*sizeof(float), str);
    printf("%s, gflops_kernel:%.3f, gflops_api:%.3f\n", str, ((double)dwords*sizeof(float)*2)/((double)ms/1000)/1000000000.0,
    ((double)dwords*sizeof(float)*2)/((double)ms_api/1000)/1000000000.0 );

    free(h_A);
    free(h_B);
    CALL(cudaFree(A));
    CALL(cudaFree(B));
}
