#include <stdio.h>
#include <string>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                      \
} while(0)

#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int num_cu;
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        num_cu = dev_prop.multiProcessorCount;
    }

    int total_loop=100;
    int warm_ups = 5;
    int i;
    int inst_iter = 5;
    int inst_loop = 256;
    int bdx = 256;
    int gdx = num_cu;

    float rand_seed = ((float)(rand() % 100));
    struct {
        float rand_seed;
        int inst_iter;
    } args;
    size_t arg_size = sizeof(args);
    args.inst_iter = inst_iter;
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};

    for(i=0;i<warm_ups;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);

    hipCtxSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipCtxSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms/total_loop;
    //float tips = (double)inst_loop*inst_blocks*num_cu*bdx/time_per_loop/1e9;
    //argv 2~5 = M, N, K, blocks
    int M = std::stoull(std::string(argv[2]));
    int N = std::stoull(std::string(argv[3]));
    int K = std::stoull(std::string(argv[4]));
    int blocks = std::stoull(std::string(argv[5]));
    float Tflops = (double)2*inst_loop*M*N*K*blocks*4*num_cu* (32*inst_iter) / time_per_loop /1e9;

    //printf("CU:%d, inst:%s, TIPS: %.3f), cost:%fms per loop\n", num_cu, argv[1], Tflops, time_per_loop);
    printf("%d\t%s\t%.3f\t%.3fms\n", num_cu, argv[1], Tflops, time_per_loop);
}
