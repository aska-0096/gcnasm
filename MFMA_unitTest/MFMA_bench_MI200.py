#This is a script for testing MI200 instructions throughout
#input: random seed, iterator number.
from __future__ import print_function
import subprocess
import os
import shutil
import sys

k_HSACO = "kernel.co"
k_HSAKN = "kernel_func"
k_WS = "build"
k_CPP_SRC = "bench.cpp"
k_CPP_TARGET = "bench.exe"
k_ASM_SRC = "kernel.s"
k_ASM_TARGET = k_HSACO
k_ARCH = "gfx90a"
k_INST_LOOP = [256, 512, 768, 1024]
USE_HIP_CLANG = True

AMDGPU_PRECISION_FP32   = (0 << 20)
AMDGPU_PRECISION_FP16   = (1 << 20)
AMDGPU_PRECISION_BF16   = (2 << 20)
AMDGPU_PRECISION_INT8   = (3 << 20)

def inst_mfma_data_type_to_string(data_type):
    if data_type == AMDGPU_PRECISION_FP32:
        return 'fp32'
    if data_type == AMDGPU_PRECISION_FP16:
        return 'fp16'
    if data_type == AMDGPU_PRECISION_BF16:
        return 'bf16'
    if data_type == AMDGPU_PRECISION_INT8:
        return 'int8'
    assert False

class inst_mfma_t(object):
    '''
    http://llvm.org/docs/AMDGPU/AMDGPUAsmGFX908.html
    '''
    def __init__(self, m, n, k, data_type, cycle, num_v_a, num_v_b, num_a_c, num_blocks, **options):
        #self.arch_config = arch_config
        self.m = m
        self.n = n
        self.k = k
        self.data_type = data_type
        self.cycle = cycle
        self.num_v_a = num_v_a
        self.num_v_b = num_v_b
        self.num_a_c = num_a_c
        self.num_blocks = num_blocks
        self.accvgpr_unified = False
        self.options = options
        # self.num_a_c_per_lanegroup = 4      # all xdlops instruction output agpr is 4 agpr per lanegroup.
        #assert arch_config.arch == AMDGPU_ARCH_GFX908 and arch_config.use_xdlops

    def name(self):
        if 'name' in self.options and self.options['name'] != None:
            return self.options['name']
        def src_datatype_string(data_type_string):
            if data_type_string == 'fp32':
                return 'f32'
            if data_type_string == 'fp16':
                return 'f16'
            if data_type_string == 'bf16':
                return 'bf16'
            if data_type_string == 'int8':
                return 'i8'
            assert False, "unknow type :{}".format(data_type_string)
        mfma_acc_type = 'i32' if self.data_type == AMDGPU_PRECISION_INT8 else 'f32' # TODO: int8 mfma accumulate type is i32
        mfma_trait = '{}x{}x{}'.format(self.m, self.n, self.k) + src_datatype_string(inst_mfma_data_type_to_string(self.data_type))
        mfma_inst = 'v_mfma_{}_{}'.format(mfma_acc_type, mfma_trait)
        if 'bf16_1k' in self.options and self.options['bf16_1k'] and self.data_type == AMDGPU_PRECISION_BF16:
            mfma_inst += '_1k'
        return mfma_inst

    def __call__(self, reg_d, reg_a, reg_b, reg_c, cbsz=0, abid=0, blgp=0):
        mfma_inst = self.name()
        cbsz_str = "cbsz:{}".format(cbsz) if cbsz != 0 else ""
        abid_str = "abid:{}".format(abid) if abid != 0 else ""
        blgp_str = "blgp:{}".format(blgp) if blgp != 0 else ""
        if self.accvgpr_unified:
            return  "{} v[{}], v[{}], v[{}], v[{}] {} {} {}".format(mfma_inst, reg_d, reg_a, reg_b, reg_c, cbsz_str, abid_str, blgp_str)
        else:
            return  "{} a[{}], v[{}], v[{}], a[{}] {} {} {}".format(mfma_inst, reg_d, reg_a, reg_b, reg_c, cbsz_str, abid_str, blgp_str)

    def get_nop_count_mfma_acc_raw(self):
        # in unit of passes, aka 4 cycle
        return (self.cycle // 4) + 2


#                                     m,  n,  k,  precision,           cycle, v_a, v_b, a_c, #block
v_mfma_f32_4x4x4f16     = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_FP16,   8,   2,   2,  4,    16)
v_mfma_f32_16x16x4f16   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_FP16,  32,   2,   2,  16,   4 )
v_mfma_f32_16x16x16f16  = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_FP16,  32,   2,   2,  4,    1 )
v_mfma_f32_32x32x4f16   = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_FP16,  64,   2,   2,  32,   2 )
v_mfma_f32_32x32x8f16   = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_FP16,  64,   2,   2,  16,   1 )

v_mfma_f32_4x4x4bf16_1k     = inst_mfma_t(4,  4,  4,  AMDGPU_PRECISION_BF16,   8,   2,   2,  4,    16, bf16_1k=True)
v_mfma_f32_16x16x4bf16_1k   = inst_mfma_t(16, 16, 4,  AMDGPU_PRECISION_BF16,  32,   2,   2,  16,   4 , bf16_1k=True)
v_mfma_f32_16x16x16bf16_1k  = inst_mfma_t(16, 16, 16, AMDGPU_PRECISION_BF16,  32,   2,   2,  4,    1 , bf16_1k=True)
v_mfma_f32_32x32x4bf16_1k   = inst_mfma_t(32, 32, 4,  AMDGPU_PRECISION_BF16,  64,   2,   2,  32,   2 , bf16_1k=True)
v_mfma_f32_32x32x8bf16_1k   = inst_mfma_t(32, 32, 8,  AMDGPU_PRECISION_BF16,  64,   2,   2,  16,   1 , bf16_1k=True)
class cpp_src_t:
    def get_cxxflags(self):
        if USE_HIP_CLANG:
            return ' -mcpu={} '.format(k_ARCH)
        else:
            return '`/opt/rocm/bin/hipconfig --cpp_config` -Wall -O2  -std=c++11 '
    def get_ldflags(self):
        if USE_HIP_CLANG:
            return ''
        else:
            return " -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64" \
                " -Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib -ldl -lm -lpthread -lhc_am " \
                " -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"
    def compile(self, src, target, working_dir):
        def do_compile():
            if USE_HIP_CLANG:
                cmd = "/opt/rocm/hip/bin/hipcc "
            else:
                cmd = "g++" + " "
            cmd += self.get_cxxflags() + " "
            cmd += src + " "
            cmd += self.get_ldflags() + " "
            cmd += "-o {}".format(target)
            proc = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "CPP Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(working_dir)
        if os.path.exists(target):
            os.remove(target)
        do_compile()
        os.chdir(save_dir)

    def get_src(self):
        src = '''\
#include <stdio.h>
#include <string>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>

#define HIP_CALL(call) do{{  \\
    hipError_t err = call;  \\
    if(err != hipSuccess){{  \\
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \\
        exit(0);            \\
    }}                      \\
}} while(0)

#define HSACO "{hsaco}"
#define HSA_KERNEL "{hsakn}"

int main(int argc, char ** argv){{
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int num_cu;
    {{
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        num_cu = dev_prop.multiProcessorCount;
    }}

    int total_loop=100;
    int warm_ups = 5;
    int i;
    int inst_iter = 1500;
    int bdx = 256;
    int gdx = num_cu;

    float rand_seed = ((float)(rand() % 100));
    struct {{
        float rand_seed;
        int inst_iter;
    }} args;
    size_t arg_size = sizeof(args);
    args.inst_iter = inst_iter;
    void* config[] = {{HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END}};

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
    float Tflops = (double)2*M*N*K*blocks*4*num_cu* (32*inst_iter) / time_per_loop /1e9;

    //printf("CU:%d, inst:%s, TIPS: %.3f), cost:%fms per loop\\n", num_cu, argv[1], Tflops, time_per_loop);
    printf("%d\\t%s\\t%.3f\\t%.3fms\\n", num_cu, argv[1], Tflops, time_per_loop);
}}
'''.format(hsaco=k_HSACO, hsakn=k_HSAKN)
        return src
    def write(self,f):
        f.write(self.get_src())

class asm_src_t:
    def __init__(self,arch,bench_inst):
        self.arch = arch
        self.bench_inst = bench_inst[0](bench_inst[1], bench_inst[2] ,bench_inst[3], bench_inst[4])   
        self.arch_str = ','.join([arch[3],arch[4],arch[5]])
    def get_asmflags():
        return ""
    def compile(self, src, target, working_dir):
        def do_compile():
            if USE_HIP_CLANG:
                cmd = "/opt/rocm/llvm/bin/clang++" + " "
                cmd += "-x assembler -target amdgcn--amdhsa -mcpu={} ".format(self.arch) + " "
            else:
                cmd = "/opt/rocm/hcc/bin/clang" + " "
                #cmd += "-x assembler -target amdgcn--amdhsa -mcpu={} -mno-code-object-v3".format(self.arch) + " "
                cmd += "-x assembler -target amdgcn--amdhsa -mcpu={} ".format(self.arch) + " "
            cmd += src + " "
            cmd += "-o {}".format(target)
            proc = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "ASM Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(working_dir)
        if os.path.exists(target):
            os.remove(target)
        do_compile()
        os.chdir(save_dir)
    def disassemble(self, hsaco, output, working_dir):
        def do_disassembly():
            if not os.path.exists(hsaco):
                print("not exist {}, fail to disassembly".format(hsaco))
                return
            if USE_HIP_CLANG:
                cmd = "/opt/rocm/llvm/bin/llvm-objdump" + " "
                cmd += "--disassemble --mcpu={}".format(self.arch) + " "
            else:
                cmd = "/opt/rocm/hcc/bin/llvm-objdump" + " "
                cmd += "-disassemble -mcpu={}".format(self.arch) + " "
            cmd += hsaco + " "
            cmd += "> {}".format(output)
            proc = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "DISASM Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(working_dir)
        do_disassembly()
        os.chdir(save_dir)

    def get_src(self):
        #if USE_HIP_CLANG:
        if True:
            asm_src='''\
.text
.global kernel_func
.p2align 8
.type kernel_func,@function

.set k_bdx,     256     ; should be 256 in bdx
.set k_end,     12
.set v_end,     128     ; hard code to this to let occupancy to be 1.  65536 / 256 = 256
.set s_rand,    12
.set s_iter,    13
.set s_tmp,     14
.set s_end,     31
.set a_end,     63
.set inst_loop, 256

kernel_func:
    s_load_dword        s[s_rand], s[0:1], 0
    s_load_dword        s[s_iter], s[0:1], 4
    s_waitcnt           lgkmcnt(0)
    .cnt=0
    .rept 128
        s_sub_u32 s[s_tmp], s[s_rand], .cnt
        v_mov_b32 v[.cnt], s[s_tmp]
        .cnt = .cnt + 1
    .endr
L_kernel_start:
    s_sub_u32 s[s_iter], s[s_iter], 1
    .itr = 0
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}

        s_nop 1

        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}

        s_nop 1

        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}

        s_nop 1

        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}
        {bench_inst}

        s_nop 1
        
        .itr = .itr+4
        .if .itr > (v_end-4+1)
            .itr = 0
        .endif
    s_cmp_gt_u32 s[s_iter], 0
    s_cbranch_scc1 L_kernel_start

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 64
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: kernel_func
    .symbol: kernel_func.kd
    .sgpr_count: 32
    .vgpr_count: 256
    .kernarg_segment_align: 4
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - {{ .name: rand_seed,   .size: 4, .offset:   0, .value_kind: by_value, .value_type: f32}}
    - {{ .name: inst_blocks, .size: 4, .offset:   4, .value_kind: by_value, .value_type: i32}}
...
.end_amdgpu_metadata

'''.format(bench_inst=self.bench_inst)
        else:
            asm_src='''\
.hsa_code_object_version 2,0
.hsa_code_object_isa {arch_str}, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel kernel_func

.set k_bdx,     256     ; should be 256 in bdx
.set k_end,     12
.set v_end,     255     ; hard code to this to let occupancy to be 1.  65536 / 256 = 256
.set s_blocks,  12
.set s_end,     31
.set inst_loop, 256

kernel_func:
    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr     = 1
        user_sgpr_count                     = 2
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 1
        enable_vgpr_workitem_id             = 0
        is_ptr64                            = 1
        float_mode                          = 2
        wavefront_sgpr_count                = s_end+1+2*3    ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count                 = v_end+1
        granulated_workitem_vgpr_count      = v_end/4  ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count     = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8
        kernarg_segment_byte_size           = k_end
        workgroup_group_segment_byte_size   = 0
    .end_amd_kernel_code_t

    s_load_dword        s[s_blocks], s[0:1], 8
    s_waitcnt           lgkmcnt(0)
L_kernel_start:
    s_sub_u32 s[s_blocks], s[s_blocks], 1
    .itr = 0
    .rept inst_loop
        {bench_inst}
        .itr = .itr+4
        .if .itr > (v_end-4+1)
            .itr = 0
        .endif
    .endr
    s_cmp_gt_u32 s[s_blocks], 0
    s_cbranch_scc1 L_kernel_start

    s_endpgm
'''.format(arch_str=self.arch_str, bench_inst=self.bench_inst)
        return asm_src
    def write(self,f):
        f.write(self.get_src())

bench_inst_dict = [
    (v_mfma_f32_16x16x16f16,'.itr+0:.itr+3', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+3'),
    (v_mfma_f32_16x16x4f16, '.itr+0:.itr+15', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+15'),
    (v_mfma_f32_32x32x4f16, '.itr+0:.itr+31', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+31'),
    (v_mfma_f32_32x32x8f16, '.itr+0:.itr+15', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+15'),
    (v_mfma_f32_4x4x4f16,   '.itr+0:.itr+3' , '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+3'),

    #(v_mfma_f32_16x16x16bf16_1k, '.itr+0:.itr+3', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+3'),
    #(v_mfma_f32_16x16x4bf16_1k, '.itr+0:.itr+15', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+15'),
    #(v_mfma_f32_32x32x4bf16_1k, '.itr+0:.itr+31', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+31'),
    #(v_mfma_f32_32x32x8bf16_1k, '.itr+0:.itr+15', '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+15'),
    #(v_mfma_f32_4x4x4bf16_1k,   '.itr+0:.itr+3' , '.itr+0:.itr+1', '.itr+2:.itr+3', '.itr+0:.itr+3')
]

benched_inst_dict = dict()

def bench():
    def prepare_cpp():
        cpp_src = cpp_src_t()
        with open(os.path.join(k_WS, k_CPP_SRC), "w") as f:
            cpp_src.write(f)
        cpp_src.compile(k_CPP_SRC, k_CPP_TARGET, k_WS)
    def prepare_asm(arch, bench_inst):
        inst = bench_inst[0].name()
        if inst in benched_inst_dict:
            cnt = benched_inst_dict[inst]
            cnt = cnt+1
            inst = inst +'_{}'.format(cnt)
            benched_inst_dict[inst] = cnt
        else:
            benched_inst_dict[inst] = 0
        asm_src = asm_src_t(arch, bench_inst)
        src_path = os.path.join(k_WS, k_ASM_SRC)
        target_path = os.path.join(k_WS, k_ASM_TARGET)
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(target_path):
            os.remove(target_path)

        with open(src_path, "w") as f:
            asm_src.write(f)
        asm_src.compile(k_ASM_SRC,k_ASM_TARGET,k_WS)
        asm_src.disassemble(k_ASM_TARGET, k_ASM_TARGET+".dump.{}.s".format(inst), k_WS)

    def run_bench(bench_inst):
        def do_run():
            if not os.path.exists(k_CPP_TARGET):
                print("not exist {}, fail to run".format(k_CPP_TARGET))
                return
            if not os.path.exists(k_HSACO):
                print("not exist {}, fail to run".format(k_HSACO))
                return
            inst = bench_inst[0].name()
            M = bench_inst[0].m
            N = bench_inst[0].n
            K = bench_inst[0].k
            blocks = bench_inst[0].num_blocks
            #          inst  M  N  K  blocks
            cmd = "./{} {} {} {} {} {}".format(k_CPP_TARGET, inst, M, N, K, blocks)
            proc = subprocess.Popen(cmd,
                        stdout=sys.stdout,
                        stderr=sys.stdout,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "Launch Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(k_WS)
        do_run()
        os.chdir(save_dir)

    shutil.rmtree(k_WS,True)
    os.mkdir(k_WS)
    prepare_cpp()
    print("CU\tinstruction      \tTflops\tper_loop")
    for item in bench_inst_dict:
        bench_inst = item
        prepare_asm(k_ARCH, bench_inst)
        run_bench(bench_inst)

def check_hip_clang():
    # return True/False
    return os.path.exists('/opt/rocm/llvm/bin/clang++')

class test_suite:
    def __init__(self):
        pass
    def __del__(self):
        pass


#gen_cpp()
#gen_asm("9,0,6","v_fmac_f32 v[.itr], v[.itr+1], v[.itr+2]")
USE_HIP_CLANG = check_hip_clang()
bench()
