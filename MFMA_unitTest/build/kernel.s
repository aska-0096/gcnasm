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
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   

        s_nop 1

        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   

        s_nop 1

        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   

        s_nop 1

        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   
        v_mfma_f32_4x4x4f16 a[.itr+0:.itr+3], v[.itr+0:.itr+1], v[.itr+2:.itr+3], a[.itr+0:.itr+3]   

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
    - { .name: rand_seed,   .size: 4, .offset:   0, .value_kind: by_value, .value_type: f32}
    - { .name: inst_blocks, .size: 4, .offset:   4, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata

