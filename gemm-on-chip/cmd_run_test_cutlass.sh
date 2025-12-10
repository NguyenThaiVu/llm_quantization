/home/tnguyen10/cuda-12.1/bin/nvcc \
  -I /home/tnguyen10/Desktop/deep_learning_research/llm_quantization/gemm-on-chip/cutlass/include \
  -I /home/tnguyen10/Desktop/deep_learning_research/llm_quantization/gemm-on-chip/cutlass/tools/util/include \
  -I /home/tnguyen10/Desktop/deep_learning_research/llm_quantization/gemm-on-chip \
  -gencode arch=compute_80,code=sm_80 \
  -o fused_two_gemms_f16_sm80_shmem fused_two_gemms_f16_sm80_shmem.cu