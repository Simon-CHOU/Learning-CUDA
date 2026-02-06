#include <algorithm>  // 用于 std::min
#include <type_traits>  // 用于在 host 端区分 float/half 的 kernel 分发
#include <vector>

#include <cuda_fp16.h>

#include "../tester/utils.h"

// CUDA Runtime 调用检查（复用 tester/utils.h 中的 RUNTIME_CHECK）
#define CHECK_CUDA(call) RUNTIME_CHECK(call)

////////// 作业题1 trace START //////////
// 定义 CUDA Kernel 用于计算 Trace
// 使用原子加法实现简单的归约求和
template <typename T>
__global__ void trace_kernel(const T* input, T* output, size_t rows, size_t cols, size_t n) {
  // 计算当前线程的全局索引
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  T local_sum = 0;
  // Grid-Stride Loop: 遍历对角线元素
  for (size_t i = idx; i < n; i += stride) {
    // 对角线元素在 flattened array 中的位置是 input[i * cols + i]
    local_sum += input[i * cols + i];
  }

  // 使用原子加法将局部和累加到全局输出中
  // 注意：atomicAdd 支持 int 和 float
  if (local_sum != 0) {
    atomicAdd(output, local_sum);
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // 计算对角线长度
  size_t n = std::min(rows, cols);
  if (n == 0) return T(0);

  // 1. 分配设备内存
  T* d_input = nullptr;
  T* d_output = nullptr;
  size_t input_size = h_input.size() * sizeof(T);
  size_t output_size = sizeof(T);

  CHECK_CUDA(cudaMalloc(&d_input, input_size));
  CHECK_CUDA(cudaMalloc(&d_output, output_size));

  // 2. 拷贝数据到设备
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_output, 0, output_size)); // 初始化输出为 0

  // 3. 启动 Kernel
  // 使用简单的配置，例如 256 个线程的 block
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  // 限制 grid 大小以防 n 非常大时产生过多 block
  if (grid_size > 1024) grid_size = 1024;

  trace_kernel<<<grid_size, block_size>>>(d_input, d_output, rows, cols, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // 4. 拷贝结果回主机
  T result = 0;
  CHECK_CUDA(cudaMemcpy(&result, d_output, output_size, cudaMemcpyDeviceToHost));

  // 5. 释放内存
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));

  return result;
}
////////// 作业题1 trace END //////////

////////// 作业题2 flashAttention START //////////

// 辅助函数：Warp 内归约求和
template <typename T>
__device__ T warpReduceSum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// 辅助函数：Block 内归约求和
template <typename T>
__device__ T blockReduceSum(T val) {
  // 共享内存用于存储每个 Warp 的结果
  // 假设最大 Block Size 为 1024，即 32 个 Warps
  static __shared__ T shared[32];
  
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // 每个 Warp 进行归约
  val = warpReduceSum(val);

  // 每个 Warp 的第一个线程将结果写入共享内存
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();

  // 由第一个 Warp 对共享内存中的结果进行二次归约
  // 只有当 Block 大小超过 1 个 Warp 时才需要
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : T(0);
  
  if (wid == 0) val = warpReduceSum(val);

  return val;
}

// Flash Attention Kernel
// 采用 Two-Pass Softmax 策略以提高数值精度，解决长序列下的累积误差问题
// Pass 1: 计算 Max Score
// Pass 2: 计算 Sum Exp 和 Output
__global__ void flash_attention_kernel_float(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int tgt_len, int src_len,
    int q_heads, int kv_heads, int head_dim,
    bool is_causal) {
  // 说明：为严格对齐参考实现的浮点累加顺序，float 路径采用 thread0 串行点积并广播 score
  int t_idx = blockIdx.x;
  int h_idx = blockIdx.y;
  int b_idx = blockIdx.z;
  int d_idx = threadIdx.x;

  int kv_h_idx = (h_idx * kv_heads) / q_heads;  // GQA 映射：比例映射更稳健

  size_t q_offset = (size_t)b_idx * tgt_len * q_heads * head_dim +
                    (size_t)t_idx * q_heads * head_dim +
                    (size_t)h_idx * head_dim;

  size_t o_offset = q_offset;

  size_t k_base_offset = (size_t)b_idx * src_len * kv_heads * head_dim +
                         (size_t)kv_h_idx * head_dim;

  size_t v_base_offset = k_base_offset;

  float o_accum = 0.0f;

  __shared__ float shared_max;
  __shared__ float shared_score;
  __shared__ float shared_denom;
  __shared__ float shared_weight;

  float scale = rsqrtf(static_cast<float>(head_dim));  // 缩放因子：1/sqrt(d)

  // Pass 1：计算 max(score)
  if (threadIdx.x == 0) {
    shared_max = -INFINITY;
  }
  __syncthreads();

  for (int s_idx = 0; s_idx < src_len; ++s_idx) {
    if (is_causal && s_idx > t_idx) continue;  // causal mask：上三角置零

    if (threadIdx.x == 0) {
      float dot = 0.0f;
      size_t k_ptr = k_base_offset + (size_t)s_idx * kv_heads * head_dim;
      for (int i = 0; i < head_dim; ++i) {
        dot += Q[q_offset + i] * K[k_ptr + i];  // 串行累加以对齐参考实现
      }
      dot *= scale;
      shared_score = dot;
      if (dot > shared_max) shared_max = dot;
    }
    __syncthreads();
  }

  float m_max = shared_max;

  // Pass 2：计算 denom 和输出
  if (threadIdx.x == 0) {
    shared_denom = 0.0f;
  }
  __syncthreads();

  for (int s_idx = 0; s_idx < src_len; ++s_idx) {
    if (is_causal && s_idx > t_idx) continue;  // causal mask：上三角置零

    if (threadIdx.x == 0) {
      float dot = 0.0f;
      size_t k_ptr = k_base_offset + (size_t)s_idx * kv_heads * head_dim;
      for (int i = 0; i < head_dim; ++i) {
        dot += Q[q_offset + i] * K[k_ptr + i];  // 串行累加以对齐参考实现
      }
      dot *= scale;
      shared_score = dot;
      float w = expf(dot - m_max);
      shared_weight = w;
      shared_denom += w;
    }
    __syncthreads();

    float w = shared_weight;
    if (d_idx < head_dim) {
      size_t v_ptr = v_base_offset + (size_t)s_idx * kv_heads * head_dim + d_idx;
      o_accum += V[v_ptr] * w;
    }
    __syncthreads();
  }

  float denom = shared_denom;
  if (d_idx < head_dim) {
    O[o_offset + d_idx] = (denom > 0.0f) ? (o_accum / denom) : 0.0f;
  }
}

template <typename T>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int tgt_len, int src_len,
    int q_heads, int kv_heads, int head_dim,
    bool is_causal) {

    // 使用 double 进行累加和中间计算以提高精度
    using AccT = double;

    // Grid 维度映射
    int t_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;

    // 当前线程处理 head_dim 中的一部分
    int d_idx = threadIdx.x;
    
    // GQA 映射：用比例映射替代整除假设，兼容 q_heads 与 kv_heads 不整除的情况
    int kv_h_idx = (h_idx * kv_heads) / q_heads;

    // Causal Mask：与 torch.tril 语义一致（对齐左上角），即仅允许 s_idx <= t_idx

    // 计算指针偏移量
    size_t q_offset = (size_t)b_idx * tgt_len * q_heads * head_dim +
                      (size_t)t_idx * q_heads * head_dim +
                      (size_t)h_idx * head_dim;
    
    size_t o_offset = q_offset; 
    
    size_t k_base_offset = (size_t)b_idx * src_len * kv_heads * head_dim +
                           (size_t)kv_h_idx * head_dim;
                           
    size_t v_base_offset = k_base_offset; 

    // 读取 Query 向量
    AccT q_val = 0.0;
    if (d_idx < head_dim) {
        q_val = static_cast<AccT>(Q[q_offset + d_idx]);
    }
    
    // 缩放因子：使用 CUDA 的 rsqrtf 生成 float 精度的 1/sqrt(head_dim)，以匹配常见参考实现
    AccT scale = static_cast<AccT>(rsqrtf(static_cast<float>(head_dim)));

    // ============================================================
    // Pass 1: Find Max Score (m)
    // ============================================================
    AccT m_max = -INFINITY;

    for (int s_idx = 0; s_idx < src_len; ++s_idx) {
        if (is_causal && s_idx > t_idx) continue;  // causal mask：上三角置零

        AccT k_val = 0.0;
        if (d_idx < head_dim) {
            size_t k_ptr = k_base_offset + (size_t)s_idx * kv_heads * head_dim + d_idx;
            k_val = static_cast<AccT>(K[k_ptr]);
        }

        AccT dot_frag = q_val * k_val;
        AccT score = blockReduceSum(dot_frag);
        
        if (threadIdx.x == 0) {
            score *= scale;  // 对齐 torch.scaled_dot_product_attention 的缩放行为
            if (score > m_max) m_max = score;
        }
    }

    // 广播 Max Score 到所有线程
    __shared__ AccT shared_max;
    if (threadIdx.x == 0) shared_max = m_max;
    __syncthreads();
    m_max = shared_max;

    // ============================================================
    // Pass 2: Compute Denominator (d) and Output (o)
    // ============================================================
    AccT d_sum = 0.0;
    AccT o_accum = 0.0;

    for (int s_idx = 0; s_idx < src_len; ++s_idx) {
        if (is_causal && s_idx > t_idx) continue;  // causal mask：上三角置零

        // 重新计算 Score (牺牲性能换精度)
        AccT k_val = 0.0;
        if (d_idx < head_dim) {
            size_t k_ptr = k_base_offset + (size_t)s_idx * kv_heads * head_dim + d_idx;
            k_val = static_cast<AccT>(K[k_ptr]);
        }

        AccT dot_frag = q_val * k_val;
        AccT score = blockReduceSum(dot_frag);

        // 广播 Score
        __shared__ AccT shared_score;
        if (threadIdx.x == 0) {
            shared_score = score * scale;  // 对齐 torch.scaled_dot_product_attention 的缩放行为
        }
        __syncthreads();
        score = shared_score;

        // 计算 Exp
        AccT exp_score = exp(score - m_max);
        
        // 累加 Denominator (仅由线程 0 维护即可，但为了代码结构一致，所有线程都算一下也无妨，或者最后广播)
        d_sum += exp_score;

        // 读取 Value 并累加 Output
        AccT v_val = 0.0;
        if (d_idx < head_dim) {
            size_t v_ptr = v_base_offset + (size_t)s_idx * kv_heads * head_dim + d_idx;
            v_val = static_cast<AccT>(V[v_ptr]);
        }
        o_accum += v_val * exp_score;
    }

    // 3. 写入结果
    if (d_idx < head_dim) {
        AccT result = (d_sum > 0.0) ? (o_accum / d_sum) : 0.0;
        O[o_offset + d_idx] = static_cast<T>(result);
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // 计算总元素数量
  size_t q_size = h_q.size() * sizeof(T);
  size_t k_size = h_k.size() * sizeof(T);
  size_t v_size = h_v.size() * sizeof(T);
  size_t o_size = h_o.size() * sizeof(T); // h_o 已经被 resize 过了

  // 1. 分配设备内存
  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, q_size));
  CHECK_CUDA(cudaMalloc(&d_k, k_size));
  CHECK_CUDA(cudaMalloc(&d_v, v_size));
  CHECK_CUDA(cudaMalloc(&d_o, o_size));

  // 2. 拷贝数据到设备
  CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));
  
  // 3. 启动 Kernel
  // Grid 配置: [Target Seq Len, Query Heads, Batch Size]
  dim3 grid(target_seq_len, query_heads, batch_size);
  
  // Block 配置: [Head Dim]
  // 确保 Block 大小是 32 的倍数以便于 Warp Reduction，且能覆盖 head_dim
  // 假设 head_dim <= 1024。常见值为 64, 128。
  int block_size = (head_dim + 31) / 32 * 32;
  if (block_size < 32) block_size = 32;
  if (block_size > 1024) block_size = 1024; // 硬件限制
  
  if constexpr (std::is_same<T, float>::value) {
    // float 路径：使用串行点积版本以最大程度匹配参考实现
    flash_attention_kernel_float<<<grid, block_size>>>(
        reinterpret_cast<const float*>(d_q),
        reinterpret_cast<const float*>(d_k),
        reinterpret_cast<const float*>(d_v),
        reinterpret_cast<float*>(d_o),
        target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal);
  } else {
    // half 路径：使用并行归约版本，利用 blockReduceSum 提升吞吐
    flash_attention_kernel<<<grid, block_size>>>(
        d_q, d_k, d_v, d_o,
        target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal);
  }
  
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // 4. 拷贝结果回主机
  CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));

  // 5. 释放内存
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_o));
}
////////// 作业题2 flashAttention END //////////

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
