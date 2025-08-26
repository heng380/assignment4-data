import torch
import triton
import triton.language as tl

# ==================== Triton Kernels ====================

@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, o_ptr,
    B_S, D, stride_x, stride_o,  # B_S = batch_size * seq_len
    eps,
    BLOCK_D: tl.constexpr,
):
    # pid_b: 哪一个 token (0 ~ B*S-1)
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)  # 按 D 分块

    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < D

    # 计算当前 token 的起始地址
    x_row_ptr = x_ptr + pid_b * stride_x + offsets_d
    w_ptr = w_ptr + offsets_d
    o_row_ptr = o_ptr + pid_b * stride_o + offsets_d

    x = tl.load(x_row_ptr, mask=mask_d, other=0.0)

    # 计算均方
    x_squared = x * x
    var = tl.sum(x_squared, axis=0) / D
    rstd = tl.math.rsqrt(var + eps)

    # 归一化 + 权重
    x_norm = x * rstd
    w = tl.load(w_ptr, mask=mask_d, other=1.0)
    output = x_norm * w

    tl.store(o_row_ptr, output, mask=mask_d)


@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, o_ptr,
    B_S, D, stride_x, stride_o,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < D

    x_row_ptr = x_ptr + pid_b * stride_x + offsets_d
    w_ptr = w_ptr + offsets_d
    o_row_ptr = o_ptr + pid_b * stride_o + offsets_d

    x = tl.load(x_row_ptr, mask=mask_d, other=0.0)

    # 均值
    mean = tl.sum(x, axis=0) / D
    x_minus_mean = x - mean
    var = tl.sum(x_minus_mean * x_minus_mean, axis=0) / D
    rstd = tl.math.rsqrt(var + eps)

    x_norm = x_minus_mean * rstd
    w = tl.load(w_ptr, mask=mask_d, other=1.0)
    output = x_norm * w

    tl.store(o_row_ptr, output, mask=mask_d)


# ==================== Python Wrapper ====================

def triton_rms_norm(x, weight, eps=1e-6):
    B, S, D = x.shape
    y = torch.empty_like(x)
    B_S = B * S

    # 2D Grid: (B*S, D_blocks)
    grid = (B_S, triton.cdiv(D, 1024))

    rms_norm_kernel[grid](
        x, weight, y,
        B_S, D, D, D, eps, BLOCK_D=1024
    )
    return y


def triton_layer_norm(x, weight, eps=1e-6):
    B, S, D = x.shape
    y = torch.empty_like(x)
    B_S = B * S

    grid = (B_S, triton.cdiv(D, 1024))

    layer_norm_kernel[grid](
        x, weight, y,
        B_S, D, D, D, eps, BLOCK_D=1024
    )
    return y


# ==================== Benchmark Function ====================

def benchmark_fn(fn, x, weight, num_warmup=100, num_steps=100):
    # 预热
    for _ in range(num_warmup):
        _ = fn(x, weight)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_steps):
        _ = fn(x, weight)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_steps  # ms


# ==================== 生成报告函数 ====================

def run_benchmark(name, param_name, param_values, fixed_params, fn_rms, fn_ln):
    print(f"\n" + "="*60)
    print(f"Benchmark: {name}")
    print("="*60)
    header = f"{param_name:<10} RMS (ms)   Layer (ms) Speedup (x)"
    print(header)
    print("-"*60)

    results = []
    B, S, D = fixed_params

    for val in param_values:
        if param_name == "Batch Size":
            batch_size = val
            seq_len, hidden_size = S, D
        elif param_name == "Seq Len":
            seq_len = val
            batch_size, hidden_size = B, D
        else:  # Hidden Size
            hidden_size = val
            batch_size, seq_len = B, S

        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
        weight = torch.ones(hidden_size, device='cuda')

        rms_time = benchmark_fn(fn_rms, x, weight)
        layer_time = benchmark_fn(fn_ln, x, weight)
        speedup = layer_time / rms_time

        print(f"{val:<10} {rms_time:<10.3f} {layer_time:<10.3f} {speedup:<10.3f}")
        results.append((val, rms_time, layer_time, speedup))

    return results


# ==================== 运行三个报告 ====================

if __name__ == "__main__":
    # 固定参数基准
    BASE_BATCH = 8
    BASE_SEQ = 2048
    BASE_HIDDEN = 4096

    print("Triton RMSNorm vs LayerNorm Performance Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 报告 1: 变化 Batch Size
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    run_benchmark(
        name="Varying Batch Size",
        param_name="Batch Size",
        param_values=batch_sizes,
        fixed_params=[BASE_BATCH, BASE_SEQ, BASE_HIDDEN],
        fn_rms=triton_rms_norm,
        fn_ln=triton_layer_norm
    )

    # 报告 2: 变化 Seq Len
    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    run_benchmark(
        name="Varying Sequence Length",
        param_name="Seq Len",
        param_values=seq_lens,
        fixed_params=[BASE_BATCH, BASE_SEQ, BASE_HIDDEN],
        fn_rms=triton_rms_norm,
        fn_ln=triton_layer_norm
    )

    # 报告 3: 变化 Hidden Size
    hidden_sizes = [256, 512, 1024, 2048, 4096, 8192]
    run_benchmark(
        name="Varying Hidden Size",
        param_name="Hidden Size",
        param_values=hidden_sizes,
        fixed_params=[BASE_BATCH, BASE_SEQ, BASE_HIDDEN],
        fn_rms=triton_rms_norm,
        fn_ln=triton_layer_norm
    )

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)