import torch
import triton

from utils import add_tritonblas_lib

add_tritonblas_lib()
import tritonblas as tb


DEVICE = triton.runtime.driver.active.get_current_target().backend


ref_lib = "cuBLAS" if DEVICE == "cuda" else "rocBLAS"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # Argument names to use as an x-axis for the plot
        x_names=["M", "N", "K"],
        # Different possible values for `x_name`
        x_vals=[128 * i for i in range(2, 33)],
        # Argument name whose value corresponds to a different line in the plot
        line_arg="provider",
        # Possible values for `line_arg`
        line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
        line_names=[ref_lib, "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp16",
        args={},
    )
)
def benchmark_gemm(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    alpha = 0.5
    beta = 100
    quantiles = [0.5, 0.2, 0.8]

    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.addmm(c, a, b, alpha=alpha, beta=beta), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tb.gemm(a, b, c=c, alpha=alpha, beta=beta), quantiles=quantiles
        )

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


def test_gemm(benchmark):
    def run_gemm_benchmark():
        benchmark_gemm.run(show_plots=False, print_data=True)

    benchmark(run_gemm_benchmark)


if __name__ == "__main__":
    benchmark_gemm.run(print_data=True, show_plots=False)
