import torch
import triton
import triton.profiler as proton
import argparse

from utils import add_tritonblas_lib

add_tritonblas_lib()
import tritonblas as tb
from tritonblas.level3.gemm import gemm_kernel


DEVICE = triton.runtime.driver.active.get_current_target().backend


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # Argument names to use as an x-axis for the plot
        x_names=["M", "N", "K"],
        # Different possible values for `x_name`
        x_vals=[128 * i for i in range(2, 33)],
        # Argument name whose value corresponds to a different line in the plot
        line_arg="provider",
        # Possible values for `line_arg`
        line_vals=["torch", "triton"],  # Label name for the lines
        line_names=["PyTorch", "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="gemm-performance",
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

    with proton.scope(f"gemm_{M}_{N}_{K}"):
        if provider == "torch":

            @proton.scope(
                "torch",
                metrics={
                    "flops": 2 * M * N * K,
                    "bytes": (M * N + N * K * M) * a.element_size(),
                },
            )
            def torch_gemm(c, a, b, alpha=1.0, beta=0.0):
                torch.addmm(c, a, b, alpha=alpha, beta=beta)

            if args.cudagraph:
                ms = triton.testing.do_bench_cudagraph(
                    lambda: torch_gemm(c, a, b, alpha=alpha, beta=beta)
                )
                min_ms = max_ms = ms
            else:
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: torch_gemm(c, a, b, alpha=alpha, beta=beta),
                    quantiles=quantiles,
                )

        if provider == "triton":

            def enter_autotune(args, reset_only=False):
                if reset_only:
                    return
                proton.enter_scope("<autotune>")

            def exit_autotune(args, exception):
                proton.exit_scope()

            gemm_kernel.pre_hook = enter_autotune
            gemm_kernel.post_hook = exit_autotune
            with proton.scope("triton"):
                if args.cudagraph:
                    ms = triton.testing.do_bench_cudagraph(
                        lambda: tb.gemm(a, b, c=c, alpha=alpha, beta=beta),
                    )
                    min_ms = max_ms = ms
                else:
                    ms, min_ms, max_ms = triton.testing.do_bench(
                        lambda: tb.gemm(a, b, c=c, alpha=alpha, beta=beta),
                        quantiles=quantiles,
                    )

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--profile", action="store_true")
    argparser.add_argument("--cudagraph", action="store_true", default=False)
    args = argparser.parse_args()

    if args.profile:
        proton.start("gemm", hook="triton")
        benchmark_gemm.run(show_plots=True, print_data=True)
        proton.finalize()
    else:
        benchmark_gemm.run(show_plots=True, print_data=True)
