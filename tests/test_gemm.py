import torch
import triton
import tritonblas as tb
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_current_target().backend


def test_gemm():
    M = 512
    N = 512
    K = 512

    alpha = 0.5
    beta = 100

    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.randn((M, N), device=DEVICE, dtype=torch.float16)

    triton_output = tb.gemm(a, b, c=c, alpha=alpha, beta=beta)
    torch_output = torch.addmm(c, a, b, alpha=alpha, beta=beta)
    triton.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=get_rtol())
