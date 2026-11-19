"""Fused log-domain Sinkhorn iterations for small batched transport problems (Triton, CUDA).

Each iteration is one kernel over (problem, chunk of rows) programs, so every
problem is spread over many streaming multiprocessors. A program first
recovers the column potentials g from the column log-sum-exp partials that
the previous iteration left per chunk, then updates the row potentials f of
its rows and writes its own partials of LSE_i((f_i - C_ij) / eps). The
updates are those of ``models.common._sinkhorn_iteration`` (f first, then g
from the new f), so results match the eager loop up to floating-point
rounding. Blocks of iterations are captured as CUDA graphs, since launching
the kernels takes longer than running them.
"""

import threading

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None

BLOCK_ROWS = 64
NUM_WARPS = 4
FIXED_BUDGET_BLOCK = 50  # Iterations per graph when no tolerance is checked.


if triton is not None:
    @triton.jit
    def _column_potentials(log_q, pmax_ptr, psum_ptr, g_ptr, b, C, K, eps, cols, cmask,
                           BLOCK_C: tl.constexpr, FROM_G: tl.constexpr):
        if FROM_G:
            g = tl.load(g_ptr + b * K + cols, mask=cmask, other=0.0)
        else:
            chunks = tl.arange(0, BLOCK_C)
            mask = (chunks < C)[:, None] & cmask[None, :]
            offsets = b * C * K + chunks[:, None] * K + cols[None, :]
            pmax = tl.load(pmax_ptr + offsets, mask=mask, other=float("-inf"))
            psum = tl.load(psum_ptr + offsets, mask=mask, other=0.0)
            top = tl.where(cmask, tl.max(pmax, axis=0), 0.0)
            total = tl.sum(psum * tl.exp(pmax - top[None, :]), axis=0)
            g = tl.where(cmask, eps * (log_q - top - tl.log(total)), 0.0)
        return g

    @triton.jit
    def _iteration_kernel(cost_ptr, log_p_ptr, log_q_ptr, f_ptr, g_ptr, pmax_in, psum_in, pmax_out, psum_out,
                          N, K, C, eps, inv_eps,
                          BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, FROM_G: tl.constexpr):
        b = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1)
        cols = tl.arange(0, BLOCK_K)
        cmask = cols < K
        log_q = tl.load(log_q_ptr + b * K + cols, mask=cmask, other=0.0)
        g = _column_potentials(log_q, pmax_in, psum_in, g_ptr, b, C, K, eps, cols, cmask, BLOCK_C, FROM_G)
        rows = c * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < N
        mask = rmask[:, None] & cmask[None, :]
        cost = tl.load(cost_ptr + b * N * K + rows[:, None] * K + cols[None, :], mask=mask, other=0.0)
        # f_i = eps * (log p_i - LSE_j((g_j - C_ij) / eps))
        x = tl.where(mask, (g[None, :] - cost) * inv_eps, float("-inf"))
        row_max = tl.where(rmask, tl.max(x, axis=1), 0.0)
        lse = row_max + tl.log(tl.sum(tl.exp(x - row_max[:, None]), axis=1))
        f = eps * (tl.load(log_p_ptr + b * N + rows, mask=rmask, other=0.0) - lse)
        tl.store(f_ptr + b * N + rows, f, mask=rmask)
        # This chunk's share of LSE_i((f_i - C_ij) / eps), as (max, sum of exp below it).
        y = tl.where(mask, (f[:, None] - cost) * inv_eps, float("-inf"))
        col_max = tl.where(cmask, tl.max(y, axis=0), 0.0)
        col_sum = tl.sum(tl.exp(y - col_max[None, :]), axis=0)
        out = b * C * K + c * K + cols
        tl.store(pmax_out + out, col_max, mask=cmask)
        tl.store(psum_out + out, col_sum, mask=cmask)

    @triton.jit
    def _finalize_kernel(cost_ptr, log_p_ptr, log_q_ptr, f_ptr, g_ptr, pmax_ptr, psum_ptr, err_ptr,
                         N, K, C, eps, inv_eps,
                         BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, CHECK: tl.constexpr):
        b = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1)
        cols = tl.arange(0, BLOCK_K)
        cmask = cols < K
        log_q = tl.load(log_q_ptr + b * K + cols, mask=cmask, other=0.0)
        g = _column_potentials(log_q, pmax_ptr, psum_ptr, g_ptr, b, C, K, eps, cols, cmask, BLOCK_C, False)
        if c == 0:
            tl.store(g_ptr + b * K + cols, g, mask=cmask)
        if CHECK:
            # Relative row-marginal error of the plan (f, g), as in the eager loop.
            rows = c * BLOCK_R + tl.arange(0, BLOCK_R)
            rmask = rows < N
            mask = rmask[:, None] & cmask[None, :]
            cost = tl.load(cost_ptr + b * N * K + rows[:, None] * K + cols[None, :], mask=mask, other=0.0)
            x = tl.where(mask, (g[None, :] - cost) * inv_eps, float("-inf"))
            row_max = tl.where(rmask, tl.max(x, axis=1), 0.0)
            lse = row_max + tl.log(tl.sum(tl.exp(x - row_max[:, None]), axis=1))
            f = tl.load(f_ptr + b * N + rows, mask=rmask, other=0.0)
            log_row = f * inv_eps + lse - tl.load(log_p_ptr + b * N + rows, mask=rmask, other=0.0)
            err = tl.max(tl.where(rmask, tl.abs(tl.exp(log_row) - 1.0), 0.0), axis=0)
            tl.store(err_ptr + b * C + c, err)


class _Solver:
    """Static buffers and captured graphs for one problem size and device."""

    def __init__(self, batch, n, k, device):
        self.batch, self.n, self.k = batch, n, k
        self.block_r = min(BLOCK_ROWS, triton.next_power_of_2(n))
        self.chunks = triton.cdiv(n, self.block_r)
        self.meta = dict(BLOCK_R=self.block_r, BLOCK_K=triton.next_power_of_2(k),
                         BLOCK_C=triton.next_power_of_2(self.chunks), num_warps=NUM_WARPS)
        options = dict(device=device, dtype=torch.float32)
        self.cost = torch.empty(batch, n, k, **options)
        self.log_p = torch.empty(batch, n, **options)
        self.log_q = torch.empty(batch, k, **options)
        self.f = torch.zeros(batch, n, **options)
        self.g = torch.zeros(batch, k, **options)
        self.pmax = torch.empty(2, batch, self.chunks, k, **options)
        self.psum = torch.empty_like(self.pmax)
        self.err = torch.zeros(batch, self.chunks, **options)
        self.graphs = {}

    def _launch(self, eps, count, check):
        sizes = (self.n, self.k, self.chunks, float(eps), 1.0 / float(eps))
        buffers = (self.cost, self.log_p, self.log_q, self.f, self.g)
        for i in range(count):
            source, target = (i - 1) % 2, i % 2
            _iteration_kernel[(self.batch, self.chunks)](
                *buffers, self.pmax[source], self.psum[source], self.pmax[target], self.psum[target], *sizes,
                FROM_G=i == 0, **self.meta)
        last = (count - 1) % 2
        _finalize_kernel[(self.batch, self.chunks if check else 1)](
            *buffers, self.pmax[last], self.psum[last], self.err, *sizes, CHECK=check, **self.meta)

    def run(self, eps, count, check, use_graph):
        """``count`` iterations from the potentials g in ``self.g``; leaves f, g (and the error) in place."""
        if not use_graph:
            self._launch(eps, count, check)
            return
        key = (float(eps), count, check)
        graph = self.graphs.get(key)
        if graph is None:
            saved = self.g.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._launch(eps, count, check)  # Compiles the kernels outside the capture.
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self._launch(eps, count, check)
            self.g.copy_(saved)
            self.graphs[key] = graph
        graph.replay()


_SOLVERS = {}
_SOLVERS_LOCK = threading.Lock()


def fused_sinkhorn_supported(cost, p, q):
    """Whether ``fused_sinkhorn`` can solve this problem."""
    return (triton is not None and cost.is_cuda and cost.dim() >= 2
            and cost.dtype == p.dtype == q.dtype == torch.float32
            and tuple(p.shape) == tuple(cost.shape[:-1])
            and tuple(q.shape) == tuple(cost.shape[:-2]) + (cost.shape[-1],))


def fused_sinkhorn(cost, log_p, log_q, schedule, epsilon, iterations, thresh=0.0, check_every=1, use_graph=True):
    """Dual potentials (f, g) after the Sinkhorn iterations of ``models.common.sinkhorn``.

    Starting from zero potentials, runs one iteration at each entropy of
    ``schedule`` and then up to ``iterations`` at ``epsilon``. With ``thresh``
    > 0, the relative row-marginal error is measured after every
    ``check_every`` iterations and the loop stops at the first check whose
    largest error (over all problems) is below ``thresh``; a final partial
    block of fewer than ``check_every`` iterations is not checked.
    """
    n, k = cost.shape[-2:]
    batch = cost.numel() // (n * k)
    key = (batch, n, k, cost.device)
    with _SOLVERS_LOCK:
        solver = _SOLVERS.get(key)
        if solver is None:
            solver = _SOLVERS[key] = _Solver(batch, n, k, cost.device)
    with torch.cuda.device(cost.device):
        solver.cost.copy_(cost.reshape(batch, n, k))
        solver.log_p.copy_(log_p.reshape(batch, n))
        solver.log_q.copy_(log_q.reshape(batch, k))
        solver.g.zero_()
        for eps in schedule:
            solver.run(eps, 1, False, use_graph=False)
        remaining = iterations
        if thresh > 0:
            while remaining >= check_every:
                solver.run(epsilon, check_every, True, use_graph)
                remaining -= check_every
                if solver.err.amax().item() < thresh:
                    remaining = 0
            if remaining:
                solver.run(epsilon, remaining, False, use_graph)
        else:
            while remaining:
                count = min(remaining, FIXED_BUDGET_BLOCK)
                solver.run(epsilon, count, False, use_graph)
                remaining -= count
        return solver.f.clone().reshape(log_p.shape), solver.g.clone().reshape(log_q.shape)
