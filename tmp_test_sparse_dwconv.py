"""
tmp_test_sparse_dwconv.py  ---  TEMPORARY sanity-check, DELETE after use.

Goal: convince ourselves the sparse depthwise conv (and the sparse ConvNeXtV2
Block built on it) does the *correct* thing under FCMAE-style masking. Three
worries, three checks:

  (1) MASKING IS PRESERVED (submanifold property)
      Submanifold sparse convs must NOT grow the active set. After a Block, the
      densified output must be exactly 0 at every masked site and the active-site
      coordinate set must be identical to the input's. If masking leaked, masked
      pixels would become nonzero -> the encoder would "see" what it shouldn't.

  (2) IT IS ACTUALLY DEPTHWISE (no cross-channel mixing)
      For SparseDepthwiseConv alone, perturbing input channel c must change ONLY
      output channel c. If other channels move, the per-channel ModuleList is
      leaking and it isn't depthwise.

  (3) THE CONVOLUTION MATH IS RIGHT
      SparseDepthwiseConv on the (zeroed) dense input, restricted to active
      sites, must equal a plain grouped nn.Conv2d (groups=C, bias=False) using
      the SAME per-channel kernels. With masked sites = 0 and no bias, the dense
      grouped conv at active sites equals the submanifold result (inactive
      neighbors contribute 0 either way).

CUDA required -- spconv's sparse forward asserts "implicit gemm only support cuda".

Run from project root:
    venv\\Scripts\\python.exe tmp_test_sparse_dwconv.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as sp

from src.models.convnext.convnextv2_sparse import Block
from src.models.convnext.utils import SparseDepthwiseConv

torch.manual_seed(0)

ATOL = 1e-4
RTOL = 1e-4


def make_masked_sparse(N, C, H, W, keep_ratio, device):
    """Build a dense (N,C,H,W) tensor, zero out (mask) a random set of *spatial*
    sites across all channels, and convert to a SparseConvTensor exactly the way
    the model does (from_dense treats all-zero vectors as inactive).

    Returns (dense_masked, sparse, keep_mask) where keep_mask is (N,H,W) bool,
    True = kept/active.
    """
    dense = torch.ones(N, C, H, W, device=device)
    # random spatial keep mask, shared across channels (mirrors FCMAE patch mask)
    keep = (torch.rand(N, H, W, device=device) < keep_ratio)
    dense = dense * keep.unsqueeze(1)                       # zero masked sites
    # from_dense expects (N,H,W,C); all-zero channel-vector => inactive site
    sparse = sp.SparseConvTensor.from_dense(dense.permute(0, 2, 3, 1).contiguous())
    return dense, sparse, keep


def coord_set(sptensor):
    """Set of (b,y,x) active coordinates as python tuples."""
    return {tuple(row.tolist()) for row in sptensor.indices.cpu()}


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def test_masking_preserved(device):
    print("\n(1) masking preserved through a full sparse Block")
    N, C, H, W = 2, 16, 16, 16
    dense, sparse, keep = make_masked_sparse(N, C, H, W, keep_ratio=0.4, device=device)

    block = Block(dim=C).to(device).eval()
    with torch.no_grad():
        out = block(sparse)
    out_dense = out.dense()                                 # (N,C,H,W)

    in_coords = coord_set(sparse)
    out_coords = coord_set(out)

    ok = True
    ok &= check("active-site set unchanged (no growth/shrink)", in_coords == out_coords)

    masked = ~keep                                          # (N,H,W)
    masked_vals = out_dense.permute(0, 2, 3, 1)[masked]     # (#masked, C)
    ok &= check("densified output is exactly 0 at masked sites",
                torch.count_nonzero(masked_vals).item() == 0)

    kept_vals = out_dense.permute(0, 2, 3, 1)[keep]
    ok &= check("densified output is NOT all-zero at kept sites (block is alive)",
                torch.count_nonzero(kept_vals).item() > 0)
    return ok


def test_depthwise_independence(device):
    print("\n(2) SparseDepthwiseConv is truly depthwise (channel independence)")
    N, C, H, W = 1, 12, 16, 16
    dense, sparse, keep = make_masked_sparse(N, C, H, W, keep_ratio=0.5, device=device)

    dw = SparseDepthwiseConv(C, 7, 3).to(device).eval()
    with torch.no_grad():
        base = dw(sparse).dense()                           # (N,C,H,W)

    # perturb ONLY channel c0, ONLY at already-active sites (keeps active set fixed)
    c0 = 5
    dense_p = dense.clone()
    pert = torch.randn(N, H, W, device=device) * keep       # nonzero only at kept sites
    dense_p[:, c0] = dense_p[:, c0] + pert
    sparse_p = sp.SparseConvTensor.from_dense(dense_p.permute(0, 2, 3, 1).contiguous())

    ok = True
    ok &= check("active set unchanged after perturbing active sites",
                coord_set(sparse) == coord_set(sparse_p))

    with torch.no_grad():
        pert_out = dw(sparse_p).dense()

    diff = (pert_out - base).abs()
    moved = diff.flatten(2).amax(dim=2).squeeze(0)          # (C,) max change per out-channel

    other = torch.cat([moved[:c0], moved[c0 + 1:]])
    ok &= check(f"output channel {c0} changed (it should)", moved[c0].item() > 1e-6)
    ok &= check("all OTHER output channels unchanged (no leakage)",
                other.max().item() < 1e-6)
    if other.max().item() >= 1e-6:
        print(f"        max leakage into other channels: {other.max().item():.3e}")
    return ok


def test_matches_dense_grouped_conv(device):
    print("\n(3) SparseDepthwiseConv == dense grouped Conv2d on active sites")
    N, C, H, W = 1, 8, 16, 16
    dense, sparse, keep = make_masked_sparse(N, C, H, W, keep_ratio=0.5, device=device)

    dw = SparseDepthwiseConv(C, 7, 3).to(device).eval()
    with torch.no_grad():
        sp_out = dw(sparse).dense()                         # (N,C,H,W)

    # pull each per-channel spconv kernel into a dense grouped-conv weight.
    # spconv SubMConv2d weight is KRSC: (out=1, kH, kW, in=1) -> squeeze to (kH,kW).
    kernels = []
    for conv in dw.convs:
        w = conv.weight.detach()
        kernels.append(w.reshape(7, 7))
    Wdense = torch.stack(kernels).reshape(C, 1, 7, 7).to(device)   # (C,1,kH,kW)

    def grouped(weight):
        return F.conv2d(dense, weight, bias=None, padding=3, groups=C)

    # try both spatial orientations -- spconv kernel offset ordering vs torch
    # cross-correlation may differ by a flip; the correct one must match exactly.
    err_asis = (grouped(Wdense) - sp_out).abs()
    err_flip = (grouped(torch.flip(Wdense, dims=[2, 3])) - sp_out).abs()

    # only compare at ACTIVE sites (submanifold defines output only there)
    m = keep.unsqueeze(1).expand_as(sp_out)
    e_asis = err_asis[m].max().item()
    e_flip = err_flip[m].max().item()
    best = min(e_asis, e_flip)
    which = "as-is" if e_asis <= e_flip else "flipped"

    print(f"        max active-site error: as-is={e_asis:.3e}  flipped={e_flip:.3e}  (matched: {which})")
    return check("matches dense grouped conv at active sites", best < ATOL)


def _grid(t2d):
    """Pretty-print a 2D tensor as an aligned integer grid (dots for 0)."""
    rows = []
    for row in t2d.tolist():
        cells = []
        for v in row:
            iv = int(round(v))
            cells.append(" ." if iv == 0 else f"{iv:2d}")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def test_allones_visual(device):
    """Most transparent check possible: kernel = all 1s, input = all 1s with 0s at
    masked sites. Then for a submanifold depthwise conv:

        output[site] = (# of ACTIVE sites inside the 7x7 window)   if site is active
        output[site] = 0                                            if site is masked

    so the printed output grid is literally a per-pixel 'active-neighbor count',
    and every masked cell must be a dot (0). We print input mask, the sparse
    output, the hand-computed reference, and their diff.
    """
    print("\n(0) all-ones kernel / all-ones input -> output = active-neighbor count")
    N, C, H, W = 1, 1, 8, 8
    K, P = 7, 3

    dense = torch.ones(N, C, H, W, device=device)
    # fixed, legible mask (not random) so the printout is reproducible
    keep = torch.ones(N, H, W, device=device)
    keep[0, 0, :] = 0          # blank top row
    keep[0, :, 0] = 0          # blank left col
    keep[0, 3:5, 3:5] = 0      # a 2x2 hole in the middle
    dense = dense * keep.unsqueeze(1)
    sparse = sp.SparseConvTensor.from_dense(dense.permute(0, 2, 3, 1).contiguous())

    dw = SparseDepthwiseConv(C, K, P).to(device).eval()
    with torch.no_grad():
        for conv in dw.convs:                       # force kernel := all ones
            conv.weight.copy_(torch.ones_like(conv.weight))
        sp_out = dw(sparse).dense()                  # (N,C,H,W)

    # reference: count active neighbors via dense conv of the keep-mask, then
    # zero it out at masked sites (submanifold only emits at active sites).
    ref = F.conv2d(keep.unsqueeze(1), torch.ones(1, 1, K, K, device=device), padding=P)
    ref = ref * keep.unsqueeze(1)

    out2d, ref2d = sp_out[0, 0], ref[0, 0]

    print("\n  input keep-mask (1 = active site, . = masked):")
    print(_grid(keep[0]))
    print("\n  sparse depthwise output (active-neighbor count, . = 0):")
    print(_grid(out2d))
    print("\n  reference  (dense count of active neighbors, masked->0):")
    print(_grid(ref2d))

    diff = (out2d - ref2d).abs()
    print(f"\n  max |sparse - reference| = {diff.max().item():.3e}")

    ok = True
    masked = (keep[0] == 0)
    ok &= check("every masked cell is exactly 0", out2d[masked].abs().max().item() == 0)
    ok &= check("output equals hand-computed active-neighbor count", diff.max().item() < ATOL)
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- spconv sparse forward is CUDA-only. Aborting.")
        raise SystemExit(1)
    device = torch.device("cuda")
    print(f"Running sparse depthwise sanity checks on {torch.cuda.get_device_name(0)}")

    results = {
        "all-ones visual": test_allones_visual(device),
        "masking preserved": test_masking_preserved(device),
        "depthwise independence": test_depthwise_independence(device),
        "matches dense grouped conv": test_matches_dense_grouped_conv(device),
    }

    print("\n==================== SUMMARY ====================")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    all_ok = all(results.values())
    print("=================================================")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
