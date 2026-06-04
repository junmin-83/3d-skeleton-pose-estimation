<#
.SYNOPSIS
  (Re)install the NVIDIA CUDA GPU inference stack for this project.

.DESCRIPTION
  rtmlib pulls in the CPU `onnxruntime`, which installs into the same
  `onnxruntime/` folder as `onnxruntime-gpu` and disables the CUDA provider
  when both are present. uv has no way to drop that transitive CPU dependency,
  so `uv sync` (and `uv sync --group gpu`) always reverts the env to CPU.

  This script repairs that in one step: it removes the CPU build and installs
  the verified GPU stack (onnxruntime-gpu + CUDA 12 wheels, cuDNN pinned <9.12),
  then verifies the CUDA provider loads. Run it after any bare `uv sync`, or for
  first-time GPU setup. Safe to run repeatedly (idempotent).

  Target: Windows x86_64 + NVIDIA GPU (CUDA 12). See README "(참고) GPU 가속".
#>

Write-Host "[setup-gpu] removing CPU onnxruntime (conflicts with onnxruntime-gpu)..."
uv pip uninstall onnxruntime   # no-op if not installed

Write-Host "[setup-gpu] installing onnxruntime-gpu + CUDA 12 wheels (cuDNN<9.12)..."
uv pip install onnxruntime-gpu "nvidia-cudnn-cu12<9.12" nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
if ($LASTEXITCODE -ne 0) { Write-Error "[setup-gpu] pip install failed"; exit 1 }

Write-Host "[setup-gpu] verifying CUDA provider..."
uv run --no-sync python -c "import onnxruntime as o; o.preload_dlls(); ps=o.get_available_providers(); print('providers:', ps); assert 'CUDAExecutionProvider' in ps, 'CUDA provider NOT available - check NVIDIA driver / wheels'; print('[setup-gpu] OK: GPU ready. Run demos with the default device (cuda); use --device cpu to force CPU.')"
exit $LASTEXITCODE
