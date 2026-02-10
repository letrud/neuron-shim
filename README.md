# neuron-shim

Drop-in replacement for MediaTek's Neuron Runtime (`libneuronrt.so`) and APU system library (`libapusys.so`). Redirects NPU inference calls to portable backends (TFLite CPU/GPU, ONNX Runtime, or stubs) so that applications built for MediaTek Genio/Dimensity platforms can run on standard ARM64 or x86_64 hardware.

## Architecture

```
┌─────────────────────────────────────────────┐
│            Target Application               │
│    (e.g. Ubiquiti camera daemon)            │
└──────────────┬──────────────────────────────┘
               │ NeuronRuntime_* calls
               │
┌──────────────▼──────────────────────────────┐
│         libneuronrt.so  (this shim)         │
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Model       │  │ Backend Selector     │  │
│  │ Resolver    │  │                      │  │
│  │ .dla→.onnx  │  │ ┌──────┐ ┌────────┐ │  │
│  │ .dla→.tflite│  │ │ ORT  │ │TFLite  │ │  │
│  │             │  │ │      │ │(CPU)   │ │  │
│  └─────────────┘  │ └──┬───┘ └────────┘ │  │
│                   │    │                 │  │
│                   │ ┌──▼──────────────┐  │  │
│                   │ │ Execution       │  │  │
│                   │ │ Providers:      │  │  │
│                   │ │ • CUDA (NVIDIA) │  │  │
│                   │ │ • TensorRT (NV) │  │  │
│                   │ │ • MIGraphX (AMD)│  │  │
│                   │ │ • CPU (fallback)│  │  │
│                   │ └─────────────────┘  │  │
│                   └──────────────────────┘  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         libapusys.so  (stub)                │
│   Returns success for all APU driver calls  │
└─────────────────────────────────────────────┘
```

## Quick Start

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

For cross-compiling to aarch64:
```bash
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake \
  -DTFLITE_DIR=/path/to/tflite-aarch64
```

### Run (stub backend — trace API calls)

```bash
NEURON_SHIM_BACKEND=stub \
LD_PRELOAD=./libneuronrt.so:./libapusys.so \
  /path/to/target_binary
```

This will log every NeuronRuntime call to stderr, showing you:
- Which model files the app loads
- Input/output tensor sizes
- Inference frequency

### Run (ONNX Runtime backend — GPU inference, NVIDIA or AMD)

```bash
# Point to your replacement .onnx models
NEURON_SHIM_BACKEND=onnx \
NEURON_SHIM_MODEL_DIR=/opt/models \
NEURON_SHIM_NUM_THREADS=4 \
LD_PRELOAD=./libneuronrt.so:./libapusys.so \
  /path/to/target_binary
```

ONNX Runtime auto-detects available GPUs:
- **NVIDIA**: picks TensorRT EP → CUDA EP → CPU (in priority order)
- **AMD**: picks MIGraphX EP → CPU

### Run (TFLite backend — CPU-only, lightweight)

```bash
NEURON_SHIM_BACKEND=tflite \
NEURON_SHIM_MODEL_DIR=/opt/models \
LD_PRELOAD=./libneuronrt.so:./libapusys.so \
  /path/to/target_binary
```

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `NEURON_SHIM_BACKEND` | `onnx`, `tflite`, `stub` | auto-detect | Inference backend (onnx preferred for GPU) |
| `NEURON_SHIM_SUFFIX` | `.onnx`, `.tflite` | auto (based on backend) | Suffix appended to .dla paths |
| `NEURON_SHIM_MODEL_DIR` | path | (empty = same dir as .dla) | Redirect model loading to this directory |
| `NEURON_SHIM_NUM_THREADS` | 1-N | 4 | CPU threads (ORT intra-op / TFLite) |
| `NEURON_SHIM_LOG_LEVEL` | 0-4 | 3 | 0=off, 1=error, 2=warn, 3=info, 4=debug |
| `NEURON_SHIM_FORCE_CPU` | 0/1 | 0 | Force CPU-only (skip GPU EP registration) |

## Model Resolution

The shim appends a suffix to the original `.dla` path:

```
Default (model_dir empty):
  App loads:  /usr/share/models/person_detect.dla
  Shim loads: /usr/share/models/person_detect.dla.onnx

With model_dir = /opt/models:
  App loads:  /usr/share/models/person_detect.dla
  Shim loads: /opt/models/person_detect.dla.onnx
```

The `model_dir` option is useful when:
- The original rootfs is mounted read-only
- You want all converted models in one place
- You're bind-mounting models into a Docker container

### Model conversion chain

```
.dla (MediaTek NPU) → .tflite (your DLA reverser) → .onnx (tf2onnx)
```

```bash
# Convert TFLite to ONNX (once you have the .tflite)
pip install tf2onnx
python -m tf2onnx.convert --tflite model.tflite --output model.onnx
```

## Docker Usage

### NVIDIA GPU
```bash
# Build
docker build -t neuron-shim-nvidia -f Dockerfile.nvidia .

# Run with GPU passthrough
docker run --gpus all --platform linux/arm64 \
  -v ./rootfs:/rootfs \
  -v ./models:/opt/models \
  -e NEURON_SHIM_BACKEND=onnx \
  -e NEURON_SHIM_MODEL_DIR=/opt/models \
  neuron-shim-nvidia \
  chroot /rootfs /usr/bin/camera_daemon
```

### AMD GPU
```bash
# Run with ROCm passthrough
docker run --device=/dev/kfd --device=/dev/dri \
  --platform linux/arm64 \
  --group-add video \
  -v ./rootfs:/rootfs \
  -v ./models:/opt/models \
  -e NEURON_SHIM_BACKEND=onnx \
  -e NEURON_SHIM_MODEL_DIR=/opt/models \
  neuron-shim-amd \
  chroot /rootfs /usr/bin/camera_daemon
```

### CPU-only (testing)
```bash
docker run --platform linux/arm64 \
  -v ./rootfs:/rootfs \
  -e NEURON_SHIM_BACKEND=onnx \
  -e NEURON_SHIM_FORCE_CPU=1 \
  neuron-shim \
  chroot /rootfs /usr/bin/camera_daemon
```

## Workflow

### Phase 1: Trace (figure out what the app needs)
```bash
NEURON_SHIM_BACKEND=stub NEURON_SHIM_LOG_LEVEL=4 \
LD_PRELOAD=... ./camera_daemon 2>trace.log

# See which models it loads and what I/O shapes it expects
grep -E "LOAD|SET_INPUT|SET_OUTPUT" trace.log
```

### Phase 2: Provide models
Either:
- Reverse `.dla` → `.tflite` (you mentioned building this tool)
- Find equivalent public models (MobileNet-SSD, YOLO, etc.)
- Place them where the resolver can find them

### Phase 3: Real inference
Switch to `NEURON_SHIM_BACKEND=tflite` and verify the app runs.

## Extending

### Adding a new backend (e.g. ONNX Runtime, TensorRT)

1. Create `src/backend_onnx.c` implementing the `NeuronShimBackend` interface
2. Add `neuron_shim_backend_onnx()` to `backend.h`
3. Register in `backend_selector.c`
4. Build with `-DONNXRUNTIME_DIR=/path/to/onnxruntime`

### Handling unknown API functions

If the target application calls NeuronRuntime functions not covered by this shim:

```bash
# Find all NeuronRuntime symbols the binary needs
readelf -d /path/to/binary | grep NEEDED
nm -D /path/to/binary | grep NeuronRuntime

# Compare with our exports
nm -D libneuronrt.so | grep " T "
```

Add missing functions to `shim_runtime.c` — most can just return `NEURONRUNTIME_NO_ERROR`.

## Files

```
neuron-shim/
├── CMakeLists.txt
├── README.md
├── config/
│   └── model_map.conf        # Example model path mappings
├── include/
│   ├── RuntimeAPI.h           # MediaTek Neuron Runtime API (reconstructed)
│   ├── backend.h              # Backend abstraction interface
│   └── model_resolver.h       # .dla → .tflite path resolution
├── src/
│   ├── shim_runtime.c         # Core NeuronRuntime_* implementation
│   ├── shim_apusys.c          # libapusys.so stub
│   ├── model_resolver.c       # Model path resolution logic
│   ├── backend_onnx.c         # ONNX Runtime backend (NVIDIA + AMD GPU)
│   ├── backend_tflite.c       # TFLite C API backend (CPU)
│   ├── backend_stub.c         # No-op backend for tracing
│   └── backend_selector.c     # Runtime backend selection
└── tests/
    └── test_basic.c           # API surface test
```

## License

MIT — use this however you want.
