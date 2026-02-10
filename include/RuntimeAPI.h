/*
 * neuron-shim: Drop-in replacement for MediaTek Neuron Runtime
 * 
 * This header mirrors the public API of libneuronrt.so so that
 * applications compiled against the MediaTek Neuron Runtime can
 * be redirected to TFLite (CPU/GPU) or ONNX Runtime without
 * recompilation — just LD_PRELOAD or replace the .so.
 *
 * API surface reconstructed from:
 *   - MediaTek IoT Yocto Neuron Runtime API V1 docs
 *   - MediaTek IoT Yocto Neuron Runtime API V2 docs
 *   - NeuroPilot SDK public documentation
 */

#ifndef NEURON_RUNTIME_API_H
#define NEURON_RUNTIME_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Error codes                                                         */
/* ------------------------------------------------------------------ */
typedef enum {
    NEURONRUNTIME_NO_ERROR              = 0,
    NEURONRUNTIME_BAD_DATA              = 1,
    NEURONRUNTIME_BAD_STATE             = 2,
    NEURONRUNTIME_UNEXPECTED_NULL       = 3,
    NEURONRUNTIME_INCOMPLETE            = 4,
    NEURONRUNTIME_OUTPUT_INSUFFICIENT   = 5,
    NEURONRUNTIME_UNAVAILABLE           = 6,
    NEURONRUNTIME_OP_FAILED             = 7,
    NEURONRUNTIME_UNMAPPABLE            = 8,
} NeuronRuntimeError;

/* ------------------------------------------------------------------ */
/* Priority / QoS                                                      */
/* ------------------------------------------------------------------ */
typedef enum {
    NEURONRUNTIME_PRIORITY_LOW  = 0,
    NEURONRUNTIME_PRIORITY_MED  = 1,
    NEURONRUNTIME_PRIORITY_HIGH = 2,
} NeuronRuntimePriority;

typedef struct {
    NeuronRuntimePriority priority;
    uint32_t              boostValue;    /* 0..100, hint for freq scaling */
    uint64_t              abortTime;     /* ns, 0 = no abort */
    uint64_t              deadline;      /* ns, 0 = no deadline */
    void*                 profiledQoSData;
    uint32_t              profiledQoSDataSize;
} QoSOptions;

/* ------------------------------------------------------------------ */
/* Runtime configuration                                               */
/* ------------------------------------------------------------------ */

/* Suppress certain hardware features — shim ignores these */
typedef enum {
    NEURONRUNTIME_SUPPRESS_NONE  = 0,
    NEURONRUNTIME_SUPPRESS_MDLA  = 1 << 0,
    NEURONRUNTIME_SUPPRESS_VPU   = 1 << 1,
} NeuronRuntimeSuppressFeature;

typedef struct {
    uint32_t    flags;           /* reserved */
    uint32_t    suppress;        /* bitmask of NeuronRuntimeSuppressFeature */
} RuntimeConfig;

/* ------------------------------------------------------------------ */
/* Tensor info (for querying input/output metadata)                    */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t  dimensions[8];
    uint32_t  dimensionCount;
    uint32_t  type;              /* 0 = float32, 1 = uint8, 2 = int8, etc. */
    float     scale;
    int32_t   zeroPoint;
    size_t    sizeBytes;
} NeuronTensorInfo;

/* Opaque handle */
typedef void* NeuronRuntime;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
int NeuronRuntime_create(const RuntimeConfig* config, NeuronRuntime* runtime);
int NeuronRuntime_release(NeuronRuntime runtime);

/* ------------------------------------------------------------------ */
/* Model loading                                                       */
/* ------------------------------------------------------------------ */
int NeuronRuntime_loadNetworkFromFile(NeuronRuntime runtime, const char* path);
int NeuronRuntime_loadNetworkFromBuffer(NeuronRuntime runtime,
                                        const void* buffer, size_t size);

/* ------------------------------------------------------------------ */
/* Input / Output                                                      */
/* ------------------------------------------------------------------ */
int NeuronRuntime_setInput(NeuronRuntime runtime,
                           int index,
                           const void* buffer,
                           size_t size,
                           int padding);  /* typically {-1} */

int NeuronRuntime_setOutput(NeuronRuntime runtime,
                            int index,
                            void* buffer,
                            size_t size,
                            int padding);

int NeuronRuntime_getInputCount(NeuronRuntime runtime, uint32_t* count);
int NeuronRuntime_getOutputCount(NeuronRuntime runtime, uint32_t* count);

int NeuronRuntime_getInputInfo(NeuronRuntime runtime,
                               int index,
                               NeuronTensorInfo* info);
int NeuronRuntime_getOutputInfo(NeuronRuntime runtime,
                                int index,
                                NeuronTensorInfo* info);

/* Required size queries */
int NeuronRuntime_getInputSize(NeuronRuntime runtime, int index, size_t* size);
int NeuronRuntime_getOutputSize(NeuronRuntime runtime, int index, size_t* size);

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */
int NeuronRuntime_inference(NeuronRuntime runtime);

/* ------------------------------------------------------------------ */
/* QoS (all no-ops in shim, return success)                            */
/* ------------------------------------------------------------------ */
int NeuronRuntime_setQoSOption(NeuronRuntime runtime, const QoSOptions* qos);
int NeuronRuntime_getProfiledQoSData(NeuronRuntime runtime,
                                      QoSOptions* qos);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_RUNTIME_API_H */
