/*
 * neuron-shim: Backend selector
 */

#include "backend.h"

#include <string.h>
#include <stdio.h>
#include <dlfcn.h>

const NeuronShimBackend* neuron_shim_select_backend(const char* name) {
    /* Explicit selection */
    if (name) {
#ifdef SHIM_HAS_ONNX
        if (strcmp(name, "onnx") == 0) return neuron_shim_backend_onnx();
#endif
#ifdef SHIM_HAS_TFLITE
        if (strcmp(name, "tflite") == 0) return neuron_shim_backend_tflite();
#endif
        if (strcmp(name, "stub") == 0)   return neuron_shim_backend_stub();
        fprintf(stderr, "[neuron-shim] unknown backend '%s', falling back\n", name);
    }

    /*
     * Auto-detect priority:
     *   1. ONNX Runtime (preferred — supports NVIDIA + AMD GPU)
     *   2. TFLite (CPU or mobile GPU)
     *   3. Stub (no-op fallback)
     */
#ifdef SHIM_HAS_ONNX
    {
        void* handle = dlopen("libonnxruntime.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!handle)
            handle = dlopen("libonnxruntime.so", RTLD_LAZY);
        if (handle) {
            dlclose(handle);
            fprintf(stderr, "[neuron-shim] auto-selected: onnx\n");
            return neuron_shim_backend_onnx();
        }
    }
#endif

#ifdef SHIM_HAS_TFLITE
    {
        void* handle = dlopen("libtensorflowlite_c.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!handle)
            handle = dlopen("libtensorflowlite_c.so", RTLD_LAZY);
        if (handle) {
            dlclose(handle);
            fprintf(stderr, "[neuron-shim] auto-selected: tflite\n");
            return neuron_shim_backend_tflite();
        }
    }
#endif

    fprintf(stderr, "[neuron-shim] using stub backend\n");
    return neuron_shim_backend_stub();
}
