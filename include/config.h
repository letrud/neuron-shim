/*
 * neuron-shim: Configuration
 *
 * Loads from (in priority order):
 *   1. Environment variables (NEURON_SHIM_*)
 *   2. ./neuron-shim.conf (current working directory)
 *   3. /etc/neuron-shim.conf
 *   4. Compiled-in defaults
 */

#ifndef NEURON_SHIM_CONFIG_H
#define NEURON_SHIM_CONFIG_H

#include <stdbool.h>

typedef struct {
    char backend[32];       /* auto | onnx | tflite | stub */
    char suffix[32];        /* auto | .onnx | .tflite */
    char model_dir[512];    /* empty = use original path, else redirect */
    int  threads;           /* CPU thread count */
    bool force_cpu;         /* skip GPU execution providers */
    int  log_level;         /* 0=off 1=err 2=warn 3=info 4=debug */
} NeuronShimConfig;

/*
 * Load configuration. Call once at init.
 * Returns pointer to global config (static lifetime).
 */
const NeuronShimConfig* neuron_shim_config_load(void);

/*
 * Get the resolved suffix for model files.
 * If config says "auto", returns suffix based on backend.
 */
const char* neuron_shim_config_get_suffix(const NeuronShimConfig* cfg);

#endif /* NEURON_SHIM_CONFIG_H */
