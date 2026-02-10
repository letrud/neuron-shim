/*
 * neuron-shim: Configuration loader
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NeuronShimConfig g_config = {
    .backend   = "auto",
    .suffix    = "auto",
    .model_dir = "",       /* empty = use original path */
    .threads   = 4,
    .force_cpu = false,
    .log_level = 3,
};

/* ------------------------------------------------------------------ */
/* Parse a config file                                                 */
/* ------------------------------------------------------------------ */
static void parse_config_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        /* Skip comments and blanks */
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;

        char key[64], value[512];
        if (sscanf(p, "%63[^= ] = %511s", key, value) != 2) continue;

        if (strcmp(key, "backend") == 0) {
            strncpy(g_config.backend, value, sizeof(g_config.backend) - 1);
            g_config.backend[sizeof(g_config.backend) - 1] = '\0';
        } else if (strcmp(key, "suffix") == 0) {
            strncpy(g_config.suffix, value, sizeof(g_config.suffix) - 1);
            g_config.suffix[sizeof(g_config.suffix) - 1] = '\0';
        } else if (strcmp(key, "model_dir") == 0)
            snprintf(g_config.model_dir, sizeof(g_config.model_dir), "%s", value);
        else if (strcmp(key, "threads") == 0)
            g_config.threads = atoi(value);
        else if (strcmp(key, "force_cpu") == 0)
            g_config.force_cpu = (strcmp(value, "true") == 0 ||
                                  strcmp(value, "1") == 0);
        else if (strcmp(key, "log_level") == 0)
            g_config.log_level = atoi(value);
    }
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* Load config: file → env override                                    */
/* ------------------------------------------------------------------ */
const NeuronShimConfig* neuron_shim_config_load(void) {
    const char* env;

    /* Try config files (later overrides earlier) */
    parse_config_file("/etc/neuron-shim.conf");
    parse_config_file("./neuron-shim.conf");

    /* Environment variables override everything */
    env = getenv("NEURON_SHIM_BACKEND");
    if (env) { strncpy(g_config.backend, env, sizeof(g_config.backend) - 1); g_config.backend[sizeof(g_config.backend) - 1] = '\0'; }

    env = getenv("NEURON_SHIM_SUFFIX");
    if (env) { strncpy(g_config.suffix, env, sizeof(g_config.suffix) - 1); g_config.suffix[sizeof(g_config.suffix) - 1] = '\0'; }

    env = getenv("NEURON_SHIM_MODEL_DIR");
    if (env) snprintf(g_config.model_dir, sizeof(g_config.model_dir), "%s", env);

    env = getenv("NEURON_SHIM_NUM_THREADS");
    if (env) g_config.threads = atoi(env);

    env = getenv("NEURON_SHIM_FORCE_CPU");
    if (env) g_config.force_cpu = (strcmp(env, "1") == 0);

    env = getenv("NEURON_SHIM_LOG_LEVEL");
    if (env) g_config.log_level = atoi(env);

    return &g_config;
}

/* ------------------------------------------------------------------ */
/* Resolve "auto" suffix based on backend                              */
/* ------------------------------------------------------------------ */
const char* neuron_shim_config_get_suffix(const NeuronShimConfig* cfg) {
    /* Explicit suffix */
    if (strcmp(cfg->suffix, "auto") != 0)
        return cfg->suffix;

    /* Derive from backend */
    if (strcmp(cfg->backend, "tflite") == 0)
        return ".tflite";

    /* Default to .onnx (works for onnx, auto, and stub) */
    return ".onnx";
}
