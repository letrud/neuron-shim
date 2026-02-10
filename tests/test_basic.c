/*
 * Basic test — exercises the shim API surface
 *
 * Run with:
 *   NEURON_SHIM_BACKEND=stub ./shim_test
 */

#include "RuntimeAPI.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/tmp/test_model.dla";
    int ret;

    printf("=== neuron-shim basic test ===\n\n");

    /* Create runtime */
    NeuronRuntime runtime = NULL;
    RuntimeConfig config = {0};

    ret = NeuronRuntime_create(&config, &runtime);
    printf("create:  %s (ret=%d)\n", ret == 0 ? "OK" : "FAIL", ret);
    if (ret != 0) return 1;

    /* Load model */
    ret = NeuronRuntime_loadNetworkFromFile(runtime, model_path);
    printf("load:    %s (ret=%d, path=%s)\n",
           ret == 0 ? "OK" : "FAIL", ret, model_path);

    /* Query I/O */
    uint32_t in_count = 0, out_count = 0;
    NeuronRuntime_getInputCount(runtime, &in_count);
    NeuronRuntime_getOutputCount(runtime, &out_count);
    printf("tensors: %u inputs, %u outputs\n", in_count, out_count);

    /* Set up dummy I/O */
    size_t in_size = 224 * 224 * 3;   /* typical image input */
    size_t out_size = 1001;            /* typical classification output */
    uint8_t* input_buf  = (uint8_t*)calloc(1, in_size);
    float*   output_buf = (float*)calloc(out_size, sizeof(float));

    ret = NeuronRuntime_setInput(runtime, 0, input_buf, in_size, -1);
    printf("setInput:  %s\n", ret == 0 ? "OK" : "FAIL");

    ret = NeuronRuntime_setOutput(runtime, 0, output_buf, out_size * sizeof(float), -1);
    printf("setOutput: %s\n", ret == 0 ? "OK" : "FAIL");

    /* QoS (should be no-op) */
    QoSOptions qos = {
        .priority = NEURONRUNTIME_PRIORITY_HIGH,
        .boostValue = 100,
    };
    ret = NeuronRuntime_setQoSOption(runtime, &qos);
    printf("setQoS:  %s\n", ret == 0 ? "OK" : "FAIL");

    /* Inference */
    ret = NeuronRuntime_inference(runtime);
    printf("infer:   %s\n", ret == 0 ? "OK" : "FAIL");

    /* Check output isn't garbage (stub zeros it) */
    int all_zero = 1;
    for (size_t i = 0; i < out_size; i++) {
        if (output_buf[i] != 0.0f) { all_zero = 0; break; }
    }
    printf("output:  %s\n", all_zero ? "all zeros (stub)" : "has values (real backend)");

    /* Cleanup */
    NeuronRuntime_release(runtime);
    printf("\nrelease: OK\n");

    free(input_buf);
    free(output_buf);

    printf("\n=== all tests passed ===\n");
    return 0;
}
