# TransformerEngine Debug Module

A debug module provided to get deeper insights into the numerics of various tensors in TransformerEngine. The debug modules follow the same high-level structure of the main pytorch modules but are simplified and do not contain any optimizations.

Debug features are enabled using [debug-tools](https://gitlab-master.nvidia.com/anmolg/debug-tools) repo that provides easy-to-use APIs that enable/disable various features. Users can also add custom features and APIs, follow the README in the debug-tools repo for more information.

## Getting Started
- Clone the debug-tools repo and install it
```sh
git clone ssh://git@gitlab-master.nvidia.com:12051/anmolg/debug-tools.git
cd debug-tools
pip install .
```
- To enable `debug_api`, initialize the debug tool from global context once on every rank (for multi-gpu training).
```python
import debug_tool.api as debug_api

# If config.yaml is not found or None, all features are disabled it follows default run.
# transformer_engine extension loads APIs specific to transformer_engine
# Specify log_dir to store debug logs.
debug_api.initialize_debug(config="path/to/config.yaml", extensions=["transformer_engine"], log_dir=log_dir)
```
- The debug APIs are already placed in the debug modules, users just have to enable features using a config.yaml file to use the APIs.
- Provide names to TE modules during initialization to log with names. If name isn't provided, it uses a generic name such as Layer_1.
```python
Linear(name="model.decoder.layers.4.self_attention.linear_proj", ...)
```

## Supported Features
Use the config.yaml file to enable various features. Features are only enabled for selected layers. Layers are selected based on their name.\
Example configs are provided in `example_configs/`.

The following fields are available in the config to enable features.

### Layer Selection
Use the following fields to select layers.
```yaml
layers:
  layer_name_regex_pattern: .*(fc1|fc2) # choose FC1 and FC2 in all layers. **Recommended** way of layer selection.
  layer_numbers: [1-12] # choose layers with numbers 1 to 12. Use [all] to select all layers.
  layer_types: [qkv, mlp, fc1] # choose QKV, MLP, FC1. Use [all] to select all layer types.
```
- These are matched against the layer names provided to TE modules during initialization.
- Atleast one of `layer_numbers`, `layer_types`, or `layer_name_regex_pattern` is requried.
- Use 'all' for `layer_numbers` or `layer_types` to use all layers.
- Only `layer_numbers` and `layer_types` can be used together.

### Disabling FP8 gemms
Disable FP8 GEMMs for FP8 training.
```yaml
transformer_engine:
  disable_fp8_gemm:
    enabled: True
    gemms: [dgrad, wgrad, fprop] # default is None if not provided.
```
- Specify one of ['dgrad', 'fprop', 'wgrad'] as GEMM types.
- Specified GEMM will be run in high precision for FP8 training.


### Disabling FP8 layers
Disable FP8 entirely in selected layers for FP8 training.
```yaml
transformer_engine:
  disable_fp8_layer: 
    enabled: True # default = False if not provided.
```

### Per Tensor Dynamic Scaling
Enable dynamic (current scaling) instead of delayed scaling for FP8 training.
```yaml
transformer_engine:
  per_tensor_scaling:
    enabled: True
    gemms: [fprop, dgrad, wgrad] # both inputs to GEMMs will be dynamically scaled
    margin: 0 # absolute_max(tensor) = absolute_max(tensor) * (2^margin)
```

### Statistics Collection
- Supported loggers:
    - File (default)
    - Tensorboard
    - Custom logger: Reference implementation provided
    
To use tensorboard writer directly, pass the writer on the required rank using the `tb_writer` kwarg when initializing debug.
```python
debug_api.initialize_debug(config, extensions=["transformer_engine"], log_dir=".", tb_writer=tensorboard_writer)
```

- All debug logs containing layers' names and configs are written to `<log_dir/debug_logs/>`.

- If default metric logging is enabled, logs are written to `<log_dir/debug_statistics_logs/>`.

- To disable default statistic logging to file, use `disable_default_logging` kwarg when initializing debug.
```python
debug_api.initialize_debug(config, extensions=["transformer_engine"], disable_default_logging=True)
```

- Supported statistics:
    1. Generic: [max, min, mean, std, L1 norm, L2 norm]
    2. High precision: [Current amax, Dynamic Range]
    3. FP8 tensors: [Underflows after casting, Overflows after casting]

```yaml
collect_tensor_stats: # Generic logging, only enabled when this field is provided in config
  enabled: True
  type: [min, max, mean, std, l1_norm, l2_norm]
  tensors: [activation, gradient, weight]
  freq: 100 # every 100 training steps
  start_step: 1000 # start logging at this step
  end_step: 5000 # end logging at this step
transformer_engine:
  collect_tensor_stats: # TE high precision tensor logging, only enabled when this field is provided in config
    enabled: True
    type: [cur_amax, dynamic_range]
    tensors: [activation, gradient, weight]
    freq: 100 # every 100 training steps
    start_step: 1000 # start logging at this step
    end_step: 5000 # end logging at this step
  collect_fp8_tensor_stats: # TE FP8 tensor logging, only enabled when this field is provided in config
    enabled: True
    type: [underflows, overflows]
    tensors: [activation, gradient, weight]
    freq: 100 # every 100 training steps
    start_step: 1000 # start logging at this step
    end_step: 5000 # end logging at this step
```

### Default run
- This is followed when a config is not specified or a layer is not selected in the config.
- For FP8 training, all layers will use delayed scaling and all GEMMs will be in FP8.
- All logging disabled.
- For higher precision training - all GEMMs higher precision.

## Using from Megatron-LM

- Provide the path to a config using the `NV_DEBUG_TOOL_CONFIG` environment variable. Example are provided in `examples_configs/`. If config is not provided, it will run with default setting (see Default Run).
```bash
export NV_DEBUG_TOOL_CONFIG="/path/to/config.yaml"
```
- Clone this [fork](https://gitlab-master.nvidia.com/shreyasm/megatron-lm-fork-2) and checkout `shreyasm/debug_tool` branch.
- Use the `NV_DEBUG_TOOL_ENABLED` environment variable to toggle debug mode in Megatron-LM.
```sh
export NV_DEBUG_TOOL_ENABLED=1
```
- Checkout this [branch](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/merge_requests/156) in TransformerEngine repo.
- Inside container, install the debug version of TE
```sh
cd /opt/TransformerEngine && pip install .
```
- Run megatron-lm script after setting environment variables.

### Performance Specs
| Model Config | TE debug (no config)    | TE debug (stat collection) |
|--------------|-------------------------|----------------------------|
| 8B, 32 GPUs, TP4 | +2.67% from main TE | +3.21% from main TE        |

Debug config used for (2):
```yaml
stat_collection:
  enabled: True
  layers:
    layer_name_regex_pattern: .*(fc1|fc2)
  collect_tensor_stats:
    enabled: True
    type: [l1_norm, l2_norm]
    tensors: [activation]
    freq: 100
  transformer_engine:
    collect_tensor_stats:
      enabled: True
      type: [cur_amax, dynamic_range]
      tensors: [activation, gradient]
      freq: 100
    collect_fp8_tensor_stats:
      enabled: True
      type: [overflows, underflows]
      tensors: [activation, gradient]
      freq: 100
```