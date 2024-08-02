import torch

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Union

try:
    import debug_tool.api as debug_api
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("[DEBUG-ERROR]: Could not locate debug_tool package. Make sure it is installed correctly.")

@dataclass
class FP8GemmState:
    FPROP: bool = None
    DGRAD: bool = None
    WGRAD: bool = None

@dataclass
class DelayedScalingState:
    FPROP: bool = None
    DGRAD: bool = None
    WGRAD: bool = None


class DebugLayerState:
    """
    A class to manage the state of debug layers.
    """
    layer_count = 1
    layers_initialized = {}
    layers_weight_current_scale_inv: Dict[str, Union[torch.Tensor, None]] = {}

    @classmethod
    def reset(cls):
        cls.layers_initialized.clear()
    
    @classmethod
    def initialize_state(cls, name, fp8_enabled: bool):
        if name not in cls.layers_initialized:
            delayed_scaling_state = DelayedScalingState(
                FPROP=debug_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="fprop", fp8_enabled=fp8_enabled)["ret"],
                DGRAD=debug_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="dgrad", fp8_enabled=fp8_enabled)["ret"],
                WGRAD=debug_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="wgrad", fp8_enabled=fp8_enabled)["ret"]
            )

            fp8_gemm_state = FP8GemmState(
                FPROP=debug_api.transformer_engine.is_fp8_gemm_enabled(name, gemm="fprop", fp8_enabled=fp8_enabled)["ret"],
                DGRAD=debug_api.transformer_engine.is_fp8_gemm_enabled(name, gemm="dgrad", fp8_enabled=fp8_enabled)["ret"],
                WGRAD=debug_api.transformer_engine.is_fp8_gemm_enabled(name, gemm="wgrad", fp8_enabled=fp8_enabled)["ret"]
            )

            cls.layers_initialized[name] = namedtuple('LayerState', ['DelayedScaling', 'FP8Gemm'])(delayed_scaling_state, fp8_gemm_state)
    
    @classmethod
    def get(cls, name: str):
        return cls.layers_initialized[name]
    
    @classmethod
    def get_layer_count(cls):
        """
        Layer counter is used when layer names are not provided to modules by user.
        """
        lc = cls.layer_count
        cls.layer_count += 1
        return lc

    @classmethod
    def set_current_scale_inv(cls, name: str, scale_inv: torch.Tensor):
        cls.layers_weight_current_scale_inv[name] = scale_inv
    
    @classmethod
    def get_current_scale_inv(cls, name: str) -> torch.Tensor:
        return cls.layers_weight_current_scale_inv[name]