import numpy as np
from functools import lru_cache


_TRITON_NUMPY: dict[str, np.dtype] = {
  "BOOL":    bool,
  "UINT8":   np.uint8,
  "UINT16":  np.uint16,
  "UINT32":  np.uint32,
  "UINT64":  np.uint64,
  "INT8":    np.int8,
  "INT16":   np.int16,
  "INT32":   np.int32,
  "INT64":   np.int64,
  "FP16":    np.float16,
  "FP32":    np.float32,
  "FP64":    np.float64,
  "STRING":  np.object_,
  "BYTES":   np.bytes_,
}

_NUMPY_TRITON: dict[np.dtype, str] = {v: k for k, v in _TRITON_NUMPY.items()}


@lru_cache(maxsize=None)
def numpy(triton: str) -> np.dtype:
  try:
    return _TRITON_NUMPY[triton.upper().removeprefix('TYPE_')]
  except KeyError:
    raise TypeError(f"Cannot convert Triton type {triton} to a NumPy dtype")
  
@lru_cache(maxsize=None)
def triton(numpy: type | np.dtype, prefix: bool = False) -> str:
  try:
    triton_type = _NUMPY_TRITON[np.dtype(numpy)]
    return f"TYPE_{triton_type}" if prefix else triton_type
  except KeyError:
    raise TypeError(f"Cannot convert NumPy type {numpy} to a Triton type")