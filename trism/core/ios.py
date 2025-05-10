import numpy as np
from trism.core import ProtoAsync, dtypes


class IOs:
  """
  This class is used to capsule inference input and output.
  """
  @property
  def name(self) -> str:
    return self._name
  
  @property
  def shape(self) -> tuple:
    return self._shape
  
  @property
  def dtype(self) -> np.dtype:
    return self._dtype
  
  def __init__(self, name: str, shape: tuple, dtype: str):
    self._name = name
    self._shape = shape
    self._dtype = dtypes.numpy(dtype)

  def input(self, proto: ProtoAsync, data: np.array) -> ProtoAsync.InferInput:
    return proto.InferInput(self.name, self.shape, self.dtype).set_data_from_numpy(data)
  
  def output(self, proto: ProtoAsync) -> ProtoAsync.InferRequestedOutput:
    return proto.InferRequestedOutput(self.name)