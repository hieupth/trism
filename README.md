# Python TRISM
**TR**iton **I**nference **S**erver **M**odel is a simple Python package that helps us infer with Triton Inference Server easily.
```bash
pip install trism
# Or
pip install https://github.com/hieupth/trism
```
## How to use
### 1. For Standard Models

```python
from trism import TritonModel

# Create triton model.
model = TritonModel(
  model="my_model",     # Model name.
  version=0,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# Inference.
outputs = model.run(data = [np.array(...)])
```


### 2. For VLM Models (Streaming)


### Configuration for VLM Streaming

To enable VLM streaming, you need to add the following configuration to your `config.pbtxt` file:

```plaintext

parameters [
  {
    key: "stream"
    value: { string_value: "true" }
  }
]

```

### Inferences VLM model

```python
from trism import TritonVLMModel
import asyncio

async def main():
    vlm = TritonLMModel(model="vllm_model", version=1, url="localhost:8001")
    sampling_parameters = {
        "temperature": 0.7,
        "max_tokens": 4096
    }
    async for token in vlm.run("Why is the color of ocean blue?", sampling_parameters=sampling_parameters, show_thinking=True):
        print(token) # Check input
    await vlm._serverclient.close()

asyncio.run(main())

```
## License
[GNU AGPL v3.0](LICENSE).<br>
Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.


