import json
import numpy as np
import asyncio
from typing import Generator
from trism import client 
import requests
import time

class TritonModel:
    @property
    def model(self) -> str:
        return self._model

    @property
    def version(self) -> str:
        return self._version

    @property
    def url(self) -> str:
        return self._url

    @property
    def grpc(self) -> bool:
        return self._grpc

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, model: str, version: int, url: str, grpc: bool = True) -> None:
        self._url = url
        self._grpc = grpc
        self._model = model
        self._version = str(version) if version > 0 else ""
        self._protoclient = client.protoclient(self.grpc)
        self._serverclient = client.serverclient(self.url, self.grpc)
        self._inputs, self._outputs = client.inout(self._serverclient, self.model, self.version)

    def run(self, data: list[np.array]):
        inputs = [self.inputs[i].make_input(self._protoclient, data[i]) for i in range(len(self.inputs))]
        outputs = [output.make_output(self._protoclient) for output in self.outputs]
        results = self._serverclient.infer(self.model, inputs, self.version, outputs)
        return {output.name: results.as_numpy(output.name) for output in self.outputs}

class TritonLMModel:
    def __init__(self, model: str, version: int = 1, url: str = 'localhost:8001', stream: bool = True):
        self._url = url
        self._grpc = True
        self._model = model
        self._stream = stream
        self._version = str(version) if version > 0 else ""
        self._protoclient = client.protoclient_async(self._grpc)
        self._serverclient = client.serverclient_async(self._url, self._grpc, async_mode=True)
        self._inputs, self._outputs = None, None

    async def _request_iterator(self, prompt: str, sampling_parameters: dict):
        inputs = []
        for inp in self._inputs:
            if inp.name == "text_input":
                data = np.array([prompt.encode("utf-8")], dtype=np.object_)
            elif inp.name == "stream":
                data = np.array([self._stream], dtype=bool)
            elif inp.name == "sampling_parameters":
                data = np.array([json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_)
            elif inp.name == "exclude_input_in_output":
                data = np.array([True], dtype=bool)
            else:
                continue
            inputs.append(inp.make_input(self._protoclient, data))

        outputs = [out.make_output(self._protoclient) for out in self._outputs]

        yield {
            "model_name": self._model,
            "inputs": inputs,
            "outputs": outputs,
        }

    async def run(self, prompt: str, sampling_parameters: dict, show_thinking: bool = True) -> Generator[str, None, None]:
        if self._inputs is None or self._outputs is None:
            self._inputs, self._outputs = await client.iout_async(self._serverclient, self._model, self._version)

        if show_thinking:
            print("🔄 Begin chain of thought ...", end="", flush=True)
            thinking_started = False
        else:
            print("🔄 Thinking ...", end="", flush=True)
            thinking_started = True

        local_history = []
        buffer = ""
        response_iterator = self._serverclient.stream_infer(
            inputs_iterator=self._request_iterator(prompt=prompt, sampling_parameters=sampling_parameters)
        )
        async for response in response_iterator:
            result, error = response
            if error is not None:
                print(f"[!] Triton returned error: {error}")
                continue
            output = result.as_numpy("text_output")
            for token in output:
                text = token.decode("utf-8")
                local_history.append(text)
                if thinking_started:
                    buffer += text
                    if "</think>" in buffer:
                        yield text
                    continue
                else:
                    yield text
