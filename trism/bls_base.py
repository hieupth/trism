import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
from typing import List, Dict, Any
import numpy as np
import os
from abc import ABC, abstractmethod

class BaseTritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
        """
        Initialize tokenizer from shared model path
        """        
        self.model_name = args["model_name"]
        self.model_name_triton = ".".join(self.model_name.split(".")[:-1])
        self.model_version = args["model_version"]
        self.model_repository = args["model_repository"]
        self.model_path = os.path.join(self.model_repository, str(self.model_version))
        self.tokenizer_path = self.model_path.replace(self.model_name, self.model_name_triton)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def decode_input_text(self, request: pb_utils.InferenceRequest) -> List[str]:
        """
        Decode input tensor of bytes into list of strings

        Args:
            request (pb_utils.InferenceRequest): Triton Inference Request

        Returns:
            A list of decoded strings
        """
        text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
        texts = text_tensor.as_numpy() # [B, 1]
        texts = [text.decode("utf-8") for text in texts]
        return texts
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Tokenize a list of input texts

        Args:
            texts: List of input texts to be tokenized

        Returns:
            A dictionary of containing tokenized inputs: input_ids, attention_mask, token_type_ids
        """
        encoded_texts = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        return encoded_texts
    
    def build_inference_request(self, encoded_texts: Dict[str, np.ndarray]) -> pb_utils.InferenceRequest:
        """
        Build a Triton Inference Request from tokenized inputs

        Args: 
            encoded_texts: Tokenized inputs from tokenizer
        
        Returns:
            A Triton Inference Request
        """
        inputs = []
        for name in self.tokenizer.model_input_names:
            inputs.append(pb_utils.Tensor(name, encoded_texts[name].astype(np.int64)))

        return pb_utils.InferenceRequest(
            model_name=self.model_name_triton,
            requested_output_names=["logits"],
            inputs=inputs,
        )
    
    @abstractmethod
    def execute(self, requests):
        """
        Subclasses must implement this method to handle inference requests.
        """
        pass

"""
Example Usage:

This shows how to subclass BaseTritonPythonModel to implement a working Triton Python backend.

```python
class TritonPythonModel(BaseTritonPythonModel):
    def initialize(self, args: Dict[str, Any]) -> None:
        super().initialize(args)

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            texts = self.decode_input_texts(request)
            encoded = self.tokenize_texts(texts)
            inference_request = self.build_inference_request(encoded)

            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise RuntimeError(inference_response.error().message())

            output_tensor = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state")
            response_tensor = pb_utils.Tensor("logits", output_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[response_tensor]))

        return responses
"""