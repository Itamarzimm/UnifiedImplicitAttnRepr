<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# LLM inference optimization

Large language models (LLMs) have pushed text generation applications, such as chat and code completion models, to the next level by producing text that displays a high level of understanding and fluency. But what makes LLMs so powerful - namely their size - also presents challenges for inference.

Basic inference is slow because LLMs have to be called repeatedly to generate the next token. The input sequence increases as generation progresses, which takes longer and longer for the LLM to process. LLMs also have billions of parameters, making it a challenge to store and handle all those weights in memory.

This guide will show you how to use the optimization techniques available in Transformers to accelerate LLM inference.

> [!TIP]
> Hugging Face also provides [Text Generation Inference (TGI)](https://hf.co/docs/text-generation-inference), a library dedicated to deploying and serving highly optimized LLMs for inference. It includes more optimization features not included in Transformers, such as continuous batching for increasing throughput and tensor parallelism for multi-GPU inference.

## Static kv-cache and torch.compile

During decoding, a LLM computes the key-value (kv) values for each input token and since it is autoregressive, it computes the same kv values each time because the generated output becomes part of the input now. This is not very efficient because you're recomputing the same kv values each time.

To optimize this, you can use a kv-cache to store the past keys and values instead of recomputing them each time. However, since the kv-cache grows with each generation step and is dynamic, it prevents you from taking advantage of [torch.compile](./perf_torch_compile), a powerful optimization tool that fuses PyTorch code into fast and optimized kernels.

The *static kv-cache* solves this issue by pre-allocating the kv-cache size to a maximum value which allows you to combine it with torch.compile for up to a 4x speed up.

> [!WARNING]
> Currently, only [Command R](./model_doc/cohere), [Gemma](./model_doc/gemma) and [Llama](./model_doc/llama2) models support static kv-cache and torch.compile.

For this example, let's load the [Gemma](https://hf.co/google/gemma-2b) model.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", device_map="auto"
)
```

There are two ways you can configure the model to use a static kv-cache. For a 7B model on an A100, both methods get a 4x speed up in the forward pass. Your speed up may vary depending on the model size (larger models have a smaller speed up) and hardware. If you're using the [`~GenerationMixin.generate`] method, the speed up is ~3x. The forward pass (which still gets 4x speed up) is only a part of the whole [`~GenerationMixin.generate`] code.

<hfoptions id="static-kv">
<hfoption id="generation_config">

Access the model's `generation_config` attribute and set the `cache_implementation` to "static".

```py
model.generation_config.cache_implementation = "static"
```

Call torch.compile on the model to compile the forward pass with the static kv-cache.

```py
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = compiled_model.generate(**input_ids)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

Under the hood, `generate` will attempt to reuse the same cache object, removing the need for re-compilation at each call. However, if the batch size or the maximum output length increase between calls, the cache will have to be reinitialized, triggering a new compilation.

</hfoption>
<hfoption id="Static Cache">

A [`StaticCache`] object can be passed to the model's forward pass under the `past_key_values` argument, enabling the use of this object as a static kv-cache. Using this strategy, you can write your own function to decode the next token given the current token and position and cache position of previously generated tokens. You can also pass the [`StaticCache`] object to [`~GenerationMixin.generate`] and use it across calls, like you would do with a dynamic cache.

```py
from transformers import LlamaTokenizer, LlamaForCausalLM, StaticCache, logging
from transformers.testing_utils import CaptureLogger
import torch

prompts = [
    "Simply put, the theory of relativity states that ",
    "My favorite all time favorite condiment is ketchup.",
]

NUM_TOKENS_TO_GENERATE = 40
torch_device = "cuda"

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="sequential")
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token
```

There are a few important things you must do to enable static kv-cache and torch.compile with the `StaticCache` method:

1. Initialize the [`StaticCache`] instance before using the model for inference. There you can configure parameters like the maximum batch size and sequence length.

2. Call torch.compile on the model to compile the forward pass with the static kv-cache.

3. Set `enable_math=True` in the [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) context manager to enable the native PyTorch C++ implementation of scaled dot product attention to speed up inference even more.

```py
batch_size, seq_length = inputs["input_ids"].shape
with torch.no_grad():
    past_key_values = StaticCache(
        config=model.config, max_batch_size=2, max_cache_len=4096, device=torch_device, dtype=model.dtype
    )
    cache_position = torch.arange(seq_length, device=torch_device)
    generated_ids = torch.zeros(
        batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=torch.int, device=torch_device
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch_device).to(torch.int)

    logits = model(
        **inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True
    )[0]
    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids[:, seq_length] = next_token[:, 0]

    decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
    cache_position = torch.tensor([seq_length + 1], device=torch_device)
    for _ in range(1, NUM_TOKENS_TO_GENERATE):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_tokens(model, next_token.clone(), None, cache_position, past_key_values)
            generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
text
['Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the same for all observers, and 3) the laws of physics are the same for all observers.',
 'My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p']
```

> [!TIP]
> If you want to reuse the [`StaticCache`] object on a new prompt, be sure to reset its contents with the `.reset()` method

</hfoption>
</hfoptions>

## Speculative decoding

> [!TIP]
> For a more in-depth explanation, take a look at the [Assisted Generation: a new direction toward low-latency text generation](https://hf.co/blog/assisted-generation) blog post!

Another issue with autoregression is that for each input token you need to load the model weights each time during the forward pass. This is slow and cumbersome for LLMs which have billions of parameters. Speculative decoding alleviates this slowdown by using a second smaller and faster assistant model to generate candidate tokens that are verified by the larger LLM in a single forward pass. If the verified tokens are correct, the LLM essentially gets them for "free" without having to generate them itself. There is no degradation in accuracy because the verification forward pass ensures the same outputs are generated as if the LLM had generated them on its own.

To get the largest speed up, the assistant model should be a lot smaller than the LLM so that it can generate tokens quickly. The assistant and LLM model must also share the same tokenizer to avoid re-encoding and decoding tokens.

> [!WARNING]
> Speculative decoding is only supported for the greedy search and sampling decoding strategies, and it also doesn't support batched inputs.

Enable speculative decoding by loading an assistant model and passing it to the [`~GenerationMixin.generate`] method.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Einstein's theory of relativity states that the speed of light is constant.    "]
```

</hfoption>
<hfoption id="sampling">

For speculative sampling decoding, add the `do_sample` and `temperature` parameters to the [`~GenerationMixin.generate`] method in addition to the assistant model.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
["Einstein's theory of relativity states that motion in the universe is not a straight line.\n"]
```

</hfoption>
</hfoptions>

### Prompt lookup decoding

Prompt lookup decoding is a variant of speculative decoding that is also compatible with greedy search and sampling. Prompt lookup works especially well for input-grounded tasks - such as summarization - where there is often overlapping words between the prompt and output. These overlapping n-grams are used as the LLM candidate tokens.

To enable prompt lookup decoding, specify the number of tokens that should be overlapping in the `prompt_lookup_num_tokens` parameter. Then you can pass this parameter to the [`~GenerationMixin.generate`] method.

<hfoptions id="pld">
<hfoption id="greedy decoding">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The second law of thermodynamics states that entropy increases with temperature.      ']
```

</hfoption>
<hfoption id="sampling">

For prompt lookup decoding with sampling, add the `do_sample` and `temperature` parameters to the [`~GenerationMixin.generate`] method.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3, do_sample=True, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
["The second law of thermodynamics states that energy cannot be created nor destroyed. It's not a"]
```

</hfoption>
</hfoptions>

## Attention optimizations

A known issue with transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this, try FlashAttention2 or PyTorch's scaled dot product attention (SDPA), which are more memory efficient attention implementations and can accelerate inference.

### FlashAttention-2

FlashAttention and [FlashAttention-2](./perf_infer_gpu_one#flashattention-2) break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to GPU memory to speed up inference. FlashAttention-2 improves on the original FlashAttention algorithm by also parallelizing over sequence length dimension and better partitioning work on the hardware to reduce synchronization and communication overhead.

To use FlashAttention-2, set `attn_implementation="flash_attention_2"` in the [`~PreTrainedModel.from_pretrained`] method.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### PyTorch scaled dot product attention

Scaled dot product attention (SDPA) is automatically enabled in PyTorch 2.0 and it supports FlashAttention, xFormers, and PyTorch's C++ implementation. SDPA chooses the most performant attention algorithm if you're using a CUDA backend. For other backends, SDPA defaults to the PyTorch C++ implementation.

> [!TIP]
> SDPA supports FlashAttention-2 as long as you have the latest PyTorch version installed.

Use the [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) context manager to explicitly enable or disable any of the three attention algorithms. For example, set `enable_flash=True` to enable FlashAttention.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.bfloat16,
)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)
```

## Quantization

Quantization reduces the size of the LLM weights by storing them in a lower precision. This translates to lower memory usage and makes loading LLMs for inference more accessible if you're constrained by your GPUs memory. If you aren't limited by your GPU, you don't necessarily need to quantize your model because it can incur a small latency cost (except for AWQ and fused AWQ modules) due to the extra step required to quantize and dequantize the weights.

> [!TIP]
> There are many quantization libraries (see the [Quantization](./quantization) guide for more details) available, such as Quanto, AQLM, AWQ, and AutoGPTQ. Feel free to try them out and see which one works best for your use case. We also recommend reading the [Overview of natively supported quantization schemes in 🤗 Transformers](https://hf.co/blog/overview-quantization-transformers) blog post which compares AutoGPTQ and bitsandbytes.

Use the Model Memory Calculator below to estimate and compare how much memory is required to load a model. For example, try estimating how much memory it costs to load [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).

<iframe
	src="https://hf-accelerate-model-memory-usage.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

To load Mistral-7B-v0.1 in half-precision, set the `torch_dtype` parameter in the [`~transformers.AutoModelForCausalLM.from_pretrained`] method to `torch.bfloat16`. This requires 13.74GB of memory.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto",
)
```

To load a quantized model (8-bit or 4-bit) for inference, try [bitsandbytes](https://hf.co/docs/bitsandbytes) and set the `load_in_4bit` or `load_in_8bit` parameters to `True`. Loading the model in 8-bits only requires 6.87 GB of memory.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config, device_map="auto"
)
```
