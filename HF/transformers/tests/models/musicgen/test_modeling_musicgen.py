# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch Musicgen model. """
import copy
import inspect
import math
import tempfile
import unittest

import numpy as np
from parameterized import parameterized
from pytest import mark

from transformers import (
    EncodecConfig,
    MusicgenConfig,
    MusicgenDecoderConfig,
    MusicgenProcessor,
    PretrainedConfig,
    T5Config,
)
from transformers.testing_utils import (
    is_torch_available,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MusicgenForCausalLM,
        MusicgenForConditionalGeneration,
        MusicgenModel,
        set_seed,
    )
    from transformers.generation import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
    )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def prepare_musicgen_decoder_inputs_dict(
    config,
    input_ids,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.reshape(-1, config.num_codebooks, input_ids.shape[-1])[:, 0, :]
        attention_mask = attention_mask.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    if encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "head_mask": head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MusicgenDecoderTester:
    def __init__(
        self,
        parent,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])

        config = self.get_config()
        inputs_dict = prepare_musicgen_decoder_inputs_dict(
            config,
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        return config, inputs_dict

    def get_config(self):
        config = MusicgenDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MusicgenDecoderTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenModel, MusicgenForCausalLM) if is_torch_available() else ()
    greedy_sample_model_classes = (
        (MusicgenForCausalLM,) if is_torch_available() else ()
    )  # we don't want to run all the generation tests, only a specific subset
    pipeline_model_mapping = {}
    test_pruning = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = MusicgenDecoderTester(self)
        self.config_tester = ConfigTester(self, config_class=MusicgenDecoderConfig, hidden_size=16)

    def test_config(self):
        self.config_tester.run_common_tests()

    # special case for labels
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_codebooks),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.use_cache = False
        config.return_dict = True
        model = MusicgenForCausalLM(config)

        model.to(torch_device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        model.train()

        # Contrarily to the initial method, we don't unfreeze freezed parameters.
        # Indeed, sinusoidal position embeddings have frozen weights that should stay frozen.

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        inputs = self._prepare_for_class(inputs_dict, MusicgenForCausalLM, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()

        for k, v in model.named_parameters():
            if v.requires_grad:
                self.assertTrue(v.grad is not None, f"{k} in {MusicgenForCausalLM.__name__} has no gradient!")

    # override since we have to compute the input embeddings over codebooks
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]

            embed_tokens = model.get_input_embeddings()

            input_ids = input_ids.reshape(-1, config.num_codebooks, input_ids.shape[-1])

            inputs["inputs_embeds"] = sum(
                [embed_tokens[codebook](input_ids[:, codebook]) for codebook in range(config.num_codebooks)]
            )

            with torch.no_grad():
                model(**inputs)[0]

    # override since we have embeddings / LM heads over multiple codebooks
    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            first_embed = model.get_input_embeddings()[0]
            self.assertIsInstance(first_embed, torch.nn.Embedding)
            lm_heads = model.get_output_embeddings()
            self.assertTrue(lm_heads is None or isinstance(lm_heads[0], torch.nn.Linear))

    @unittest.skip(reason="MusicGen does not use inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    # skip as this model doesn't support all arguments tested
    def test_model_outputs_equivalence(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tie_model_weights(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tied_weights_keys(self):
        pass

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]

        # take max batch_size
        sequence_length = input_ids.shape[-1]
        input_ids = input_ids[: batch_size * config.num_codebooks, :]

        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
        return config, input_ids, attention_mask

    @staticmethod
    def _get_logits_processor_and_warper_kwargs(
        input_length,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    ):
        process_kwargs = {}
        warper_kwargs = {}
        return process_kwargs, warper_kwargs

    def test_greedy_generate_stereo_outputs(self):
        for model_class in self.greedy_sample_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            config.audio_channels = 2
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)

            self.assertNotIn(config.pad_token_id, output_generate)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_inference_equivalence
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, 1:] = 1
                    dummy_attention_mask[:, :1] = 0

                # Ignore copy
                outputs = model(dummy_input, output_hidden_states=True)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                # Ignore copy
                other_inputs = {
                    "output_hidden_states": True,
                }
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask

                outputs = model(dummy_input, **other_inputs)
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[1:], logits[1:], atol=4e-2, rtol=4e-2)

                # check with inference + dropout
                model.train()
                _ = model_fa(dummy_input, **other_inputs)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_inference_equivalence_right_padding
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, :-1] = 1
                    dummy_attention_mask[:, -1:] = 0

                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)

                    outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                else:
                    outputs = model(dummy_input, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)
                # Ignore copy
                other_inputs = {
                    "output_hidden_states": True,
                }
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask

                outputs = model(dummy_input, **other_inputs)
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[:-1], logits[:-1], atol=4e-2, rtol=4e-2)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_generate_left_padding
    def test_flash_attn_2_generate_left_padding(self):
        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
                    torch_device
                )

                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                    dummy_input = dummy_input.to(torch.float16)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # make sure we do left padding
                dummy_attention_mask[:, :-1] = 0
                dummy_attention_mask[:, -1:] = 1

                out = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                out_fa = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                self.assertTrue(torch.allclose(out, out_fa))

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_generate_padding_right
    def test_flash_attn_2_generate_padding_right(self):
        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
                    torch_device
                )

                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                    dummy_input = dummy_input.to(torch.float16)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # make sure we do right padding
                dummy_attention_mask[:, :-1] = 1
                dummy_attention_mask[:, -1:] = 0

                out = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                out_fa = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                self.assertTrue(torch.allclose(out, out_fa))

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_generate_use_cache
    def test_flash_attn_2_generate_use_cache(self):
        max_new_tokens = 30

        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                # Just test that a large cache works as expected
                _ = model.generate(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_eager_matches_sdpa_inference
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        has_sdpa = True
                        break
                if not has_sdpa and model_sdpa.config.model_type != "falcon":
                    raise ValueError("The SDPA model should have SDPA attention layers")

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 8 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for batch_size in [1, 5]:
                            # Ignore copy
                            batch_size_input_ids = self.model_tester.num_codebooks * batch_size
                            dummy_input = inputs_dict[model.main_input_name]

                            if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                dummy_input = dummy_input.to(torch_dtype)

                            # Ignore copy
                            dummy_input = dummy_input[:batch_size_input_ids]
                            # Ignore copy
                            if dummy_input.shape[0] != batch_size_input_ids:
                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    # Ignore copy
                                    extension = torch.rand(
                                        batch_size_input_ids - dummy_input.shape[0],
                                        *dummy_input.shape[1:],
                                        dtype=torch_dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                else:
                                    # Ignore copy
                                    extension = torch.randint(
                                        high=5,
                                        size=(batch_size_input_ids - dummy_input.shape[0], *dummy_input.shape[1:]),
                                        dtype=dummy_input.dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                            if not use_mask:
                                dummy_attention_mask = None
                            else:
                                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                if dummy_attention_mask is None:
                                    if is_encoder_decoder:
                                        seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                    else:
                                        seqlen = dummy_input.shape[-1]
                                    dummy_attention_mask = (
                                        torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                    )

                                dummy_attention_mask = dummy_attention_mask[:batch_size]
                                if dummy_attention_mask.shape[0] != batch_size:
                                    extension = torch.ones(
                                        batch_size - dummy_attention_mask.shape[0],
                                        *dummy_attention_mask.shape[1:],
                                        dtype=dummy_attention_mask.dtype,
                                        device=torch_device,
                                    )
                                    dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                    dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                dummy_attention_mask[:] = 1
                                if padding_side == "left":
                                    dummy_attention_mask[-1, :-1] = 1
                                    dummy_attention_mask[-1, -4:] = 0
                                elif padding_side == "right":
                                    dummy_attention_mask[-1, 1:] = 1
                                    dummy_attention_mask[-1, :3] = 0

                            for enable_kernels in [False, True]:
                                failcase = f"padding_side={padding_side}, use_mask={use_mask}, batch_size={batch_size}, enable_kernels={enable_kernels}"

                                other_inputs = {
                                    "output_hidden_states": True,
                                }

                                # Otherwise fails for e.g. WhisperEncoderModel
                                if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                                    other_inputs["attention_mask"] = dummy_attention_mask

                                # TODO: test gradients as well (& for FA2 as well!)
                                with torch.no_grad():
                                    with torch.backends.cuda.sdp_kernel(
                                        enable_flash=enable_kernels,
                                        enable_math=True,
                                        enable_mem_efficient=enable_kernels,
                                    ):
                                        outputs_eager = model_eager(dummy_input, **other_inputs)
                                        outputs_sdpa = model_sdpa(dummy_input, **other_inputs)

                                logits_eager = (
                                    outputs_eager.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_eager.decoder_hidden_states[-1]
                                )
                                logits_sdpa = (
                                    outputs_sdpa.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_sdpa.decoder_hidden_states[-1]
                                )

                                if torch_device in ["cpu", "cuda"]:
                                    atol = atols[torch_device, enable_kernels, torch_dtype]
                                    rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                else:
                                    atol = 1e-7
                                    rtol = 1e-4

                                # Masked tokens output slightly deviates - we don't mind that.
                                if use_mask:
                                    if padding_side == "left":
                                        sub_sdpa = logits_sdpa[:-1]
                                        sub_eager = logits_eager[:-1]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        sub_sdpa = logits_sdpa[-1, :-4]
                                        sub_eager = logits_eager[-1, :-4]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        # Testing the padding tokens is not really meaningful but anyway
                                        # sub_sdpa = logits_sdpa[-1, -4:]
                                        # sub_eager = logits_eager[-1, -4:]
                                        # if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                        #     fail_cases.append(get_mean_reldiff(failcase, sub_sdpa, sub_eager, 4e-2, 4e-2))
                                    elif padding_side == "right":
                                        sub_sdpa = logits_sdpa[:-1]
                                        sub_eager = logits_eager[:-1]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        sub_sdpa = logits_sdpa[-1, 3:]
                                        sub_eager = logits_eager[-1, 3:]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        # Testing the padding tokens is not really meaningful but anyway
                                        # sub_sdpa = logits_sdpa[-1, :3]
                                        # sub_eager = logits_eager[-1, :3]
                                        # if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                        #     fail_cases.append(get_mean_reldiff(failcase, sub_sdpa, sub_eager, 4e-2, 4e-2))

                                else:
                                    if not torch.allclose(logits_sdpa, logits_eager, atol=atol, rtol=rtol):
                                        fail_cases.append(
                                            get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                        )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    @require_torch_sdpa
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_eager_matches_sdpa_generate
    def test_eager_matches_sdpa_generate(self):
        max_new_tokens = 30

        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))

                model_sdpa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                ).to(torch_device)

                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        has_sdpa = True
                        break
                if not has_sdpa:
                    raise ValueError("The SDPA model should have SDPA attention layers")

                # Just test that a large cache works as expected
                res_eager = model_eager.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=max_new_tokens, do_sample=False
                )

                res_sdpa = model_sdpa.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=max_new_tokens, do_sample=False
                )

                self.assertTrue(torch.allclose(res_eager, res_sdpa))


def prepare_musicgen_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    labels=None,
):
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.reshape(
            -1, config.decoder.num_codebooks, decoder_input_ids.shape[-1]
        )[:, 0, :]
        decoder_attention_mask = decoder_attention_mask.ne(config.decoder.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(
            config.text_encoder.num_hidden_layers, config.text_encoder.num_attention_heads, device=torch_device
        )
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "labels": labels,
    }


class MusicgenTester:
    def __init__(
        self,
        parent,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
        num_filters=4,
        codebook_size=128,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.num_filters = num_filters
        self.codebook_size = codebook_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_musicgen_inputs_dict(config, input_ids, decoder_input_ids=decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        text_encoder_config = T5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.intermediate_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
        )
        audio_encoder_config = EncodecConfig(
            hidden_size=self.vocab_size,
            compress=1,
            num_filters=self.num_filters,
            codebook_size=self.codebook_size,
            codebook_dim=self.vocab_size,
        )
        decoder_config = MusicgenDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
        )
        config = MusicgenConfig.from_sub_models_config(text_encoder_config, audio_encoder_config, decoder_config)
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MusicgenTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenForConditionalGeneration,) if is_torch_available() else ()
    greedy_sample_model_classes = (MusicgenForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-to-audio": MusicgenForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False  # training is not supported yet for MusicGen
    test_headmasking = False
    test_resize_embeddings = False
    # not to test torchscript as the model tester doesn't prepare `input_values` and `padding_mask`
    # (and `torchscript` hates `None` values).
    test_torchscript = False

    def setUp(self):
        self.model_tester = MusicgenTester(self)

    # special case for labels
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_codebooks),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True
            model = model_class(config)

            model.to(torch_device)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            model.train()

            # The audio encoder weights are not used during the forward pass (only during the generate pass)
            # So we need to freeze it to be able to train.
            model.freeze_audio_encoder()

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()

            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.assertTrue(v.grad is not None, f"{k} in {model_class.__name__} has no gradient!")

    def _check_output_with_attentions(self, outputs, config, input_ids, decoder_input_ids):
        text_encoder_config = config.text_encoder
        decoder_config = config.decoder

        encoder_attentions = outputs["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), text_encoder_config.num_hidden_layers)

        self.assertEqual(
            encoder_attentions[0].shape[-3:],
            (text_encoder_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]),
        )

        decoder_attentions = outputs["decoder_attentions"]
        num_decoder_layers = decoder_config.num_hidden_layers
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, input_ids.shape[-1]),
        )

    def check_musicgen_model_output_attentions(
        self,
        model_class,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=True,
                **kwargs,
            )
        self._check_output_with_attentions(outputs, config, input_ids, decoder_input_ids)

    def check_musicgen_model_output_attentions_from_config(
        self,
        model_class,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        # Similar to `check_musicgen_model_output_attentions`, but with `output_attentions` triggered from the
        # config file. Contrarily to most models, changing the model's config won't work -- the defaults are loaded
        # from the inner models' configurations.
        config.output_attentions = True  # model config -> won't work

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        self.assertTrue(
            all(key not in outputs for key in ["encoder_attentions", "decoder_attentions", "cross_attentions"])
        )
        config.text_encoder.output_attentions = True  # inner model config -> will work
        config.audio_encoder.output_attentions = True
        config.decoder.output_attentions = True

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        self._check_output_with_attentions(outputs, config, input_ids, decoder_input_ids)

    # override since changing `output_attentions` from the top-level model config won't work
    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            self.check_musicgen_model_output_attentions(model_class, config, **inputs_dict)
            self.check_musicgen_model_output_attentions_from_config(model_class, config, **inputs_dict)

    # override since we have a specific forward signature for musicgen
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_ids",
                "attention_mask",
                "input_values",
                "padding_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # override since changing `gradient_checkpointing` from the top-level model config won't work
    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.text_encoder.gradient_checkpointing = True
            config.audio_encoder.gradient_checkpointing = True
            config.decoder.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tie_model_weights(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tied_model_weights_key_ignore(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tied_weights_keys(self):
        pass

    # override since changing `output_hidden_states` / `output_attentions` from the top-level model config won't work
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_encoder.output_hidden_states = True
        config.audio_encoder.output_hidden_states = True
        config.decoder.output_hidden_states = True

        config.text_encoder.output_attentions = True
        config.decoder.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        decoder_hidden_states = outputs.decoder_hidden_states[0]
        decoder_hidden_states.retain_grad()

        if self.has_attentions:
            encoder_attentions = outputs.encoder_attentions[0]
            encoder_attentions.retain_grad()

            decoder_attentions = outputs.decoder_attentions[0]
            decoder_attentions.retain_grad()

            cross_attentions = outputs.cross_attentions[0]
            cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(decoder_hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(encoder_attentions.grad)
            self.assertIsNotNone(decoder_attentions.grad)
            self.assertIsNotNone(cross_attentions.grad)

    # override since changing `output_hidden_states` from the top-level model config won't work
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states

            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.seq_length
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            hidden_states = outputs.decoder_hidden_states
            self.assertIsInstance(hidden_states, (list, tuple))
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.text_encoder.output_hidden_states = True
            config.audio_encoder.output_hidden_states = True
            config.decoder.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # override since the conv layers and lstm's in encodec are exceptions
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv"]
                ignore_init = ["lstm"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif not any(x in name for x in ignore_init):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # override since we have embeddings / LM heads over multiple codebooks
    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), torch.nn.Embedding)
            lm_heads = model.get_output_embeddings()
            self.assertTrue(lm_heads is None or isinstance(lm_heads[0], torch.nn.Linear))

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]

        # take max batch_size
        sequence_length = input_ids.shape[-1]
        input_ids = input_ids[:batch_size, :]
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)

        return config, input_ids, attention_mask

    # override since the `input_ids` cannot be used as the `decoder_input_ids` for musicgen (input / outputs are
    # different modalities -> different shapes)
    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **model_kwargs,
        )

        return output_generate

    # override since the `input_ids` cannot be used as the `decoder_input_ids` for musicgen (input / outputs are
    # different modalities -> different shapes)
    def _sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        num_return_sequences,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        torch.manual_seed(0)
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **model_kwargs,
        )

        return output_generate

    @staticmethod
    def _get_logits_processor_and_warper_kwargs(
        input_length,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    ):
        process_kwargs = {}
        warper_kwargs = {}
        return process_kwargs, warper_kwargs

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.greedy_sample_model_classes:
            # disable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)

            self.assertNotIn(config.pad_token_id, output_generate)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.greedy_sample_model_classes:
            # enable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)

    def test_sample_generate(self):
        for model_class in self.greedy_sample_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()

            # check `generate()` and `sample()` are equal
            output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                num_return_sequences=1,
            )
            self.assertIsInstance(output_generate, torch.Tensor)

    def test_sample_generate_dict_output(self):
        for model_class in self.greedy_sample_model_classes:
            # disable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()

            output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                num_return_sequences=3,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)

    def test_generate_without_input_ids(self):
        config, _, _ = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            return

        for model_class in self.greedy_sample_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(
                do_sample=False, max_new_tokens=self.max_new_tokens, remove_invalid_values=True
            )
            self.assertIsNotNone(output_ids_generate)

    @require_torch_fp16
    @require_torch_accelerator  # not all operations are supported in fp16 on CPU
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()

        for model_class in self.greedy_sample_model_classes:
            model = model_class(config).eval().to(torch_device)
            model.half()
            # greedy
            model.generate(input_dict["input_ids"], attention_mask=input_dict["attention_mask"], max_new_tokens=10)
            # sampling
            model.generate(
                input_dict["input_ids"], attention_mask=input_dict["attention_mask"], do_sample=True, max_new_tokens=10
            )

    def test_greedy_generate_stereo_outputs(self):
        for model_class in self.greedy_sample_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            config.audio_channels = 2

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)

            self.assertNotIn(config.pad_token_id, output_generate)

    @unittest.skip("MusicgenModel is actually not the base of MusicgenForCausalLM as the latter is a composit model")
    def test_save_load_fast_init_from_base(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_inference_equivalence
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, 1:] = 1
                    dummy_attention_mask[:, :1] = 0

                # Ignore copy
                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)
                # Ignore copy
                outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)
                # Ignore copy
                other_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": dummy_attention_mask,
                    "output_hidden_states": True,
                }
                # Ignore copy
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask
                # Ignore copy
                outputs = model(dummy_input, **other_inputs)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[1:], logits[1:], atol=4e-2, rtol=4e-2)

                # check with inference + dropout
                model.train()
                _ = model_fa(dummy_input, **other_inputs)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_inference_equivalence_right_padding
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, :-1] = 1
                    dummy_attention_mask[:, -1:] = 0

                # Ignore copy
                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)
                # Ignore copy
                outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                # Ignore copy
                other_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": dummy_attention_mask,
                    "output_hidden_states": True,
                }
                # Ignore copy
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask
                # Ignore copy
                outputs = model(dummy_input, **other_inputs)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[:-1], logits[:-1], atol=4e-2, rtol=4e-2)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_generate_left_padding
    def test_flash_attn_2_generate_left_padding(self):
        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
                    torch_device
                )

                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                    dummy_input = dummy_input.to(torch.float16)

                dummy_attention_mask = inputs_dict.get("attention_mask")
                if dummy_attention_mask is None:
                    dummy_attention_mask = torch.ones_like(dummy_input)

                # make sure we do left padding
                dummy_attention_mask[:, :-1] = 0
                dummy_attention_mask[:, -1:] = 1

                out = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                out_fa = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                self.assertTrue(torch.allclose(out, out_fa))

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_generate_padding_right
    def test_flash_attn_2_generate_padding_right(self):
        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
                    torch_device
                )

                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                    dummy_input = dummy_input.to(torch.float16)

                dummy_attention_mask = inputs_dict.get("attention_mask")
                if dummy_attention_mask is None:
                    dummy_attention_mask = torch.ones_like(dummy_input)
                # make sure we do right padding
                dummy_attention_mask[:, :-1] = 1
                dummy_attention_mask[:, -1:] = 0

                out = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                out_fa = model.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=8, do_sample=False
                )

                self.assertTrue(torch.allclose(out, out_fa))

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_generate_use_cache
    def test_flash_attn_2_generate_use_cache(self):
        max_new_tokens = 30

        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                # Just test that a large cache works as expected
                _ = model.generate(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_eager_matches_sdpa_inference
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        has_sdpa = True
                        break
                if not has_sdpa and model_sdpa.config.model_type != "falcon":
                    raise ValueError("The SDPA model should have SDPA attention layers")

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 8 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for batch_size in [1, 5]:
                            dummy_input = inputs_dict[model.main_input_name]

                            if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                dummy_input = dummy_input.to(torch_dtype)

                            dummy_input = dummy_input[:batch_size]
                            if dummy_input.shape[0] != batch_size:
                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    extension = torch.rand(
                                        batch_size - dummy_input.shape[0],
                                        *dummy_input.shape[1:],
                                        dtype=torch_dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                else:
                                    extension = torch.randint(
                                        high=5,
                                        size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                        dtype=dummy_input.dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                            if not use_mask:
                                dummy_attention_mask = None
                            else:
                                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                if dummy_attention_mask is None:
                                    # Ignore copy
                                    seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                    # Ignore copy
                                    dummy_attention_mask = (
                                        torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                    )

                                dummy_attention_mask = dummy_attention_mask[:batch_size]
                                if dummy_attention_mask.shape[0] != batch_size:
                                    extension = torch.ones(
                                        batch_size - dummy_attention_mask.shape[0],
                                        *dummy_attention_mask.shape[1:],
                                        dtype=dummy_attention_mask.dtype,
                                        device=torch_device,
                                    )
                                    dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                    dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                dummy_attention_mask[:] = 1
                                if padding_side == "left":
                                    dummy_attention_mask[-1, :-1] = 1
                                    dummy_attention_mask[-1, -4:] = 0
                                elif padding_side == "right":
                                    dummy_attention_mask[-1, 1:] = 1
                                    dummy_attention_mask[-1, :3] = 0

                            for enable_kernels in [False, True]:
                                failcase = f"padding_side={padding_side}, use_mask={use_mask}, batch_size={batch_size}, enable_kernels={enable_kernels}"
                                # Ignore copy
                                batch_size_input_ids = self.model_tester.num_codebooks * batch_size
                                # Ignore copy
                                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[
                                    :batch_size_input_ids
                                ]
                                # Ignore copy
                                if decoder_input_ids.shape[0] != batch_size_input_ids:
                                    # Ignore copy
                                    extension = torch.ones(
                                        batch_size_input_ids - decoder_input_ids.shape[0],
                                        *decoder_input_ids.shape[1:],
                                        dtype=decoder_input_ids.dtype,
                                        device=torch_device,
                                    )
                                    decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                                    decoder_input_ids = decoder_input_ids.to(torch_device)

                                # TODO: never an `attention_mask` arg here?
                                # Ignore copy
                                other_inputs = {
                                    "decoder_input_ids": decoder_input_ids,
                                    "decoder_attention_mask": dummy_attention_mask,
                                    "output_hidden_states": True,
                                }

                                # TODO: test gradients as well (& for FA2 as well!)
                                # Ignore copy
                                with torch.no_grad():
                                    with torch.backends.cuda.sdp_kernel(
                                        enable_flash=enable_kernels,
                                        enable_math=True,
                                        enable_mem_efficient=enable_kernels,
                                    ):
                                        outputs_eager = model_eager(dummy_input, **other_inputs)
                                        outputs_sdpa = model_sdpa(dummy_input, **other_inputs)

                                logits_eager = (
                                    outputs_eager.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_eager.decoder_hidden_states[-1]
                                )
                                logits_sdpa = (
                                    outputs_sdpa.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_sdpa.decoder_hidden_states[-1]
                                )

                                if torch_device in ["cpu", "cuda"]:
                                    atol = atols[torch_device, enable_kernels, torch_dtype]
                                    rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                else:
                                    atol = 1e-7
                                    rtol = 1e-4

                                # Masked tokens output slightly deviates - we don't mind that.
                                if use_mask:
                                    if padding_side == "left":
                                        sub_sdpa = logits_sdpa[:-1]
                                        sub_eager = logits_eager[:-1]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        sub_sdpa = logits_sdpa[-1, :-4]
                                        sub_eager = logits_eager[-1, :-4]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        # Testing the padding tokens is not really meaningful but anyway
                                        # sub_sdpa = logits_sdpa[-1, -4:]
                                        # sub_eager = logits_eager[-1, -4:]
                                        # if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                        #     fail_cases.append(get_mean_reldiff(failcase, sub_sdpa, sub_eager, 4e-2, 4e-2))
                                    elif padding_side == "right":
                                        sub_sdpa = logits_sdpa[:-1]
                                        sub_eager = logits_eager[:-1]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        sub_sdpa = logits_sdpa[-1, 3:]
                                        sub_eager = logits_eager[-1, 3:]
                                        if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                            )

                                        # Testing the padding tokens is not really meaningful but anyway
                                        # sub_sdpa = logits_sdpa[-1, :3]
                                        # sub_eager = logits_eager[-1, :3]
                                        # if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                        #     fail_cases.append(get_mean_reldiff(failcase, sub_sdpa, sub_eager, 4e-2, 4e-2))

                                else:
                                    if not torch.allclose(logits_sdpa, logits_eager, atol=atol, rtol=rtol):
                                        fail_cases.append(
                                            get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                        )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    @require_torch_sdpa
    @slow
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_eager_matches_sdpa_generate
    def test_eager_matches_sdpa_generate(self):
        max_new_tokens = 30

        # Ignore copy
        for model_class in self.greedy_sample_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))

                model_sdpa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                ).to(torch_device)

                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        has_sdpa = True
                        break
                if not has_sdpa:
                    raise ValueError("The SDPA model should have SDPA attention layers")

                # Just test that a large cache works as expected
                res_eager = model_eager.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=max_new_tokens, do_sample=False
                )

                res_sdpa = model_sdpa.generate(
                    dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=max_new_tokens, do_sample=False
                )

                self.assertTrue(torch.allclose(res_eager, res_sdpa))

    def test_requires_grad_with_frozen_encoders(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.freeze_audio_encoder()

            audio_encoder_grads = [param.requires_grad for param in model.audio_encoder.parameters()]
            text_encoder_grads = [param.requires_grad for param in model.text_encoder.parameters()]

            self.assertFalse(all(audio_encoder_grads))
            self.assertTrue(all(text_encoder_grads))

            model = model_class(config)
            model.freeze_text_encoder()

            audio_encoder_grads = [param.requires_grad for param in model.audio_encoder.parameters()]
            text_encoder_grads = [param.requires_grad for param in model.text_encoder.parameters()]

            self.assertTrue(all(audio_encoder_grads))
            self.assertFalse(all(text_encoder_grads))


def get_bip_bip(bip_duration=0.125, duration=0.5, sample_rate=32000):
    """Produces a series of 'bip bip' sounds at a given frequency."""
    timesteps = np.arange(int(duration * sample_rate)) / sample_rate
    wav = np.cos(2 * math.pi * 440 * timesteps)
    time_period = (timesteps % (2 * bip_duration)) / (2 * bip_duration)
    envelope = time_period >= 0.5
    return wav * envelope


def place_dict_on_device(dict_to_place, device):
    for key in dict_to_place:
        if dict_to_place[key] is not None and isinstance(dict_to_place[key], torch.Tensor):
            dict_to_place[key] = dict_to_place[key].to(device)
    return dict_to_place


@require_torch
class MusicgenIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(torch_device)

    @cached_property
    def processor(self):
        return MusicgenProcessor.from_pretrained("facebook/musicgen-small")

    @slow
    def test_logits_text_prompt(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        # prepare the decoder inputs
        pad_token_id = model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones((input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long).to(torch_device)
            * pad_token_id
        )

        with torch.no_grad():
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                -0.9708, -3.0149, -4.6415, -1.4754, -0.2786, -2.3523, -2.6049, -6.7467,
                -1.0206, -3.2984, -3.3968, -1.5108, -1.5786, -3.1493, -1.1503, -0.0545,
            ]
        )
        # fmt: on

        self.assertTrue(logits.shape == (*decoder_input_ids.shape, model.decoder.config.vocab_size))
        self.assertTrue(torch.allclose(logits[0, 0, :16].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_logits_text_audio_prompt(self):
        model = self.model
        processor = self.processor

        audio = [get_bip_bip(duration=0.5), get_bip_bip(duration=1.0)]
        text = ["80s music", "Club techno"]

        inputs = processor(audio=audio, text=text, padding=True, return_tensors="pt")

        # prepare the text encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        # prepare the audio encoder inputs
        input_values = inputs.input_values.to(torch_device)
        padding_mask = inputs.padding_mask.to(torch_device)

        with torch.no_grad():
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                input_values=input_values,
                padding_mask=padding_mask,
            ).logits

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                0.1841, -2.9324, -0.7898, 0.1857, 0.4971, -2.8685, -1.6525, -1.6541,
                2.7757, -2.5942, -3.0959, -1.0120, -1.0147, -0.4605, -0.8885, 0.6820,
            ]
        )
        # fmt: on

        self.assertTrue(logits.shape == (8, 50, 2048))
        self.assertTrue(torch.allclose(logits[0, -1, :16].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_generate_unconditional_greedy(self):
        model = self.model

        # only generate 1 sample with greedy - since it's deterministic all elements of the batch will be the same
        unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        unconditional_inputs = place_dict_on_device(unconditional_inputs, device=torch_device)

        output_values = model.generate(**unconditional_inputs, do_sample=False, max_new_tokens=5)

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                0.0056, 0.0064, 0.0063, 0.0054, 0.0042, 0.0033, 0.0024, 0.0015,
                0.0015, 0.0010, 0.0004, -0.0012, -0.0036, -0.0055, -0.0067, -0.0071,
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (1, 1, 3200))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_unconditional_sampling(self):
        model = self.model

        # for stochastic sampling we can generate multiple outputs
        unconditional_inputs = model.get_unconditional_inputs(num_samples=2)
        unconditional_inputs = place_dict_on_device(unconditional_inputs, device=torch_device)

        set_seed(0)
        output_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=10)

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -0.0099, -0.0140, 0.0079, 0.0080, -0.0046,  0.0065, -0.0068, -0.0185,
                 0.0105,  0.0059, 0.0329, 0.0249, -0.0204, -0.0341, -0.0465,  0.0053,
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_prompt_greedy(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        output_values = model.generate(
            input_ids, attention_mask=attention_mask, do_sample=False, guidance_scale=None, max_new_tokens=10
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -1.1998e-04, -2.2302e-04, 4.6296e-04, 1.0524e-03, 2.4827e-04,
                -4.0288e-05, -1.2468e-04, 4.9846e-05, 7.1485e-04, 4.4197e-04,
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :10].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_prompt_greedy_with_classifier_free_guidance(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        output_values = model.generate(
            input_ids, attention_mask=attention_mask, do_sample=False, guidance_scale=3, max_new_tokens=10
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                0.0283, 0.0246, 0.0650, 0.0640, 0.0599, 0.0711, 0.0420, 0.0112,
                0.0511, 0.0746, 0.1363, 0.1213, 0.0185, -0.0578, -0.0908, 0.0443,
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_prompt_sampling(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        set_seed(0)
        output_values = model.generate(
            input_ids, attention_mask=attention_mask, do_sample=True, guidance_scale=None, max_new_tokens=10
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -0.0111, -0.0154, 0.0047, 0.0058, -0.0068,  0.0012, -0.0109, -0.0229,
                 0.0010, -0.0038, 0.0167, 0.0042, -0.0421, -0.0610, -0.0764, -0.0326,
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_audio_prompt(self):
        model = self.model
        processor = self.processor

        audio = [get_bip_bip(duration=0.5), get_bip_bip(duration=1.0)]
        text = ["80s music", "Club techno"]

        inputs = processor(audio=audio, text=text, padding=True, return_tensors="pt")
        inputs = place_dict_on_device(inputs, device=torch_device)

        output_values = model.generate(**inputs, do_sample=False, guidance_scale=None, max_new_tokens=10)

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -0.0036, -0.0130, -0.0261, -0.0384, -0.0557, -0.0718, -0.0680, -0.0632,
                -0.0529, -0.0403, -0.0289, -0.0198, -0.0136, -0.0101, -0.0095, -0.0040,
            ]
        )
        # fmt: on

        self.assertTrue(
            output_values.shape == (2, 1, 36480)
        )  # input values take shape 32000 and we generate from there
        self.assertTrue(torch.allclose(output_values[0, 0, -16:].cpu(), EXPECTED_VALUES, atol=1e-4))


@require_torch
class MusicgenStereoIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-stereo-small").to(torch_device)

    @cached_property
    def processor(self):
        return MusicgenProcessor.from_pretrained("facebook/musicgen-stereo-small")

    @slow
    def test_generate_unconditional_greedy(self):
        model = self.model

        # only generate 1 sample with greedy - since it's deterministic all elements of the batch will be the same
        unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        unconditional_inputs = place_dict_on_device(unconditional_inputs, device=torch_device)

        output_values = model.generate(**unconditional_inputs, do_sample=False, max_new_tokens=12)

        # fmt: off
        EXPECTED_VALUES_LEFT = torch.tensor(
            [
                 0.0017,  0.0004,  0.0004,  0.0005,  0.0002,  0.0002, -0.0002, -0.0013,
                -0.0010, -0.0015, -0.0018, -0.0032, -0.0060, -0.0082, -0.0096, -0.0099,
            ]
        )
        EXPECTED_VALUES_RIGHT = torch.tensor(
            [
                0.0038, 0.0028, 0.0031,  0.0032,  0.0031,  0.0032,  0.0030,  0.0019,
                0.0021, 0.0015, 0.0009, -0.0008, -0.0040, -0.0067, -0.0087, -0.0096,
            ]
        )
        # fmt: on

        # (bsz, channels, seq_len)
        self.assertTrue(output_values.shape == (1, 2, 5760))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES_LEFT, atol=1e-4))
        self.assertTrue(torch.allclose(output_values[0, 1, :16].cpu(), EXPECTED_VALUES_RIGHT, atol=1e-4))

    @slow
    def test_generate_text_audio_prompt(self):
        model = self.model
        processor = self.processor

        # create stereo inputs
        audio = [get_bip_bip(duration=0.5)[None, :].repeat(2, 0), get_bip_bip(duration=1.0)[None, :].repeat(2, 0)]
        text = ["80s music", "Club techno"]

        inputs = processor(audio=audio, text=text, padding=True, return_tensors="pt")
        inputs = place_dict_on_device(inputs, device=torch_device)

        output_values = model.generate(**inputs, do_sample=False, guidance_scale=3.0, max_new_tokens=12)

        # fmt: off
        EXPECTED_VALUES_LEFT = torch.tensor(
            [
                 0.2535,  0.2008,  0.1471,  0.0896,  0.0306, -0.0200, -0.0501, -0.0728,
                -0.0832, -0.0856, -0.0867, -0.0884, -0.0864, -0.0866, -0.0744, -0.0430,
            ]
        )
        EXPECTED_VALUES_RIGHT = torch.tensor(
            [
                 0.1695,  0.1213,  0.0732,  0.0239, -0.0264, -0.0705, -0.0935, -0.1103,
                -0.1163, -0.1139, -0.1104, -0.1082, -0.1027, -0.1004, -0.0900, -0.0614,
            ]
        )
        # fmt: on

        # (bsz, channels, seq_len)
        self.assertTrue(output_values.shape == (2, 2, 37760))
        # input values take shape 32000 and we generate from there - we check the last (generated) values
        self.assertTrue(torch.allclose(output_values[0, 0, -16:].cpu(), EXPECTED_VALUES_LEFT, atol=1e-4))
        self.assertTrue(torch.allclose(output_values[0, 1, -16:].cpu(), EXPECTED_VALUES_RIGHT, atol=1e-4))
