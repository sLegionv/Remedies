import os

import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


class ModelGenerateText:
    def __init__(self, model_type, model_name_or_path, prompt="", length=20, stop_token="</s>", temperature=1.0,
                 repetition_penalty=1.0, k=0, p=0.9, padding_text="", xlm_language="", seed=42, no_cuda=False,
                 num_return_sequences=1):
        def adjust_length_to_model(length, max_sequence_length):
            if length < 0 and max_sequence_length > 0:
                length = max_sequence_length
            elif 0 < max_sequence_length < length:
                length = max_sequence_length  # No generation bigger than model size
            elif length < 0:
                length = MAX_LENGTH  # avoid infinite loop
            return length

        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.prompt = prompt
        self.length = length
        self.stop_token = stop_token
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.k = k
        self.p = p
        self.padding_text = padding_text
        self.xlm_language = xlm_language
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_return_sequences = num_return_sequences
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()

        self.MAX_LENGTH = int(10000)
        self.MODEL_CLASSES = {
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
            "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
            "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
            "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
            "xlm": (XLMWithLMHeadModel, XLMTokenizer),
        }

        # Initialize the model and tokenizer
        try:
            self.model_type = self.model_type.lower()
            model_class, tokenizer_class = self.MODEL_CLASSES[self.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)
        self.model = model_class.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.length = adjust_length_to_model(self.length, max_sequence_length=self.model.config.max_position_embeddings)

    def generate_text(self, data, isPrint=False):  # data = [text1, text2]
        generated_sequences = []
        return_sequences = []

        def prepare_xlnet_input(model, padding_text, tokenizer, prompt_text, PADDING_TEXT=""):
            prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
            return prompt_text

        def prepare_transfoxl_input(model, padding_text, tokenizer, prompt_text, PADDING_TEXT=""):
            prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
            return prompt_text

        def prepare_ctrl_input(model, padding_text, tokenizer, prompt_text, PADDING_TEXT=""):
            return prompt_text

        def prepare_xlm_input(model, padding_text, tokenizer, prompt_text, PADDING_TEXT=""):
            return prompt_text

        PREPROCESSING_FUNCTIONS = {
            "ctrl": prepare_ctrl_input,
            "xlm": prepare_xlm_input,
            "xlnet": prepare_xlnet_input,
            "transfo-xl": prepare_transfoxl_input,
        }
        for prompt_text in data:
            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = self.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(self.model_type)
                preprocessed_prompt_text = prepare_input(model=self.model, padding_text=self.padding_text,
                                                         tokenizer=self.tokenizer, prompt_text=prompt_text)
                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt",
                    add_space_before_punct_symbol=True
                )
            else:
                encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(self.device)
            output_sequences = self.model.generate(
                input_ids=encoded_prompt,
                max_length=self.length + len(encoded_prompt[0]),
                temperature=self.temperature,
                top_k=self.k,
                top_p=self.p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                num_return_sequences=self.num_return_sequences,
            )
            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()
                # Decode text
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                # Remove all text after the stop token
                text = text[: text.find(self.stop_token) if self.stop_token else None]
                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                        prompt_text + text[
                                      len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                )
                generated_sequences.append(total_sequence)
                if isPrint:
                    print(total_sequence)
        return generated_sequences
