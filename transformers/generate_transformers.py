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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class Arguments:
    def __init__(self, model_type, model_name_or_path, prompt="", length=20, stop_token="</s>", temperature=1.0,
                 repetition_penalty=1.0, k=0.0, p=0.9, padding_text="", xlm_language="", seed=42,
                 no_cuda=False, num_return_sequences=1):
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


def generate(model_type, model_name_or_path, prompt="", length=20, stop_token="</s>", temperature=1.0,
             repetition_penalty=1.0, k=0.0, p=0.9, padding_text="", xlm_language="", seed=42,
             no_cuda=False, num_return_sequences=1):
    # model_type Model type selected in the list
    # model_name_or_path Path to pre-trained model or shortcut name selected in the list

    # prompt
    # length
    # stop_token Token at which text generation is stopped
    # temperature help="temperature of 1.0 has no effect, lower tend toward greedy sampling
    # repetition_penalty primarily useful for CTRL model; in that case, use 1.2
    # k
    # p

    # padding_text Padding text for Transfo-XL and XLNet
    # xlm_language Optional language when used with the XLM model.

    # seed", type=int, default=42, help="random seed for initialization
    # --no_cuda Avoid using CUDA when available
    # num_return_sequences The number of samples to generate.

    args = Arguments(model_type, model_name_or_path, prompt, length, stop_token, temperature,
                     repetition_penalty, k, p, padding_text, xlm_language, seed,
                     no_cuda, num_return_sequences)

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)
    generated_sequences = []
    prompt_text = ""
    while prompt_text != "stop":
        while not len(prompt_text):
            prompt_text = args.prompt if args.prompt else input("Context >>> ")

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt",
                add_space_before_punct_symbol=True
            )
        else:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("ruGPT:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()
            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                    prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            )
            generated_sequences.append(total_sequence)
            # os.system('clear')
            print(total_sequence)
        prompt_text = ""
        if args.prompt:
            break

    return generated_sequences


