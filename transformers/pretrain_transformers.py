# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import logging
import os
import random
import re
import shutil
import string
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import razdel
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


class MaskingTokens:
    def __init__(self, tokenizer, mask_probability, rnd_probability):
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.rnd_probability = rnd_probability / (1 - self.mask_probability)

    def default_masking(self, tokens, probability_matrix):
        labels = tokens.clone()
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_probability)).bool() & masked_indices
        tokens[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, self.rnd_probability)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        tokens[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return tokens, labels


class TransformData:
    def __init__(self):
        self.token_cls = "[CLS]"
        self.token_sep = "[SEP]"
        self.token_unk = "[UNK]"
        self.bad_elements = list(string.punctuation) + ["...", self.token_cls, self.token_sep]

    def transform_token_answer(self, data):
        question, answer = data.question, data.answer
        return "".join([self.token_cls, question, self.token_sep, answer, self.token_sep])

    def transform_token_words(self, data):
        question, answer = data.question, data.answer
        data_string = f"{question} {answer}"
        words = razdel.tokenize(data_string)
        data_list = list(data_string)
        for j in range(len(words), -1, -1):
            word = words[j]
            text = word.text
            if text not in string.punctuation:
                start, end = word.start, word.end
                data_list.insert(start, self.token_unk)
        return "".join(data_list)

    def transform_token_answer_words(self, data):
        question, answer = data.question, data.answer
        words = razdel.tokenize(answer)
        answer_list = list(answer)
        for j in range(len(words), -1, -1):
            word = words[j]
            text = word.text
            if text not in string.punctuation:
                start, end = word.start, word.end
                answer_list.insert(start, self.token_unk)
        return f'{question}{self.token_unk}{"".join(answer_list)}'


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
transforms_data = TransformData()


class PdTextDataset(ABC, Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.Series, mlm_probability, block_size=512):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mlm_probability = mlm_probability
        self.tokens = []
        self.masks = []
        self.id_token_sep = 102
        self.fit_transform()

    def fit_transform(self):
        transformed_data = [self.transform_data(data_unit[1]) for data_unit in self.data.iterrows()]
        tokens_data = self.tokenizer.batch_encode_plus(transformed_data, add_special_tokens=False,
                                                       max_length=self.block_size)["input_ids"]
        for tokens_data_unit in tokens_data:
            tokens_data_unit, mask = self.transform_tokens(tokens_data_unit)
            self.tokens.append(list(tokens_data_unit))
            self.masks.append(list(mask))

    @abstractmethod
    def transform_data(self, data_unit):
        pass

    @abstractmethod
    def transform_tokens(self, tokens):
        pass

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return torch.Tensor([self.tokens[i], self.masks[i]])


class PdTextDatasetDefault(PdTextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.Series, mlm_probability, block_size=512):
        super(PdTextDatasetDefault, self).__init__(tokenizer, data, mlm_probability, block_size)

    def transform_data(self, data_unit):
        return data_unit

    def transform_tokens(self, tokens_unit):
        probability_matrix = torch.full(tokens_unit.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens_unit, already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = tokens_unit.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        return tokens_unit, probability_matrix


class PdTextDatasetTokensAnswer(PdTextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.Series, mlm_probability, block_size=512):
        super(PdTextDatasetTokensAnswer, self).__init__(tokenizer, data, mlm_probability, block_size)

    def transform_data(self, data_unit):
        return transforms_data.transform_token_answer(data_unit)

    def transform_tokens(self, tokens_unit):
        index_first_token_sep = tokens_unit.index(self.id_token_sep)
        tokens_unit = torch.tensor(tokens_unit, dtype=torch.long)
        probability_matrix = torch.full(tokens_unit.shape, self.mlm_probability)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens_unit, already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = tokens_unit.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        probability_matrix[:index_first_token_sep] = 0.0
        return tokens_unit, probability_matrix


class PdTextDatasetTokenWords(PdTextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.Series, mlm_probability, block_size=512):
        super(PdTextDatasetTokenWords, self).__init__(tokenizer, data, mlm_probability, block_size)

    def transform_data(self, data_unit):
        transforms_data.transform_token_words(data_unit)

    def transform_tokens(self, tokens_unit):
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens_unit, already_has_special_tokens=True)
        tokens_unit = np.array(tokens_unit)
        idx = np.where(np.array(tokens_unit) == transforms_data.token_unk)[0]
        new_idx = idx - np.arange(idx.shape[0])
        tokens = np.split(tokens_unit[tokens_unit != transforms_data.token_unk], new_idx)
        tokens = torch.tensor(tokens, dtype=torch.long)
        probability_matrix = torch.full(tokens_unit.shape, self.mlm_probability)

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        probability_matrices = torch.split(probability_matrix, new_idx)
        return tokens, probability_matrices


class PdTextDatasetTokenAnswerWords(PdTextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.Series, mlm_probability, block_size=512):
        super(PdTextDatasetTokenAnswerWords, self).__init__(tokenizer, data, mlm_probability, block_size)

    def transform_data(self, data_unit):
        transforms_data.transform_token_answer_words(data_unit)

    def transform_tokens(self, tokens_unit):
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens_unit, already_has_special_tokens=True)
        tokens_unit = np.array(tokens_unit)
        idx = np.where(np.array(tokens_unit) == transforms_data.token_unk)[0]
        new_idx = idx - np.arange(idx.shape[0])
        tokens = np.split(tokens_unit[tokens_unit != transforms_data.token_unk], new_idx)
        tokens = torch.tensor(tokens, dtype=torch.long)
        probability_matrix = torch.full(tokens_unit.shape, self.mlm_probability)

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix[:tokens[0].shape[0]] = 0.0

        probability_matrices = torch.split(probability_matrix, new_idx)






        # tokens_split = tokens_unit.split(transforms_data.token_unk)
        # idx_answer = np.where(answer_tokens == transforms_data.token_unk)[0]
        # idx_answer -= np.arange(idx_answer.shape[0])
        #
        # answer_tokens = np.split(answer_tokens[answer_tokens != transforms_data.token_unk], idx_answer)
        # answer_tokens = torch.tensor(answer_tokens, dtype=torch.long)
        #
        # probability_matrix = torch.full(tokens.shape, self.mlm_probability)
        # special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens_unit, already_has_special_tokens=True)
        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        # probability_matrices = torch.split(probability_matrix, idx)
        return tokens, probability_matrices


def load_and_cache_examples(args, tokenizer, evaluate=False):
    masking_tokens = MaskingTokens(tokenizer, args.mask_probability, args.rnd_probability)
    file = args.eval_data if evaluate else args.train_data
    if args.use_token_answer:
        args.func_masking_tokens = masking_tokens.default_masking
        return PdTextDatasetTokensAnswer(tokenizer, file, args.block_size)
    elif args.user_words:
        return PdTextDatasetTokenWords(tokenizer, file, args.block_size)
    elif args.use_answer_words:
        return PdTextDatasetTokenAnswerWords(tokenizer, file, args.block_size)
    args.func_masking_tokens = masking_tokens.default_masking
    return PdTextDatasetDefault(tokenizer, file, args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    inputs, labels = torch.empty(inputs.shape), torch.empty(inputs.shape)
    for i, input_data in enumerate(inputs):
        tokens, probability_matrix = input_data[0], input_data[1]
        tokens, labels_unit = args.func_masking_tokens(tokens, probability_matrix)
        inputs[i] = tokens
        labels[i] = labels_unit
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    print(train_sampler)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )
    print(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        print(epoch_iterator)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.batch_size = args.batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


class Arguments:
    def __init__(self, output_dir, model_type, model_name_or_path, train_data, eval_data, should_continue=False,
                 mlm=False, mlm_probability=0.15, config_name=None, tokenizer_name=None, cache_dir=None,
                 block_size=-1, do_train=False, do_eval=False, evaluate_during_training=False,
                 batch_size=1, gradient_accumulation_steps=1, learning_rate=2e-5, weight_decay=0.01,
                 adam_epsilon=1e-8, max_grad_norm=1.0, num_train_epochs=1.0, max_steps=-1, warmup_steps=0,
                 logging_steps=500,
                 save_steps=500, save_total_limit=None, eval_all_checkpoints=False, no_cuda=False,
                 overwrite_output_dir=False,
                 overwrite_cache=False, seed=42, fp16=False, fp16_opt_level="O1", local_rank=-1, server_ip="",
                 server_port="",
                 mask_probability=0.8,
                 rnd_probability=0.1,
                 use_answer=False,
                 user_words=False,
                 use_answer_words=False):
        self.output_dir = output_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.train_data = train_data
        self.eval_data = eval_data
        self.should_continue = should_continue
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.block_size = block_size
        self.do_train = do_train
        self.do_eval = do_eval
        self.evaluate_during_training = evaluate_during_training
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.eval_all_checkpoints = eval_all_checkpoints
        self.no_cuda = no_cuda
        self.overwrite_output_dir = overwrite_output_dir
        self.overwrite_cache = overwrite_cache
        self.seed = seed
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.local_rank = local_rank

        self.mask_probability = mask_probability
        self.rnd_probability = rnd_probability

        self.use_token_answer = use_answer
        self.user_words = user_words
        self.use_answer_words = use_answer_words
        self.func_masking_tokens = None

        self.server_ip = server_ip
        self.server_port = server_port
        self.n_gpu = 0
        self.device = None


def train_transformers(output_dir, model_type, model_name_or_path, train_data, eval_data, should_continue=False,
                       mlm=False, mlm_probability=0.15, config_name=None, tokenizer_name=None, cache_dir=None,
                       block_size=-1, do_train=True, do_eval=False, evaluate_during_training=False,
                       batch_size=1, gradient_accumulation_steps=1, learning_rate=2e-5, weight_decay=0.01,
                       adam_epsilon=1e-8, max_grad_norm=1.0, num_train_epochs=1.0, max_steps=-1, warmup_steps=0,
                       logging_steps=500,
                       save_steps=500, save_total_limit=None, eval_all_checkpoints=False, no_cuda=False,
                       overwrite_output_dir=False,
                       overwrite_cache=False, seed=42, fp16=False, fp16_opt_level="O1", local_rank=-1, server_ip="",
                       server_port="",
                       mask_probability=0.8, rnd_probability=0.1,
                       use_answer=False,
                       user_words=False,
                       use_answer_words=False):
    # train_data help="The input training data file (a text file)."
    # output_dir help="The output directory where the model predictions and checkpoints will be written."
    # model_type help="The model architecture to be trained or fine-tuned."
    # eval_data help="An optional input evaluation data file to evaluate the perplexity on (a text file)."
    # should_continue help="Whether to continue from latest checkpoint in output_dir"
    # model_name_or_path help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
    # mlm" help="Train with masked-language modeling loss instead of language modeling."
    # mlm_probability help="Ratio of tokens to mask for masked language modeling loss"
    # config_name help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config."
    # tokenizer_name help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer."
    # cache_dir help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)"
    # block_size help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",

    # do_train help="Whether to run training."
    # do_eval help="Whether to run eval on the dev set."
    # evaluate_during_training help="Run evaluation during training at each logging step."

    # per_gpu_train_batch_size help="Batch size per GPU/CPU for training."
    # per_gpu_eval_batch_size help="Batch size per GPU/CPU for evaluation."
    # gradient_accumulation_steps help="Number of updates steps to accumulate before performing a backward/update pass."
    # learning_rate help="The initial learning rate for Adam.
    # weight_decay help="Weight decay if we apply some."
    # adam_epsilon help="Epsilon for Adam optimizer."
    # max_grad_norm help="Max gradient norm."
    # num_train_epochs help="Total number of training epochs to perform."
    # max_steps help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    # warmup_steps help="Linear warmup over warmup_steps."

    # logging_steps help="Log every X updates steps."
    # save_steps help="Save checkpoint every X updates steps."
    # save_total_limit help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete "
    # eval_all_checkpoints help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number"
    # no_cuda true", help="Avoid using CUDA when available"
    # overwrite_output_dir help="Overwrite the content of the output directory"
    # overwrite_cache help="Overwrite the cached training and evaluation sets"
    # seed help="random seed for initialization")
    # fp16 help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    # fp16_opt_level help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html"
    # local_rank help="For distributed training: local_rank"
    # server_ip help="For distant debugging."
    # server_port help="For distant debugging."

    # args = {"output_dir": output_dir, "model_type": model_type, "model_name_or_path": model_name_or_path,
    #         "train_data": train_data, "eval_data": eval_data, "should_continue": should_continue,
    #         "mlm": mlm, "mlm_probability": mlm_probability, "config_name": config_name, "tokenizer_name": tokenizer_name,
    #         "cache_dir": cache_dir, "block_size": block_size, "do_train": do_train, "do_eval": do_eval,
    #         "evaluate_during_training": evaluate_during_training, "per_gpu_train_batch_size": per_gpu_train_batch_size,
    #         "per_gpu_eval_batch_size": per_gpu_eval_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
    #         "learning_rate": learning_rate, "weight_decay": weight_decay,
    #         "adam_epsilon": adam_epsilon, "max_grad_norm": max_grad_norm, "num_train_epochs": num_train_epochs,
    #         "max_steps": max_steps, "warmup_steps": warmup_steps, "logging_steps": logging_steps,
    #         "save_steps": save_steps, "save_total_limit": save_total_limit,
    #         "eval_all_checkpoints": eval_all_checkpoints,"no_cuda": no_cuda, "overwrite_output_dir": overwrite_output_dir,
    #         "overwrite_cache": overwrite_cache, "seed": seed, "fp16": fp16, "fp16_opt_level": fp16_opt_level,
    #         "local_rank": local_rank, "server_ip": server_ip, "server_port": server_port}

    args = Arguments(output_dir, model_type, model_name_or_path, train_data, eval_data, should_continue,
                     mlm, mlm_probability, config_name, tokenizer_name, cache_dir,
                     block_size, do_train, do_eval, evaluate_during_training, batch_size,
                     gradient_accumulation_steps, learning_rate, weight_decay,
                     adam_epsilon, max_grad_norm, num_train_epochs, max_steps, warmup_steps, logging_steps,
                     save_steps, save_total_limit, eval_all_checkpoints, no_cuda, overwrite_output_dir,
                     overwrite_cache, seed, fp16, fp16_opt_level, local_rank, server_ip, server_port,
                     mask_probability=mask_probability,
                     rnd_probability=rnd_probability,
                     use_answer=use_answer, user_words=user_words, use_answer_words=use_answer_words)

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from "
            "another script, save it, "
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from "
            "another script, save it, "
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed
            # training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results
