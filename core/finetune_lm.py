########################################################################################################
# This script is (heavily) adapted from a script by HuggingFace Inc. 								   #
# It has been modified by Michael Wilson (2025).   			   										   #
########################################################################################################
#
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO
)
logger = logging.getLogger(__name__)

import os
import re
import sys
import gzip
import json
import time
import torch
import datasets
datasets.logging.set_verbosity_error()

import transformers
transformers.utils.logging.set_verbosity_error()

import pandas as pd
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
from typing import *
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from transformers import (
	AutoConfig,
	AutoTokenizer,
	HfArgumentParser,
	AutoModelForCausalLM
)

if __name__ == '__main__':
	from constants import *
	from model_arguments import ModelArguments
	from kl_baseline_loss import KLBaselineLoss
	from data_training_arguments import DataTrainingArguments
else:
	from .constants import *
	from .model_arguments import ModelArguments
	from .kl_baseline_loss import KLBaselineLoss
	from .data_training_arguments import DataTrainingArguments

def parse_arguments(*args: Tuple) -> Tuple:
	'''
	Parse command line arguments into ModelArguments and DataTrainingArguments.
	See ModelArguments and DataTrainingArguments for details.
	'''
	parser = HfArgumentParser(args)
	model_args, data_args = parser.parse_args_into_dataclasses()
	
	return model_args, data_args

def load_model(model_name_or_path: str, *args, **kwargs):
	'''
	Loads the model using the appropriate function.
	'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return AutoModelForCausalLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def load_tokenizer_and_model(model_args: ModelArguments) -> Tuple:
	'''Loads the tokenizer and model as specified in model_args.'''
	if model_args.model_name_or_path in NON_HF_LLAMA_MODELS:
		raise ValueError(model_not_supported_message(model_name_or_path))
	
	config = AutoConfig.from_pretrained(
		model_args.config_name,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=model_args.token
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=model_args.token,
		**model_args.tokenizer_kwargs,
	)
	
	if tokenizer.name_or_path in HF_LLAMA_MODELS:
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	
	model = load_model(
		model_args.model_name_or_path,
		from_flax=model_args.from_flax,
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		token=model_args.token,
	)
	
	if model.name_or_path in HF_LLAMA_MODELS:
		model.resize_token_embeddings(len(tokenizer))
	
	if model_args.model_name_or_path in GPT2_MODELS:
		tokenizer.pad_token = tokenizer.eos_token
		model.config.pad_token_id = model.config.eos_token_id
		tokenizer.bos_token = tokenizer.eos_token
		model.config.bos_token_id = model.config.eos_token_id
	
	return tokenizer, model

def preprocess_dataset(
	dataset: Dataset,
	data_args: DataTrainingArguments, 
	tokenizer: AutoTokenizer,
	split: str = 'train',
) -> Dataset:
	'''
	Formats the dataset for use with a model.
	
	params:
		dataset (Dataset)			: a huggingface dataset. Must contain a "split split,
									  with examples in the "text" column.
		split (str)					: the name of the split to use
		data_args (DataTrainingArguments): the arguments containing information about the data.
					  				  see the DataTrainingArguments class for more details.
		tokenizer (AutoTokenizer)	: the tokenizer to use to prepare the examples for the model.
	
	returns:
		Dataset 					: the dataset formatted for use with a model.
	'''
	drop_cols = dataset.get(split).column_names
	
	def preprocess_function(examples: list[str]) -> dict:
		'''Tokenizes a batch of string inputs.'''
		model_inputs = tokenizer(
			examples['text'], 
			max_length=data_args.max_length, 
			padding=True,
			truncation=True
		)
		
		start_ids = [tokenizer.bos_token_id, tokenizer.cls_token_id]
		start_ids = [token_id for token_id in start_ids if token_id is not None]
		if any(start_ids):
			start_id = start_ids[0]
			for i in range(len(model_inputs['input_ids'])):
				# add the cls/bos token to models that don't automatically include it
				# such as gpt2. we also need to ensure the other keys are the same length
				if model_inputs['input_ids'][i][0] != start_id:
					model_inputs['input_ids'][i].insert(0, start_id)
					for k in model_inputs.keys():
						if k == 'attention_mask':
							model_inputs[k][i].insert(0, 1)
						
						if k == 'token_type_ids':
							model_inputs[k][i].insert(0, 0)
	
		return model_inputs
	
	dataset = dataset.get(split)
	
	if data_args.max_test_samples is not None:
		dataset = dataset.select(range(data_args.max_test_samples))
	
	dataset = dataset.map(
		preprocess_function,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
		remove_columns=drop_cols,
		load_from_cache_file=not data_args.overwrite_cache,
	)
	
	dataset.set_format(type='torch')
	
	return dataset

def get_model_task(model_name_or_path: str) -> str:
	'''Returns the model task based on the name.'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return 'LM'
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def align_words_to_subword_tokens(
	tokenizer: AutoTokenizer, 
	words: list[str], 
	tokens: list[int]
) -> list[list[int]]:
	'''
	Aligns words to subword tokens.
	
	params:
		tokenizer: AutoTokenizer: the tokenizer used to generate `tokens`
		words (list[str]): a list of words to align
		tokens (list[int]): a list of tokens generated from the sequence of words.
	
	returns:
		list of list of ints, which is the same length as `words`.
		Each sublist contains the tokens corresponding to the word
		at the same position as the sublist in `words`.	
	
	raises:
		IndexError: if the words and tokens cannot be aligned.
	'''
	# pop works backward
	num_words = len(words)
	tokens = tokens[::-1]
	words = words[::-1]
	
	uncased = tokenizer.tokenize('A') == tokenizer.tokenize('a')
	if uncased:
		words = [w.lower() for w in words]
	
	aligned = []
	while tokens:
		aligned_tokens = [tokens.pop()]
		word = words.pop()
		while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != word:
			aligned_tokens += [tokens.pop()]
	
		aligned.append(aligned_tokens)
	
	assert len(aligned) == num_words, (
		f'Unable to find {num_words} in text.'
	)
	
	return aligned

def finetune_model(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	train_dataset: Dataset,
	validation_dataset: Dataset,
	data_args: DataTrainingArguments
) -> None:
	'''
	Finetunes a language model.
	
	params:
	model (AutoModelForCasualLM): the model to finetune.
	tokenizer (AutoTokenizer): the tokenizer for the model.
	train_dataset (Dataset): the training dataset.
	validation_dataset (Dataset): the validation dataset.
	data_args (DataTrainingArguments): additional DataTrainingArguments.
									   see DataTrainingArguments for details.
	'''
	def compute_loss(
		outputs: 'MaskedLMOutput', 
		eval: bool = False
	) -> torch.Tensor:
		'''
		Computes loss on language model outputs according to the DataTrainingArguments settings
		
			params:
				outputs (MaskedLMOutput): outputs from a masked language model
				eval (bool)				: whether the model is being evaluated or trained on the basis of
										  the loss. for eval/dev sets, we never use the KL baseline loss,
										  since we are interested in measuring overfitting to the generalization sets
			
			returns:
				loss (torch.Tensor)		: if using KL divergence loss, add KL divergence loss to the original model's outputs loss
										  KL divergence loss is defined at class initialization according to passed options
										  otherwise, return the original model loss
		'''
		if data_args.use_kl_baseline_loss and not eval:
			setattr(outputs, 'original_loss', outputs.loss)
			setattr(outputs, 'loss', outputs.loss + KL_baseline_loss())
			return outputs.loss
		else:
			setattr(outputs, 'original_loss', outputs.loss)
			return outputs.loss
	
	if data_args.use_kl_baseline_loss:
		KL_baseline_loss = kl_baseline_loss.KLBaselineLoss(
			model=model, 
			tokenizer=tokenizer, 
			dataset=load_dataset(data_args.kl_dataset),
			batch_size=data_args.kl_batch_size,
			scaleby=data_args.kl_scaleby,
			n_examples_per_step=data_args.kl_n_examples_per_step,
			model_kwargs=model.model_kwargs, 
			tokenizer_kwargs=model.tokenizer_kwargs
		)
	
	basename = re.sub(r'[\\/]', '-', model.name_or_path)
	
	# make sure we don't run into directory conflicts.
	# somehow, this does happen rarely if we run in parallel, even
	# with microseconds included.
	output_dir = os.path.join(
		os.path.split(train_file)[-1].replace('.txt.gz', ''),
		model.name_or_path.replace("/", "-"),
		datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
	)
	while os.path.isdir(output_dir):
		output_dir = os.path.join(
			os.path.split(train_file)[-1].replace('.txt.gz', ''),
			model.name_or_path.replace("/", "-"),
			datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
		)
	
	# save arguments to file for reproducability
	with open('config.txt', 'wt') as out_file:
		_ = out_file.write(
			f'Model parameters: {model_args}\n\n'
			f'Data parameters: {data_args}'
		)
	
	metadata = dict(
		train=load_metadata(data_args.train_file),
		validation=load_metadata(data_args.validation_file),
	)
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=data_args.lr, weight_decay=0)
	
	metrics = []
	with logging_redirect_tqdm(), trange(data_args.epochs) as epochs:
		patience_counter = 0
		best_dev_loss = np.inf
		best_epoch = 0
		
		for epoch in epochs:
			dataloader = Dict(
				train=DataLoader(
					train_dataset,
					batch_size=data_args.per_device_train_batch_size,
					collate_fn=pad_batch
				),
				validation=DataLoader(
					validation_dataset,
					batch_size=data_args.per_device_validation_batch_size,
					collate_fn=pad_batch
				)
			)
			
			# pass through the training set
			n_observed_examples = 0
			for i, inputs in enumerate(dataloader['train']):
				n_examples_in_batch = inputs['input_ids'].shape[0]
				
				# use this as a unique input identifier
				input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
				n_observed_examples += n_examples_in_batch
				
				batch_metadata = metadata['train'][(n_observed_examples - n_examples_in_batch):n_observed_examples]
				
				model.train()
				optimizer.zero_grad(set_to_none=True) # this is supposed to be faster than .zero_grad()
				outputs = model(**inputs)
				loss = compute_loss(outputs)
				loss.backward()
				optimizer.step()
				model.eval()
				
				metrics.extend(
					evaluate_batch(
						model=model,
						tokenizer=tokenizer,
						inputs=inputs,
						input_nums=input_nums,
						outputs=outputs,
						batch_metadata=batch_metadata,
						epoch=epoch,
						batch_number=i,
						dataset_type='train',
						loss=loss.item(),
					)
				)
			
			# pass through the validation set
			n_observed_examples = 0
			for i, inputs in enumerate(dataloader['validation']):
				n_examples_in_batch = inputs['input_ids'].shape[0]
				
				# use this as a unique input identifier
				input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
				n_observed_examples += n_examples_in_batch
				
				batch_metadata = metadata['validation'][(n_observed_examples - n_examples_in_batch):n_observed_examples]
				
				with torch.no_grad():
					outputs = model(**inputs)
				
				dev_loss = compute_loss(outputs, eval=True)
				
				metrics.extend(
					evaluate_batch(
						model=model,
						tokenizer=tokenizer,
						inputs=inputs,
						input_nums=input_nums,
						outputs=outputs,
						batch_metadata=batch_metadata,
						epoch=epoch,
						batch_number=i,
						dataset_type='validation',
						loss=dev_loss.item(),
					)
				)
			
			# patience implementation
			if dev_loss < best_dev_loss - data_args.delta:
				best_dev_loss = dev_loss
				patience_counter = 0
				best_epoch = epoch + 1
				best_model_state_dict = deepcopy(model.state_dict())
			else:
				patience_counter += 1
				patience_counter = min(patience, patience_counter)
			
			if patience_counter >= patience and epoch + 1 >= data_args.min_epochs:
				log.info(
					f'Validation loss has not improved by {data_args.delta} in {patience_counter} epochs '
					f'(min_epochs={data_args.min_epochs}). Halting training at epoch {epoch}.'
				)
				break
		
		metrics = pd.DataFrame(metrics)
		metrics = metrics.assign(
			model_name=re.sub('["\']', '', model.config.name_or_path),
			task=get_model_task(model.config.name_or_path),
			n_params=f'{round(model.num_parameters()/1000000)}M',
		)
		
		move_to_beginning = ['model_name', 'task', 'n_params']
		metrics = metrics[move_to_beginning + [c for c in metrics.columns if not c in move_to_beginning]]
		
		# we've already created the output directory
		# if we're saving tmp files
		if not data_args.save_tmp:
			os.makedirs(data_args.output_dir, exist_ok=True)
		
		metrics.to_csv('metrics.csv.gz', index=False, na_rep='NA')
		
		# save the best-performing model state
		log.info(f'Saving model state with lowest dev loss (epoch={best_epoch}) to disk')
		with open(os.path.join(data_args.output_dir, 'model.pt'), 'wb') as f:
			torch.save(best_model_state_dict, f)

def evaluate_batch(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_texts: list[str],
	input_nums: List[int] = None,
	batch_outputs,
	batch_metadata: List[Dict] = None,
	**additional_metadata
) -> List[Dict]:
	'''Record metrics for a single batch of inputs, depending on the model type.'''
	if input_nums is None:
		input_nums = range(len(inputs['input_ids'].shape[0]))
	
	if batch_metadata is None:
		batch_metadata = {}
	
	model_eval_function = get_model_eval_function(model_name_or_path=model.name_or_path)
	
	return model_eval_function(
		model=model, 
		tokenizer=tokenizer, 
		inputs=inputs, 
		input_texts=input_texts,
		input_nums=input_nums, 
		batch_outputs=outputs,
		batch_metadata=batch_metadata,
		additional_metadata=additional_metadata,
	)

def evaluate_LM_batch(
	model: Union[AutoModelForCausalLM, llama.LLaMA, llama_2.Llama, llama_3.Llama],
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_nums: list[int],
	batch_outputs,
	batch_metadata: list[dict],
	additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a batch of examples for a Language Model.
	For each input, determines the surprisal of each eval token
	as a prediction for the next token.
	'''
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs.logits, dim=-1)
	
	next_word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	
	metrics = []
	records = zip(input_nums, input_texts, next_word_ids, batch_surprisals, batch_metadata)
	for input_num, input_text, next_word_tokens, surprisal, example_metadata in records:
		input_words = input_text.split()
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer, 
			words=input_words, 
			tokens=next_word_tokens
		)
		
		tokens_seen = 0
		for word_num, tokens in enumerate(aligned_tokens):
			for token_num, token in enumerate(tokens):
				metrics.extend([{
					'item': input_num,
					'input_text': input_text,
					'word_num': word_num,
					'token_num_in_word': token_num,
					'token': tokenizer.decode(token),
					'token_id': token,
					'token_is_start_of_word': token_num == 0,
					'token_is_word': len(tokens) == 1,
					'surprisal': surprisal[tokens_seen,token].item(),
					'predicted_token': tokenizer.decode(
						torch.argmin(surprisal[tokens_seen,:], dim=-1).item()
					),
					**example_metadata,
					**additional_metadata,
				}])
				tokens_seen += 1
	
	return metrics

def finetune_lm() -> None:
	'''Main function.'''
	breakpoint()
	model_args, data_args = parse_cl_arguments(ModelArguments, DataTrainingArguments)
	
	logger.info(f'Model parameters: {model_args}')
	logger.info(f'Data parameters: {data_args}')
	
	tokenizer, model = load_tokenizer_and_model(model_args)
	train_dataset = load_dataset('text', data_files={'train': self.train_file})
	validation_dataset = load_dataset('text', data_files={'validation': self.validation_file})
	
	finetune_model(
		model=model, 
		tokenizer=tokenizer,
		train_dataset=train_dataset,
		validation_dataset=validation_dataset,
		data_args=data_args
	)

if __name__ == '__main__':
	finetune_lm()