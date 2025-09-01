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

# setup logging
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
import optuna
import datasets
datasets.logging.set_verbosity_error()

import transformers
transformers.utils.logging.set_verbosity_error()

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn.functional as F

from glob import glob
from copy import deepcopy
from tqdm import tqdm, trange
from typing import *
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, Dataset, DatasetDict
from functools import partial
from dataclasses import dataclass, field
from transformers import (
	AutoConfig,
	AutoTokenizer,
	# HfArgumentParser,
	AutoModelForCausalLM
)
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

if __name__ == '__main__':
	from parser import parse_args_into_dataclasses
	from constants import *
	from model_arguments import ModelArguments
	from kl_baseline_loss import (
		KLBaselineLoss,
		pad_tensor,
		pad_batch
	)
	from optimization_arguments import OptimizationArguments
	from data_training_arguments import DataTrainingArguments
else:
	from .parser import parse_args_into_dataclasses
	from .constants import *
	from .model_arguments import ModelArguments
	from .kl_baseline_loss import (
		KLBaselineLoss,
		pad_tensor,
		pad_batch
	)
	from .optimization_arguments import OptimizationArguments
	from .data_training_arguments import DataTrainingArguments

def parse_arguments(*args: tuple) -> tuple:
	'''
	Parse command line arguments.
	'''
	arg_classes = parse_args_into_dataclasses(ModelArguments, DataTrainingArguments, OptimizationArguments)
	return arg_classes

def load_model(model_name_or_path: str, *args, **kwargs):
	'''
	Loads the model using the appropriate function.
	'''
	if model_name_or_path in NEXT_WORD_MODELS:
		model = AutoModelForCausalLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
		# store any kwargs in the model so
		# we can pass them to the KL baseline loss later
		setattr(model, 'model_kwargs', kwargs)
		return model
	
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
	# store the tokenizer kwargs for use with KL baseline loss later
	setattr(tokenizer, 'tokenizer_kwargs', model_args.tokenizer_kwargs)
	
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
	
	if model_args.use_gpu and torch.cuda.is_available():
		model.to('cuda')
	elif model_args.use_gpu:
		logger.warning('`use_gpu` was set, but no GPU was found. Defaulting to CPU.')
	
	return tokenizer, model

def load_metadata(dataset_path: str) -> list[dict]:
	'''
	Loads the metadata file for a dataset.
	'''
	try:
		with gzip.open(dataset_path.replace('.txt.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
			metadata = [json.loads(l) for l in in_file.readlines()]
	except FileNotFoundError:
		logger.warning(
			f'No metadata file found for {dataset_path}. Using empty metadata.'
		)
		with gzip.open(dataset_path, 'rt', encoding='utf-8') as in_file:
			n_lines = len(in_file.readlines())
		
		metadata = [{} for _ in range(n_lines)]
		
	return metadata

def preprocess_dataset(
	dataset: Dataset,
	data_args: DataTrainingArguments, 
	tokenizer: AutoTokenizer,
	max_samples: int = None,
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
		
		if tokenizer.name_or_path in NEXT_WORD_MODELS:
			model_inputs['labels'] = model_inputs['input_ids'].copy()
		
		return model_inputs
	
	dataset = dataset.get(split)
	
	if max_samples is not None:
		dataset = dataset.select(range(max_samples))
	
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
	train_texts: list[str],
	validation_dataset: Dataset,
	validation_texts: list[str],
	model_args: ModelArguments,
	data_args: DataTrainingArguments,
	trial: optuna.trial.Trial = None,
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
		outputs: 'CausalLMOutput', 
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
			kl_loss = KL_baseline_loss()
			setattr(outputs, 'kl_loss', kl_loss)
			setattr(outputs, 'loss', outputs.loss + kl_loss)
			return outputs.loss
		else:
			# if we're in eval mode, we just want the dev loss to measure it,
			# but we don't want to add the KL baseline loss used for training.
			setattr(outputs, 'original_loss', outputs.loss)
			return outputs.loss
	
	def save_loss_plot(output_dir: str, metrics: pd.DataFrame) -> None:
		'''
		Saves a plot of the losses during training.
		'''
		def add_batch_number_prop(group):
			# we add one to the max since otherwise we'd end up getting to the next epoch,
			# since we'd be adding 1 for the final batch.
			group['batch_number_prop'] = group['batch_number']/(max(group['batch_number']) + 1)
			return group
		
		metrics = (
			metrics
				.drop(
					columns=[
						'item', 'input_text', 'word_num', 'token_num_in_word',
						'token', 'token_id', 'token_is_start_of_word', 'token_is_word',
						'surprisal', 'predicted_token',
					]
				)
				.drop_duplicates(ignore_index=True)
		)
		# set the values for each batch to intermediates
		metrics = (
			metrics
				.groupby('epoch')[metrics.columns]
				.apply(add_batch_number_prop)
				.reset_index(drop=True)
				.assign(
					original_epoch = lambda df: df['epoch'],
					epoch = lambda df: df['epoch'] + df['batch_number_prop']
				)
		)
		
		sns.set_theme(rc={'figure.figsize': (8.5*(16/9), 8.5)})
		plot_title = f'{metrics["model_name"].unique()[0]} ({metrics["n_params"].unique()[0]}) loss curves'
		
		# if we have the kl_loss, subtract it from the regular loss
		# for training to get the loss on just the training set
		if 'kl_loss' in metrics.columns:
			kl_loss = metrics.dropna(subset=['kl_loss']).reset_index(drop=True)
			kl_loss.dataset_type = kl_loss.dataset_type + ' + KL loss'
			
			metrics.loss = metrics.loss - metrics.kl_loss.fillna(0)
			metrics = pd.concat([metrics, kl_loss]).reset_index(drop=True)
			metrics = metrics.sort_values(['epoch', 'dataset_type']).reset_index(drop=True)
		
		metrics.dataset_name = metrics.dataset_type + ' (' + metrics.dataset_name + ')'
		
		p = sns.lineplot(
			data=metrics,
			x='epoch',
			y='loss',
			hue='dataset_name',
		)
		p.axes.set_ylim(bottom=0)
		p.set_title(plot_title)
		fig = p.get_figure()
		fig.savefig(os.path.join(output_dir, 'loss_curves.pdf'))
	
	if data_args.use_kl_baseline_loss:
		kl_baseline_dataset = DatasetDict.load_from_disk(data_args.kl_dataset)
		kl_baseline_dataset = preprocess_dataset(
			dataset=kl_baseline_dataset,
			data_args=data_args,
			tokenizer=tokenizer,
			max_samples=data_args.kl_max_samples,
			split='train'
		)
		KL_baseline_loss = KLBaselineLoss(
			model=model, 
			tokenizer=tokenizer, 
			dataset=kl_baseline_dataset,
			batch_size=data_args.kl_batch_size,
			scaleby=data_args.kl_scaleby,
			n_examples_per_batch=data_args.kl_n_examples_per_batch,
			reduction=data_args.kl_reduction,
			model_kwargs=model.model_kwargs, 
			tokenizer_kwargs=tokenizer.tokenizer_kwargs
		)
	
	basename = re.sub(r'[\\/]', '-', model.name_or_path)
	
	# make sure we don't run into directory conflicts.
	# somehow, this does happen rarely if we run in parallel, even
	# with microseconds included.
	os.makedirs(data_args.output_dir, exist_ok=True)
	
	# save arguments to file for reproduceability
	with open(os.path.join(data_args.output_dir, 'config.txt'), 'at') as out_file:
		_ = out_file.write(
			f'Model parameters: {model_args}\n\n'
			f'Data parameters: {data_args}\n\n\n'
		)
	
	metadata = dict(
		train=load_metadata(data_args.train_file),
		validation=load_metadata(data_args.validation_file),
	)
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=data_args.lr, weight_decay=0)
	
	if trial is None:
		metrics = []
	
	logger.info(f'Beginning training ({data_args.min_epochs=}).')
	with logging_redirect_tqdm(), trange(data_args.epochs) as epochs:
		patience_counter = 0
		best_dev_loss = float('inf')
		best_epoch = 0
		
		for epoch in epochs:
			dataloader = dict(
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
			epoch_train_losses = []
			if data_args.use_kl_baseline_loss:
				epoch_kl_losses = []
			
			for i, inputs in enumerate(dataloader['train']):
				n_examples_in_batch = inputs['input_ids'].shape[0]
				inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
				
				# use this as a unique input identifier
				input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
				n_observed_examples += n_examples_in_batch
				
				input_texts = train_texts[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				batch_metadata = metadata['train'][(n_observed_examples - n_examples_in_batch):n_observed_examples]
				
				model.train()
				optimizer.zero_grad(set_to_none=True) # this is supposed to be faster than .zero_grad()
				outputs = model(**inputs)
				loss = compute_loss(outputs)
				loss.backward()
				optimizer.step()
				model.eval()
				
				if trial is None:
					addl_kwargs = dict(dataset_name=os.path.split(data_args.train_file)[-1].rstrip('.txt.gz'))
					if hasattr(outputs, 'kl_loss'):
						addl_kwargs.update(dict(kl_loss=outputs.kl_loss.item()))
					
					metrics.extend(
						evaluate_batch(
							model=model,
							tokenizer=tokenizer,
							inputs=inputs,
							input_nums=input_nums,
							input_texts=input_texts,
							batch_outputs=outputs,
							batch_metadata=batch_metadata,
							epoch=epoch,
							batch_number=i,
							dataset_type='train',
							loss=loss.item(),
							**addl_kwargs
						)
					)
				
				epoch_train_losses.append(outputs.original_loss.item())
				if data_args.use_kl_baseline_loss:
					epoch_kl_losses.append(outputs.kl_loss.item())
			
			# pass through the validation set
			n_observed_examples = 0
			epoch_validation_losses = []
			for i, inputs in enumerate(dataloader['validation']):
				n_examples_in_batch = inputs['input_ids'].shape[0]
				inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
				
				# use this as a unique input identifier
				input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
				n_observed_examples += n_examples_in_batch
				
				input_texts = validation_texts[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				batch_metadata = metadata['validation'][(n_observed_examples - n_examples_in_batch):n_observed_examples]
				
				with torch.no_grad():
					outputs = model(**inputs)
				
				dev_loss = compute_loss(outputs, eval=True)
				
				if trial is None:
					addl_kwargs = dict(dataset_name=os.path.split(data_args.validation_file)[-1].rstrip('.txt.gz'))
					if data_args.use_kl_baseline_loss:
						addl_kwargs.update(dict(kl_loss=np.nan))
					
					metrics.extend(
						evaluate_batch(
							model=model,
							tokenizer=tokenizer,
							inputs=inputs,
							input_nums=input_nums,
							input_texts=validation_texts,
							batch_outputs=outputs,
							batch_metadata=batch_metadata,
							epoch=epoch,
							batch_number=i,
							dataset_type='validation',
							loss=dev_loss.item(),
							**addl_kwargs,
						)
					)
				
				epoch_validation_losses.append(dev_loss.item())
			
			# patience implementation
			if np.mean(epoch_validation_losses) < best_dev_loss - data_args.delta:
				best_dev_loss = np.mean(epoch_validation_losses)
				patience_counter = 0
				best_epoch = epoch + 1
				best_model_state_dict = deepcopy(model.state_dict())
			else:
				patience_counter += 1
				patience_counter = min(data_args.patience, patience_counter)
			
			postfix = dict(
				train_loss=f'{np.mean(epoch_train_losses).item():.2f}', 
				dev_loss=f'{np.mean(epoch_validation_losses).item():.2f}', 
				pat=f'{data_args.patience - patience_counter}', 
			)
			if data_args.use_kl_baseline_loss:
				postfix.update(dict(
					kl_loss=f'{np.mean(epoch_kl_losses).item():.2f}'
				))
			
			epochs.set_postfix(**postfix)
			
			# for trial pruning
			if trial is not None:
				trial.report(np.mean(epoch_validation_losses).item(), step=epoch)
				if trial.should_prune():
					raise optuna.TrialPruned()
			
			if patience_counter >= data_args.patience and epoch + 1 >= data_args.min_epochs:
				logger.info(
					f'Validation loss has not improved by {data_args.delta} in {patience_counter} epochs '
					f'(min_epochs={data_args.min_epochs}). Halting training at epoch {epoch}.'
				)
				break
		
		if trial is None:
			metrics = pd.DataFrame(metrics)
			metrics = metrics.assign(
				model_name=re.sub('["\']', '', model.config.name_or_path),
				task=get_model_task(model.config.name_or_path),
				n_params=f'{round(model.num_parameters()/1000000)}M',
			)
			
			move_to_beginning = ['model_name', 'task', 'n_params']
			metrics = metrics[move_to_beginning + [c for c in metrics.columns if not c in move_to_beginning]]
			save_loss_plot(output_dir=data_args.output_dir, metrics=metrics)
			
			metrics.to_csv(os.path.join(data_args.output_dir, 'metrics.csv.gz'), index=False, na_rep='NA')
			
			logger.info(f'Saving model state with lowest dev loss (epoch={best_epoch}) to disk')
			model.save_pretrained(
				save_directory=os.path.join(data_args.output_dir, 'model'), 
				state_dict=best_model_state_dict,
			)
			
			# save the tokenizer, too. This is a bit redundant, since we haven't modified it,
			# but it doesn't take up much space, and it makes it easier when loading later,
			# since we don't have to guess what the right tokenizer is.
			tokenizer.save_pretrained(
				save_directory=os.path.join(data_args.output_dir, 'tokenizer'),
			)
		
		# send back the lowest dev loss for optuna
		return best_dev_loss

def get_model_eval_function(model_name_or_path: str) -> Callable:
	'''
	Returns the appropriate function for eval based on the kind of 
	model.
	'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return evaluate_LM_batch
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def tokenize_texts(tokenizer: AutoTokenizer, text: list[str]) -> list[list[int]]:
	'''
	Tokenize a list of examples without special tokens for use during evaluation.
	'''
	tokenized = tokenizer(text, add_special_tokens=False)['input_ids']
	return tokenized

def evaluate_batch(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	batch_outputs: 'CausalLMOutput',
	input_nums: list[int] = None,
	batch_metadata: list[dict] = None,
	**additional_metadata
) -> list[dict]:
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
		batch_outputs=batch_outputs,
		batch_metadata=batch_metadata,
		additional_metadata=additional_metadata,
	)

def evaluate_LM_batch(
	model: Union[AutoModelForCausalLM],
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_nums: list[int],
	batch_outputs: 'CausalLMOutput',
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

def finetune_lm(
	model_args: ModelArguments, 
	data_args: DataTrainingArguments, 
	optim_args: OptimizationArguments,
	trial: Optional[optuna.trial.Trial] = None
) -> torch.Tensor:
	'''Main function.'''
	if optim_args.do_optimize:
		optim_args.set_suggested_values(data_args=data_args, trial=trial)
	
	if data_args.output_dir is None:
		data_args._set_output_dir(model_name=model_args.model_name_or_path)
	
	logger.info(f'Model parameters: {model_args}')
	logger.info(f'Data parameters: {data_args}')
	
	tokenizer, model = load_tokenizer_and_model(model_args)
	
	train_texts = load_dataset('text', data_files={'train': data_args.train_file})
	train_dataset = preprocess_dataset(
		dataset=train_texts,
		data_args=data_args,
		tokenizer=tokenizer,
		max_samples=data_args.max_train_samples,
		split='train'
	)
	train_texts = list(train_texts['train']['text'])
	
	validation_texts = load_dataset('text', data_files={'validation': data_args.validation_file})
	validation_dataset = preprocess_dataset(
		dataset=validation_texts,
		data_args=data_args,
		tokenizer=tokenizer,
		max_samples=data_args.max_val_samples,
		split='validation'
	)
	validation_texts = list(validation_texts['validation']['text'])
	
	return finetune_model(
		model=model, 
		tokenizer=tokenizer,
		train_dataset=train_dataset,
		train_texts=train_texts,
		validation_dataset=validation_dataset,
		validation_texts=validation_texts,
		model_args=model_args,
		data_args=data_args,
		trial=trial,
	)

def optimize_finetune_lm(
	model_args: ModelArguments, 
	data_args: DataTrainingArguments, 
	optim_args: OptimizationArguments,
) -> None:
	'''
	Wrapper function to optimize the results of finetune_lm.
	'''
	def save_optimization_results(study: optuna.study.study.Study) -> None:
		'''
		Save results from optimization study.
		params: study: optuna.study.study.Study: the study whose results are to be saved
		'''
		df = study.trials_dataframe()
		
		# remove any blank rows (those for which state is still RUNNING seem
		# to correspond to what the study object reports before it's been concluded)
		df = df[df.state != 'RUNNING']
		df = df.assign(
			model_name=re.sub('["\']', '', model_args.model_name_or_path),
			task=get_model_task(model_args.model_name_or_path),
		)
		
		df.to_csv(os.path.join(data_args.output_dir, 'optimization_trials.csv.gz'), index=False)
	
	if not optim_args.do_optimize:
		logger.warning(
			'`optimize_finetune_lm` was called, but `optim_args.do_optimize` was set to False. '
			'`finetune_lm` will be run instead.'
		)
		return finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
	
	logger.info(f'Optimization parameters: {optim_args.original_repr}')
	if data_args.output_dir is None:
		data_args._set_output_dir(model_name=model_args.model_name_or_path)
	
	optim_args.study.optimize(
		# we need to wrap finetune_lm in this function since optuna
		# doesn't support passing additional arguments to the optimized function,
		# and it will only pass the trial object as a positional argument in the
		# first slot. In general, we don't want to required using only kwargs in
		# in the `finetune_lm` function when we're not running a trial, so this is 
		# how we get around that
		lambda trial: finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args, trial=trial), 
		**optim_args.optimize_kwargs
	)
	
	logger.info(f'Best parameters: {optim_args.study.best_params}')
	logger.info(f'Best result: {optim_args.study.best_value}')
	save_optimization_results(study=optim_args.study)

if __name__ == '__main__':
	model_args, data_args, optim_args = parse_arguments(ModelArguments, DataTrainingArguments, OptimizationArguments)
	if not optim_args.do_optimize:
		_ = finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
	else:
		_ = optimize_finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
