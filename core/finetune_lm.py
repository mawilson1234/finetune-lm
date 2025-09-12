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
import types
import torch
import random
import optuna
import datasets
datasets.logging.set_verbosity_error()

import tempfile
import loss_classes
import transformers
transformers.utils.logging.set_verbosity_error()

import data_preprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn.functional as F

from glob import glob
from copy import deepcopy
from tqdm import tqdm, trange
from pprint import pformat
from PyPDF2 import PdfMerger
from typing import *
from pathlib import Path
from datetime import datetime
from dataset import Dataset
from datasets import load_dataset, DatasetDict
from functools import partial
from dataclasses import dataclass, field
from transformers import (
	AutoConfig,
	AutoTokenizer,
	AutoModelForMaskedLM,
	AutoModelForCausalLM,
	AutoModelForSeq2SeqLM,
)
from transformers.utils.generic import ModelOutput
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

# we need to do this in a try block so that this 
# isn't imported lazily, since we'll access the 
# vars of the module to find the plot functions later
try:
	import optuna.visualization
finally:
	pass

if __name__ == '__main__':
	from parser import parse_args_into_dataclasses
	from constants import *
	from model_arguments import ModelArguments
	from optimization_arguments import OptimizationArguments
	from data_training_arguments import DataTrainingArguments
else:
	from .parser import parse_args_into_dataclasses
	from .constants import *
	from .model_arguments import ModelArguments
	from .optimization_arguments import OptimizationArguments
	from .data_training_arguments import DataTrainingArguments

def parse_arguments(*args: tuple) -> tuple:
	'''
	Parse command line arguments.
	'''
	arg_classes = parse_args_into_dataclasses(
		ModelArguments, DataTrainingArguments, OptimizationArguments
	)
	return arg_classes

def set_seed(seed: int) -> None:
	'''
	Set all random seeds.
	'''
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

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
	
	tokenizer = load_tokenizer(
		model_args.tokenizer_name,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=model_args.token,
		**model_args.tokenizer_kwargs,
	)
	
	model = load_model(
		model_args.model_name_or_path,
		from_flax=model_args.from_flax,
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		token=model_args.token,
	)
	
	# if model.name_or_path in HF_LLAMA_MODELS:
	# 	model.resize_token_embeddings(len(tokenizer))
	
	if model_args.use_gpu and torch.cuda.is_available():
		model.to('cuda')
	elif model_args.use_gpu:
		logger.warning('`use_gpu` was set, but no GPU was found. Defaulting to CPU.')
	
	for fn in model_args.model_modifier_fns:
		kwargs = model_modifier_fn_kwargs.get(fn.__name__, {})
		tokenizer, model = fn(model=model, tokenizer=tokenizer, **kwargs)
	
	return tokenizer, model

def get_model_task(model_name_or_path: str) -> str:
	'''Returns the model task based on the name.'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return 'LM'
	
	if model_name_or_path in MASKED_LANGUAGE_MODELS:
		return 'MLM'
	
	if model_name_or_path in SEQ2SEQ_MODELS:
		return 'Seq2Seq'
	
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
		IndexError, AssertionError: if the words and tokens cannot be aligned.
	'''
	# pop works backward, so reverse first
	num_words = len(words)
	tokens = tokens[::-1]
	words = words[::-1]
	
	# these should be in every tokenizer that has cased letters,
	# so we can use them to test
	uncased = tokenizer.tokenize('A') == tokenizer.tokenize('a')
	if uncased:
		# sometimes uncased tokenizer have uppercase special tokens, so we need
		# to deal with that and NOT uncase special tokens if they're already
		# in the inputs.
		if any(t in w for t in tokenizer.all_special_tokens for w in words):
			special_words = []
			for w in words:
				# this handles a case where we have a word without a special
				# token and still want to lower case it.
				if not any(t in w for t in tokenizer.all_special_tokens):
					w = w.lower()
				
				for t in tokenizer.all_special_tokens:
					if t in w:
						w = w.split(t)
						# this is needed in case we have multiple special tokens in the same word
						w = [part.lower() for part in w if not part in tokenizer.all_special_tokens]
						w = t.join(w)
					
				special_words.append(w)
			
			words = special_words
		else:
			words = [w.lower() for w in words]
	
	aligned = []
	while tokens:
		aligned_tokens = [tokens.pop()]
		# some tokenizers apparently have tokens that evaluate
		# to an empty string. that won't show up in the word list,
		# of course (they only appear during open-ended generation).
		# Let's never consider these to be part of a word, so we'll 
		# just continue on and grab the next word in the next iteration
		if tokenizer.decode(aligned_tokens[-1]) == '':
			aligned.append(aligned_tokens)
			continue
		
		word = words.pop()
		
		# we need to replace all spaces here rather than
		# just stripping because some tokenizers don't handle
		# words with punctuation in the middle correctly
		# e.g, 'bert-large-cased' tokenizes 're-wrapped' as
		# [1231, 118, 4293], but decodes that sequence as
		# 're - wrapped', with spaces in the middle.
		if 'babyt5' in tokenizer.name_or_path:
			# babyt5 doesn't tokenize commas correctly, but
			# as its <unk> token. in general, an <unk> token
			# should not be used to identify a word, since
			# not all <unk> tokens have the same source.
			# in this case, we build in a very specific hack.
			# we don't want a more general solution, since
			# that could mask an actually problematic case
			while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != re.sub('[0-9,]', tokenizer.unk_token, word):
				aligned_tokens += [tokens.pop()]
		else:
			while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != word:
				aligned_tokens += [tokens.pop()]
		
		aligned.append(aligned_tokens)
	
	assert len([l for l in aligned if tokenizer.decode(l) != '']) == num_words, (
		f'Unable to find {num_words} words in text.'
	)
	
	return aligned

def finetune_model(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	train_dataset: Dataset,
	validation_dataset: Dataset,
	model_args: ModelArguments,
	data_args: DataTrainingArguments,
	trial: optuna.trial.Trial = None,
) -> float:
	'''
	Finetunes a language model.
	
	params:
	model (AutoModelForCasualLM): the model to finetune.
	tokenizer (AutoTokenizer): the tokenizer for the model.
	train_dataset (Dataset): the training dataset.
	validation_dataset (Dataset): the validation dataset.
	data_args (DataTrainingArguments): additional DataTrainingArguments.
									   see DataTrainingArguments for details.
	
	returns:
		float: the best loss on the validation dataset
	'''
	def compute_loss(
		outputs: ModelOutput,
		labels: torch.Tensor,
		losses_to_compute: list = [loss_classes.OutputsDefaultLoss()],
		loss_reduction_fn: Callable = sum,
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
		losses = [loss(outputs, labels) for loss in losses_to_compute]
		loss_names = [loss.__class__.__name__ for loss in losses_to_compute]
		
		# set the individual losses as attributes
		# on the outputs so we can have access to them later.
		outputs['losses'] = {}
		for loss, loss_name in zip(losses, loss_names):
			outputs.losses[loss_name] = loss
		
		losses_reduced = loss_reduction_fn(torch.stack(losses))
		outputs.loss = losses_reduced
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
		
		# so we don't accidentally modify anything in place
		metrics = deepcopy(metrics)
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
					original_epoch=lambda df: df['epoch'],
					epoch=lambda df: df['epoch'] + df['batch_number_prop']
				)
		)
		
		sns.set_theme(rc={'figure.figsize': (8.5*(16/9), 8.5)})
		plot_title = f'{metrics["model_name"].unique()[0]} ({metrics["n_params"].unique()[0]}) loss curves'
		
		# if we have the kl_loss, subtract it from the regular loss
		# for training to get the loss on just the training set
		metrics['loss_type'] = metrics.dataset_type.copy()
		for c in [col for col in metrics.columns if col.endswith('Loss')]:
			loss_col = metrics.dropna(subset=[c]).reset_index(drop=True)
			# if there's only one loss value for a dataset type, then we don't 
			# need to plot the individual losses separately.
			for dataset in loss_col.dataset_type.unique():
				# this is true if there's only one column for the dataset
				# we're checking that has non-NA loss values. In that case,
				# we don't need to plot the loss twice redundantly (since we'll
				# already be plotting it as the overall loss), so we remove
				# this dataset from the losses
				if len(loss_col[loss_col.dataset_type == dataset][['dataset_type'] + [c for c in loss_col.columns if c.endswith('Loss')]].dropna(axis=1).columns) == 2:
					loss_col = loss_col[loss_col.dataset_type != dataset].copy()
			
			# if there's anything left after we've removed the identical
			# losses to the overall loss, add them so they'll be included
			# separately in the plot.
			if len(loss_col) != 0:
				loss_col.loss = loss_col[c].copy()
				loss_col.loss_type = loss_col.dataset_type + f' ({c})'
				metrics = pd.concat([metrics, loss_col]).reset_index(drop=True)
		
		metrics.dataset_type = metrics.loss_type
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
	
	losses = {}
	for dataset_type in data_args.loss_classes:
		losses[dataset_type] = []
		for klass in data_args.loss_classes[dataset_type]:
			# we load these from the data_args directly rather than the kwargs
			# within it to make it easier to (since that way we don't have to
			# have optuna set values nested inside other things, but can instead
			# just suggest a value directly on any attr of the data_args.
			kwargs = {
				key: getattr(data_args, f'{dataset_type}_{klass.__name__}_{key}')
				for key in data_args.loss_classes_kwargs.get(dataset_type, {}).get(klass.__name__, {})
			}
			losses[dataset_type].append(klass(model=model, tokenizer=tokenizer, **kwargs))
	
	pre_train_step_callbacks = model_args.model_pre_train_step_callbacks
	if pre_train_step_callbacks and not isinstance(pre_train_step_callbacks, list):
		pre_train_step_callbacks = [pre_train_step_callbacks]
	
	if pre_train_step_callbacks:
		pre_train_step_callbacks = [
			callback(
				model=model, tokenizer=tokenizer,
			 	**model_args.model_pre_train_step_callbacks_kwargs.get(callback.__name__, {})
			)
			for callback in pre_train_step_callbacks
		]
	
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
	
	# this makes optimization easier, since optuna can more easily set the stuff
	# not nested. So, we unpack them here and replace the values in the dict
	# with the values set to the dataclass. It's ugly since we're duplicating the 
	# info in multiple places, but it works.
	optimizer_kwargs = {k: getattr(data_args, k) for k in data_args.train_optimizer_kwargs}
	optimizer = data_args.train_optimizer(params=model.parameters(), **optimizer_kwargs)
	
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
					train_dataset.dataset,
					batch_size=data_args.per_device_train_batch_size,
					collate_fn=data_preprocessing.pad_batch
				),
				validation=DataLoader(
					validation_dataset.dataset,
					batch_size=data_args.per_device_validation_batch_size,
					collate_fn=data_preprocessing.pad_batch
				),
			)
			
			# pass through the training set
			n_observed_examples = 0
			epoch_train_losses = []
			each_train_losses = {}
			
			for i, inputs in enumerate(dataloader['train']):
				n_examples_in_batch = inputs['input_ids'].shape[0]
				
				# use this as a unique input identifier
				input_nums = list(range(n_observed_examples, n_observed_examples + n_examples_in_batch))
				n_observed_examples += n_examples_in_batch
				
				input_texts = train_dataset.texts[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				input_labels = train_dataset.labels[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				batch_metadata = train_dataset.metadata[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				
				if data_args.data_preprocessing_fn_strategy.get('train') == 'per_batch':
					inputs = data_args.data_preprocessing_fn.get('train', data_preprocessing.identity)(
						inputs=inputs,
						tokenizer=tokenizer,
						**data_args.data_preprocessing_fn_kwargs.get('train', {}),
					)
					
					if data_args.data_preprocessing_fn.get('train') in data_preprocessing.UPDATE_LABELS_FNS:
						# update the labels if needed. This is for span denoising objectives.
						input_labels = [tokenizer.decode(label[label != -100]) for label in input_labels]
					
					if 'expanded_length' in inputs:
						input_nums, input_texts, input_labels, batch_metadata = data_preprocessing._expand_rows(
							input_nums,
							input_texts,
							input_labels,
							batch_metadata,
							expanded_lengths=inputs['expanded_length'],
						)
						del inputs['expanded_length']
				
				inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
				
				model.train()
				optimizer.zero_grad(set_to_none=True) # this is supposed to be faster than .zero_grad()
				outputs = model(**inputs)
				loss = compute_loss(
					outputs=outputs, 
					labels=inputs['labels'],
					losses_to_compute=losses.get('train', [loss_classes.OutputsDefaultLoss()]),
					loss_reduction_fn=data_args.loss_reduction_fns.get('train', torch.sum),
				)
				loss.backward()
				for callback in pre_train_step_callbacks:
					callback(epoch=epoch, batch=i)
				
				optimizer.step()
				model.eval()
				
				if trial is None:
					addl_kwargs = dict(
						dataset_name=re.sub(
							r'\.(txt|json)\.gz$', 
							'', 
							os.path.split(data_args.train_file)[-1]
						)
					)
					
					# add individual losses to kwargs here.
					for k in outputs.losses:
						addl_kwargs[k] = outputs.losses[k].item()
					
					metrics.extend(
						evaluate_batch(
							model=model,
							tokenizer=tokenizer,
							inputs=inputs,
							input_texts=input_texts,
							input_labels=input_labels,
							input_nums=input_nums,
							batch_outputs=outputs,
							batch_metadata=batch_metadata,
							epoch=epoch,
							batch_number=i,
							dataset_type='train',
							loss=loss.item(),
							**addl_kwargs
						)
					)
				
				epoch_train_losses.append(outputs.loss.item())
				for k in outputs.losses:
					each_train_losses[k] = each_train_losses.get(k, [])
					each_train_losses[k].append(outputs.losses[k].item())
			
			# pass through the validation set
			n_observed_examples = 0
			epoch_validation_losses = []
			each_validation_losses = {}
			for i, inputs in enumerate(dataloader['validation']):
				n_examples_in_batch = inputs['input_ids'].shape[0]
				
				# use this as a unique input identifier
				input_nums = list(range(n_observed_examples, n_observed_examples + n_examples_in_batch))
				n_observed_examples += n_examples_in_batch
				
				input_texts = validation_dataset.texts[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				input_labels = validation_dataset.labels[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				batch_metadata = validation_dataset.metadata[(n_observed_examples - n_examples_in_batch):n_observed_examples]
				
				if data_args.data_preprocessing_fn_strategy.get('validation') == 'per_batch':
					inputs = data_args.data_preprocessing_fn.get('validation', data_preprocessing.identity)(
						inputs=inputs,
						tokenizer=tokenizer,
						**data_args.data_preprocessing_fn_kwargs.get('validation', {}),
					)
					
					if data_args.data_preprocessing_fn.get('validation') in data_preprocessing.UPDATE_LABELS_FNS:
						# update the labels if needed. This is for span denoising objectives.
						input_labels = [tokenizer.decode(label[label != -100]) for label in input_labels]
					
					if 'expanded_length' in inputs:
						input_nums, input_texts, input_labels, batch_metadata = data_preprocessing._expand_rows(
							input_nums,
							input_texts,
							input_labels,
							batch_metadata,
							expanded_lengths=inputs['expanded_length'],
						)
						del inputs['expanded_length']
				
				inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
				
				with torch.no_grad():
					outputs = model(**inputs)
				
				dev_loss = compute_loss(
					outputs=outputs, 
					labels=inputs['labels'],
					losses_to_compute=losses.get('validation', [loss_classes.OutputsDefaultLoss()]),
					loss_reduction_fn=data_args.loss_reduction_fns.get('validation', torch.sum),
				)
				
				if trial is None:
					addl_kwargs = dict(
						dataset_name=re.sub(
							r'\.(txt|json)\.gz$', '', os.path.split(data_args.validation_file)[-1]
						)
					)
					
					# add individual losses to kwargs here.
					for k in outputs.losses:
						addl_kwargs[k] = outputs.losses[k].item()
					
					metrics.extend(
						evaluate_batch(
							model=model,
							tokenizer=tokenizer,
							inputs=inputs,
							input_texts=input_texts,
							input_labels=input_labels,
							input_nums=input_nums,
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
				for k in outputs.losses:
					each_validation_losses[k] = each_validation_losses.get(k, [])
					each_validation_losses[k].append(outputs.losses[k].item())
			
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
				train_losses=str({k: float(f'{np.mean(v):.2f}') for k, v in each_train_losses.items()}), 
				dev_loss=f'{np.mean(epoch_validation_losses).item():.2f}', 
				dev_losses=str({k: float(f'{np.mean(v):.2f}') for k, v in each_validation_losses.items()}),
				pat=f'{data_args.patience - patience_counter}', 
			)
			
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
		
		# do this so that the model is in its best performing state
		# if we go on to use it later in the script
		logger.info('Loading model state with lowest dev loss')
		model.load_state_dict(best_model_state_dict)
		
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
	if get_model_task(model_name_or_path=model_name_or_path) == 'LM':
		return evaluate_LM_batch
	
	if get_model_task(model_name_or_path=model_name_or_path) == 'MLM':
		return evaluate_MLM_batch
	
	if get_model_task(model_name_or_path=model_name_or_path) == 'Seq2Seq':
		return evaluate_S2S_batch
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def tokenize_texts(tokenizer: AutoTokenizer, text: list[str]) -> list[list[int]]:
	'''
	Tokenize a list of examples without special tokens for use during evaluation.
	'''
	tokenized = tokenizer(text, add_special_tokens=False)['input_ids']
	return tokenized

def extract_surprisals(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	dataset: Dataset,
	model_args: ModelArguments,
	data_args: DataTrainingArguments,
	**additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a model on the test dataset.
	Saves results in data_args.output_dir as a csv.
	
	params:
		model (AutoModelForCausalLM)		: the model to evaluate.
		tokenizer (AutoTokenizer)			: the tokenizer for the model
		dataset (Dataset)					: the dataset to evaluate on
		metadata (list[dict])				: the metadata for each example
		input_texts (list[str])				: the dataset as input strings.
		model_args (ModelArguments)			: the ModelArguments.
		data_args (DataTrainingArguments)	: the arguments containing information about the data.
											  see the DataTrainingArguments class for more details.
		**additional_metadata				: additional metadata to add to each dictionary in the
											  output
	
	returns:
		list[dict]: a list of dictionaries containing surprisals for each token of each sentence
				    in the test dataset along with associated metadata.
	'''
	dataloader = DataLoader(
		dataset.dataset,
		batch_size=data_args.per_device_test_batch_size,
		collate_fn=data_preprocessing.pad_batch,
	)
	_ = model.eval()
	
	if data_args.save_tmp_test_files:
		os.makedirs(data_args.output_dir, exist_ok=True)
	
	n_observed_examples = 0
	metrics = []
	for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
		n_examples_in_batch = inputs['input_ids'].shape[0]
		
		# use this as a unique input identifier
		input_nums = list(range(n_observed_examples, n_observed_examples + n_examples_in_batch))
		n_observed_examples += n_examples_in_batch
		
		if data_args.save_tmp_test_files and os.path.isfile(f'tmpbatch{i}.json.gz'):
			with gzip.open(f'tmpbatch{i}.json.gz', 'rt', encoding='utf8') as in_file:
				metrics.extend([json.loads(l.strip()) for l in in_file.readlines()])
			
			continue
		
		batch_texts = dataset.texts[(n_observed_examples - n_examples_in_batch):n_observed_examples]
		batch_labels = dataset.labels[(n_observed_examples - n_examples_in_batch):n_observed_examples]
		batch_metadata = dataset.metadata[(n_observed_examples - n_examples_in_batch):n_observed_examples]
		
		if data_args.data_preprocessing_fn_strategy.get('test') == 'per_batch':
			inputs = data_args.data_preprocessing_fn.get('test', data_preprocessing.identity)(
				inputs=inputs,
				tokenizer=tokenizer,
				**data_args.data_preprocessing_fn_kwargs.get('test', {}),
			)
			
			if data_args.data_preprocessing_fn.get('test') in data_preprocessing.UPDATE_LABELS_FNS:
				# update the labels if needed. This is for span denoising objectives.
				input_labels = [tokenizer.decode(label[label != -100]) for label in input_labels]
			
			if 'expanded_length' in inputs:
				input_nums, input_texts, input_labels, batch_metadata = data_preprocessing._expand_rows(
					input_nums,
					input_texts,
					input_labels,
					batch_metadata,
					expanded_lengths=inputs['expanded_length'],
				)
				del inputs['expanded_length']
		
		inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
		
		with torch.no_grad():
			batch_outputs = model(**inputs)
		
		results = evaluate_batch(
			model=model,
			tokenizer=tokenizer,
			inputs=inputs,
			input_texts=batch_texts,
			input_labels=batch_labels,
			input_nums=input_nums,
			batch_outputs=batch_outputs,
			batch_metadata=batch_metadata,
			**additional_metadata,
		)
		
		if data_args.save_tmp_test_files:
			with gzip.open(f'tmpbatch{i}.json.gz', 'wt', encoding='utf8') as out_file:
				for i, result in enumerate(results):
					_ = out_file.write(json.dumps(result))
					if i != len(results) - 1:
						_ = out_file.write('\n')
		
		metrics.extend(results)
	
	if data_args.save_tmp_test_files:
		for file in glob('tmpbatch*.json.gz'):
			os.remove(file)
	
	return metrics

def evaluate_batch(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	batch_outputs: ModelOutput,
	input_nums: list[int] = None,
	batch_metadata: list[dict] = None,
	**additional_metadata
) -> list[dict]:
	'''Record metrics for a single batch of inputs, depending on the model type.'''
	if input_nums is None:
		input_nums = range(len(inputs.get('labels', inputs['input_ids']).shape[0]))
	
	if batch_metadata is None:
		batch_metadata = {}
	
	model_eval_function = get_model_eval_function(model_name_or_path=model.name_or_path)
	
	return model_eval_function(
		model=model, 
		tokenizer=tokenizer, 
		inputs=inputs, 
		input_texts=input_texts,
		input_labels=input_labels,
		input_nums=input_nums, 
		batch_outputs=batch_outputs,
		batch_metadata=batch_metadata,
		additional_metadata=additional_metadata,
	)

def evaluate_LM_batch(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
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
	records = zip(
		input_nums, input_texts, inputs['input_ids'], input_labels, 
		next_word_ids, batch_surprisals, batch_metadata,
		strict=True
	)
	for (
		input_num, input_text, input_ids, input_label, 
		next_word_tokens, surprisal, example_metadata
	) in records:
		# get this here in case the original text that was passed
		# was modified by preprocessing (i.e., lower-casing, masking, etc.)
		actual_input_text = [
			t for t in input_ids 
			if tokenizer.convert_ids_to_tokens(t.item()) not in tokenizer.all_special_tokens
		]
		actual_input_text = tokenizer.decode(actual_input_text)
		
		input_words = input_text.split()
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer, 
			words=input_words, 
			tokens=next_word_tokens
		)
		
		tokens_seen = 0
		for word_num, tokens in enumerate(aligned_tokens):
			for token_num, token in enumerate(tokens):
				predicted_token_id = torch.argmin(surprisal[tokens_seen,:], dim=-1).item()
				predicted_token = tokenizer.decode(predicted_token_id)
				metrics.extend([{
					'item': input_num,
					'original_text': input_text,
					'input_text': actual_input_text,
					'input_label': input_label,
					'word_num': word_num,
					'token_num_in_word': token_num,
					'token': tokenizer.decode(token),
					'token_id': token,
					'token_is_start_of_word': token_num == 0,
					'token_is_word': len(tokens) == 1,
					'surprisal': surprisal[tokens_seen,token].item(),
					'predicted_token': predicted_token,
					'predicted_token_id': predicted_token_id,
					'predicted_token_surprisal': surprisal[tokens_seen,predicted_token_id].item(),
					**example_metadata,
					**additional_metadata,
				}])
				tokens_seen += 1
	
	return metrics

def evaluate_MLM_batch(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	input_nums: list[int],
	batch_outputs: 'MaskedLMOutput',
	batch_metadata: list[dict],
	additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a batch of examples for a Masked Language Model.
	For each input and label, for each mask position in the input, 
	determines the surprisal of the token at that position in the label.
	'''
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs.logits, dim=-1)
	
	# these are the positions we want to get predictions for
	mask_locations = torch.nonzero(inputs['input_ids'] == tokenizer.mask_token_id, as_tuple=True)
	batch_surprisals = batch_surprisals[mask_locations]
	
	# we need to repeat the text, num, and metadata associated with each input
	# for each time a mask token occurs in that input mask_locations[0] does 
	# this, since it repeats the example number for each mask token in it.
	# this will also have the effect of excluding any sentences with no masks,
	# which is what we want.
	input_texts = [input_texts[ex_idx] for ex_idx in mask_locations[0]]
	input_labels = [input_labels[ex_idx] for ex_idx in mask_locations[0]]
	input_nums = [input_nums[ex_idx] for ex_idx in mask_locations[0]]
	batch_metadata = [batch_metadata[ex_idx] for ex_idx in mask_locations[0]]
	
	# we want to record whether this mask token is at the start of a word or not,
	# so we need to tokenize the whole sentence, and align the tokens to determine
	# this.
	word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	label_word_ids = tokenize_texts(tokenizer=tokenizer, text=input_labels)
	# this will help us align the words to the subword tokens
	# we ignore the mask token since it's the one we're getting
	# predictions for.
	special_token_ids = [
		t for t in tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens) 
		if t != tokenizer.mask_token_id
	]
	# for each input (for each mask token), tell us where to start looking for 
	# "real" (i.e., non-special) tokens.
	input_ids = [inputs['input_ids'][ex_idx] for ex_idx in mask_locations[0]]
	starting_ids = [[t not in special_token_ids for t in row].index(True) for row in input_ids]
	labels = [inputs['labels'][ex_idx] for ex_idx in mask_locations[0]]
	
	metrics = []
	records = zip(
		input_nums, input_texts, input_labels, labels, starting_ids, inputs['input_ids'],
		mask_locations[-1], word_ids, label_word_ids, batch_surprisals, batch_metadata,
		strict=True
	)
	for (
		input_num, input_text, input_label, label_ids, starting_id, input_ids,
		mask_location, word_tokens, label_tokens, surprisal, example_metadata
	) in records:
		# get the actual text the model was run on, in case it was
		# changed in preprocessing.
		actual_input_text = [
			t for t in input_ids 
			if tokenizer.convert_ids_to_tokens(t.item()) not in tokenizer.all_special_tokens or 
			t == tokenizer.mask_token_id
		]
		actual_input_text = tokenizer.decode(actual_input_text)
	
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer,
			words=input_text.split(),
			tokens=word_tokens,
		)
		aligned_label_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer,
			words=input_label.split(),
			tokens=label_tokens,
		)
		
		# make sure that for MLM, the number of tokens is the same
		# in the inputs and the labels, since this is required
		if sum(len(l) for l in aligned_tokens) != sum(len(l) for l in aligned_label_tokens):
			raise ValueError(
				'The mask token in at least one example is associated with more than one token '
				f'in the label: {input_text=} (n_tokens={sum(len(l) for l in aligned_tokens)}) '
				f'and {input_label=} (n_tokens={sum(len(l) for l in aligned_label_tokens)})!'
			)
		
		# we need to extract the aligned token set that has the token_num in it,
		# so we can determine whether it starts a word or not. Then, we need to get
		# the token at the corresponding position in the labels.
		
		# this tells us which token number starts each word
		starting_token_numbers = [
			sum(len(t) for t in aligned_tokens[:i]) for i, _ in enumerate(aligned_tokens)
		]
		
		# the aligned tokens are indexed without the special tokens,
		# so we need to adjust to find it in the starting_token_numbers,
		# by subtracting out the starting position of the non-special tokens
		starting_token_num_of_word = mask_location - starting_id
		# if the token doesn't start a word, we need to move backward till
		# we find it, so we know which token in the middle of the word
		# we want to get the surprisal for (the token at the actual
		# mask position in the word)
		while starting_token_num_of_word not in starting_token_numbers:
			starting_token_num_of_word -= 1
		
		# get the tokens corresponding to the word containing this mask token
		tokens = aligned_label_tokens[starting_token_numbers.index(starting_token_num_of_word)]
		
		# get the index of the mask token we're looking at in this word
		# subtract one since Python indexes start at 0, and we want to
		# use this as an index
		token_num_in_word = mask_location - starting_token_num_of_word - 1
		
		# get the token that we want the prediction for in the [MASK] location
		token = tokens[token_num_in_word]
		
		# get the number of the word that contains this mask token
		word_num = starting_token_numbers.index(starting_token_num_of_word.item())
		
		# get the predicted token and id, in case it's not the one
		# that's the actual target
		predicted_token_id = torch.argmin(surprisal, dim=-1).item()
		predicted_token = tokenizer.decode(predicted_token_id)
		
		metrics.append({
			'item': input_num,
			'original_text': input_text,
			'input_text': actual_input_text,
			'input_label': input_label,
			'word_num': word_num,
			'token_num_in_word': token_num_in_word.item(),
			'token': tokenizer.decode(token),
			'token_id': token,
			'token_is_start_of_word': (token_num_in_word == 0).item(),
			'token_is_word': len(tokens) == 1,
			'surprisal': surprisal[token].item(),
			'predicted_token': predicted_token,
			'predicted_token_id': predicted_token_id,
			'predicted_token_surprisal': surprisal[predicted_token_id].item(),
			**example_metadata,
			**additional_metadata,
		})
	
	return metrics

def evaluate_S2S_batch(
	model: AutoModelForSeq2SeqLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	input_nums: list[int],
	batch_outputs: 'Seq2SeqLMOutput',
	batch_metadata: list[dict],
	additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a batch of examples for a Seq2Seq Language Model.
	For each input, determines the surprisal of each token of the
	generated sequence, as well as the surprisal of each token
	of the label corresponding to the input (using teacher-forcing).
	'''
	def fix_aligned_mask_span_tokens(
		aligned_tokens: list[list[int]], 
		single_word_tokens: list[int]
	) -> list[list[int]]:
		'''
		Ensures that each single word token is a single word.
		All other tokens will retain their relative positions
		in the aligned tokens list.
		'''
		fixed_aligned_tokens = [[]]
		for i, l in enumerate(aligned_tokens):
			for j, t in enumerate(l):
				if t in single_word_tokens:
					fixed_aligned_tokens.append([t])
					fixed_aligned_tokens.append([])
				else:
					fixed_aligned_tokens[-1].append(t)
		
		fixed_aligned_tokens = [l for l in fixed_aligned_tokens if l]
		return fixed_aligned_tokens
	
	def prefix_allowed_tokens_fn_factory(
		# we don't actually need to pass these, but we
		# need them here so that the inner function
		# can access them.
		labels: torch.Tensor = inputs['labels'], 
		pad_token_id: int = tokenizer.pad_token_id
	) -> Callable[[int, torch.Tensor], list[int]]:
		'''
		Returns a function that constrains the output generation to the label sequence.
		If we don't wrap this, then the inner function doesn't have access to the labels.
		'''
		def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
			'''
			Forces prediction of the label sequence corresponding to the
			batch we're evaluating.
			'''
			next_token = labels[batch_id][len(input_ids)].item()
			# this means we're ignoring this token for loss, so just force it
			# to be the pad token. We do this since the tokenizer won't be able
			# to decode -100.
			if next_token == -100:
				next_token = pad_token_id
			
			return [next_token]
		
		return prefix_allowed_tokens_fn
		
	# we have the batch_outputs corresponding to the generated sequence
	# already. But now we should get the outputs corresponding to the 
	# conditional probability of each token in the label sequence,
	# given the preceding tokens in that label. we do this with the
	# prefx_allow_tokens_fn and output_logits.
	with torch.no_grad():
		label_outputs = model.generate(
			inputs=inputs['input_ids'],
			prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_factory(),
			return_dict_in_generate=True,
			# allow generating only as many tokens as we need
			# to get the surprisals for the label sequence
			max_new_tokens=inputs['labels'].shape[-1] - 1,
			output_logits=True,
		)
		# transform these to be the same shape as the normal outputs
		# these will be one shorter than the batch_output logits,
		# since these are forced to start with a [0], so we don't
		# have the logits for that token.
		label_outputs.logits = torch.stack(label_outputs.logits, dim=1)
	
	surprisals = {
		'generated_sequence': batch_outputs.logits,
		'label_sequence': label_outputs.logits,
	}
	surprisals = {
		k: -(1/torch.log(torch.tensor(2.))) * F.log_softmax(logits, dim=-1) 
		for k, logits in surprisals.items()
	}
	
	metrics = []
	for generated_sequence_type, batch_surprisals in surprisals.items():
		if generated_sequence_type == 'generated_sequence':
			label_word_ids = torch.argmin(batch_surprisals, dim=-1).tolist()
			label_texts = tokenizer.batch_decode(torch.argmin(batch_surprisals, dim=-1))
		else:
			label_word_ids = inputs['labels'].detach().clone().tolist()
			
			# we need to remove the first value from this, since we don't
			# have predictions for the pad tokens for these ones
			label_word_ids = [l[1:] if l[0] == tokenizer.pad_token_id else l for l in label_word_ids]
			
			# remove this since the tokenizer won't be able to decode the 
			# "ignore_for_loss" value in the labels.
			label_word_ids = [[t for t in l if t != -100] for l in label_word_ids]
			label_texts = [re.sub(fr'^{tokenizer.pad_token}', '', l) for l in input_labels]
		
		records = zip(
			input_nums, input_texts, inputs['input_ids'], label_texts, 
			label_word_ids, batch_surprisals, batch_metadata,
			strict=True
		)
		for (
			input_num, input_text, input_ids, label_text, 
			label_word_ids, surprisal, example_metadata
		) in records:
			actual_input_text = [
				t for t in input_ids 
				if tokenizer.convert_ids_to_tokens(t.item()) not in tokenizer.all_special_tokens or 
				t in tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
			]
			actual_input_text = tokenizer.decode(actual_input_text)
			
			aligned_tokens = align_words_to_subword_tokens(
				tokenizer=tokenizer,
				words=label_text.split(),
				tokens=label_word_ids,
			)
			
			# for the mask span tokens, we need to fix this up here,
			# since we don't want to count them as word parts in the 
			# labels (even though they could be in the inputs).
			aligned_tokens = fix_aligned_mask_span_tokens(
				aligned_tokens=aligned_tokens, 
				single_word_tokens=tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens),
			)
			
			tokens_seen = 0
			for word_num, tokens in enumerate(aligned_tokens):
				for token_num, token in enumerate(tokens):
					predicted_token_id = torch.argmin(surprisal[tokens_seen,:], dim=-1).item()
					predicted_token = tokenizer.decode(predicted_token_id)
					metrics.append({
						'item': input_num,
						'original_text': input_text,
						'input_text': actual_input_text,
						'generated_sequence_type': generated_sequence_type,
						'label_text': label_text,
						'word_num': word_num,
						'token_num_in_word': token_num,
						'token': tokenizer.decode(token),
						'token_id': token,
						'token_is_start_of_word': token_num == 0,
						'token_is_word': len(tokens) == 1,
						'surprisal': surprisal[tokens_seen,token].item(),
						'predicted_token': predicted_token,
						'predicted_token_id': predicted_token_id,
						'predicted_token_surprisal': surprisal[tokens_seen,predicted_token_id].item(),
						**example_metadata,
						**additional_metadata,
					})
					tokens_seen += 1
	
	return metrics

def finetune_lm(
	model_args: ModelArguments, 
	data_args: DataTrainingArguments, 
	optim_args: OptimizationArguments,
	trial: Optional[optuna.trial.Trial] = None
) -> float:
	'''Main function.'''
	if optim_args.do_optimize:
		optim_args.set_suggested_values(data_args=data_args, trial=trial)
	else:
		logger.info(f'Setting seed to {data_args.seed}.')
		set_seed(data_args.seed)
	
	if data_args.output_dir is None:
		data_args._set_output_dir(model_name=model_args.model_name_or_path)
	
	logger.info(f'Model parameters: {model_args}')
	logger.info(f'Data parameters: {data_args}')
	
	tokenizer, model = load_tokenizer_and_model(model_args)
	
	# if we don't provide a train file, this lets us just evaluate the model
	if data_args.train_file:
		train_dataset = Dataset(
			file=data_args.train_file,
			tokenizer=tokenizer,
			split_name='train',
			max_samples=data_args.max_train_samples,
			max_length=data_args.max_length,
			preprocessing_num_workers=data_args.preprocessing_num_workers,
			overwrite_cache=data_args.overwrite_cache,
			data_preprocessing_fn=data_args.data_preprocessing_fn.get('train', data_preprocessing.identity),
			data_preprocessing_fn_kwargs=data_args.data_preprocessing_fn_kwargs.get('train', {}),
			data_preprocessing_fn_strategy=data_args.data_preprocessing_fn_strategy.get('train', 'once'),
		)
		
		validation_dataset = Dataset(
			file=data_args.validation_file,
			tokenizer=tokenizer,
			split_name='validation',
			max_samples=data_args.max_validation_samples,
			max_length=data_args.max_length,
			preprocessing_num_workers=data_args.preprocessing_num_workers,
			overwrite_cache=data_args.overwrite_cache,
			data_preprocessing_fn=data_args.data_preprocessing_fn.get('validation', data_preprocessing.identity),
			data_preprocessing_fn_kwargs=data_args.data_preprocessing_fn_kwargs.get('validation', {}),
			data_preprocessing_fn_strategy=data_args.data_preprocessing_fn_strategy.get('validation', 'once'),
		)
		
		best_dev_loss = finetune_model(
			model=model, 
			tokenizer=tokenizer,
			train_dataset=train_dataset,
			validation_dataset=validation_dataset,
			model_args=model_args,
			data_args=data_args,
			trial=trial,
		)
	
	# only run the test if we're not optimizing, since we shouldn't optimize
	# on the test datasets anyway
	if trial is None and data_args.test_file:
		# if we only have one dataset, put it in a list
		# so the loop below works
		if isinstance(data_args.test_file, str):
			data_args.test_file = [data_args.test_file]
		
		test_results = []
		logger.info('Beginning testing...')
		for test_file in data_args.test_file:
			test_dataset = Dataset(
				file=test_file,
				tokenizer=tokenizer,
				split_name='test',
				max_samples=data_args.max_test_samples,
				max_length=data_args.max_length,
				preprocessing_num_workers=data_args.preprocessing_num_workers,
				overwrite_cache=data_args.overwrite_cache,
				data_preprocessing_fn=data_args.data_preprocessing_fn.get('test', data_preprocessing.identity),
				data_preprocessing_fn_kwargs=data_args.data_preprocessing_fn_kwargs.get('test', {}),
				data_preprocessing_fn_strategy=data_args.data_preprocessing_fn_strategy.get('test', 'once'),
			)
			
			test_results.extend(
				extract_surprisals(
					model=model,
					tokenizer=tokenizer,
					dataset=test_dataset,
					model_args=model_args,
					data_args=data_args,
					**dict(
						dataset_name=re.sub(r'\.(txt|json)\.gz$', '', os.path.split(test_file)[-1])
					)
				)
			)
		
		test_results = pd.DataFrame(test_results)
		test_results = test_results.assign(
			model_name=data_args.output_dir,
			task=get_model_task(model.config.name_or_path),
			n_params=f'{round(model.num_parameters()/1000000)}M',
		)
		move_to_beginning = ['model_name', 'task', 'n_params', 'dataset_name']
		test_results = test_results[
			move_to_beginning + 
			[c for c in test_results.columns if not c in move_to_beginning]
		]
		if data_args.test_output_file_prefix is None:
			basename = re.sub(r'[\\/]', '-', model.name_or_path)
		else:
			basename = data_args.test_output_file_prefix
		
		# this happens if we're just running test files on
		# a huggingface model directly, and not on one we've
		# fine-tuned
		if not os.path.isdir(data_args.output_dir):
			os.makedirs(data_args.output_dir, exist_ok=True)
		
		test_results.to_csv(
			os.path.join(
				data_args.output_dir,
				f'{basename}-test_results.csv.gz'
			),
			index=False,
			na_rep='NA',
		)
	
	logger.info('All tasks complete.')
	if data_args.train_file:
		return best_dev_loss

def optimize_finetune_lm(
	model_args: ModelArguments, 
	data_args: DataTrainingArguments, 
	optim_args: OptimizationArguments,
) -> optuna.study.Study:
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
		
		# save plots
		# find all the available visualization functions in optuna currently.
		# we'll try each of them and then only keep the ones that don't error
		# due to the current study structure.
		viz_functions = [(k, v) for k, v in vars(optuna.visualization).items() if isinstance(v, types.FunctionType)]
		# sort to keep the order consistent
		viz_functions = sorted(viz_functions, key=lambda t: t[0])
		viz_functions = [v for _, v in viz_functions]
		
		# if we don't do this, we get a ton of log messages we don't want
		logging.getLogger('kaleido').setLevel(logging.WARNING)
		logging.getLogger('choreographer').setLevel(logging.WARNING)
		
		# plotly can only save a single file to a pdf. So we'll save them in a
		# temporary directory, and then merge them into the output dir.
		with tempfile.TemporaryDirectory(dir=data_args.output_dir) as tempdir:
			for i, viz_function in enumerate(viz_functions):
				try:
					fig = viz_function(study)
					fig.write_image(os.path.join(tempdir, f'{i:0{len(viz_functions)}d}.pdf'))
				except Exception:
					pass
			
			files = glob(os.path.join(tempdir, '*.pdf'))
			if files:
				with PdfMerger() as merger:
					for file in files:
						merger.append(file)
					
					# if we're running multiple processes, we don't want them stepping on each other's
					# toes here, so handle the potential file exists issue gracefully
					try:
						merger.write(os.path.join(data_args.output_dir, 'optimization_plots.pdf'))
					except Exception:
						pass
	
	if not optim_args.do_optimize:
		logger.warning(
			'`optimize_finetune_lm` was called, but `optim_args.do_optimize` was set to False. '
			'`finetune_lm` will be run instead.'
		)
		return finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
	
	logger.info(f'Setting seed to {data_args.seed}.')
	set_seed(data_args.seed)
	logger.info(f'Optimization parameters: {optim_args.original_repr}')
	if data_args.output_dir is None:
		data_args._set_output_dir(model_name=model_args.model_name_or_path)
	
	trials = optim_args.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
	n_complete = len(trials)
	# if we're not limiting the max_trials or if we haven't reached that many trials yet,
	# run the study
	if optim_args.max_trials is None or n_complete < optim_args.max_trials:
		# add the callback if we've set the max trials
		# if we haven't, we'll just run the full number
		# of trials set
		if optim_args.max_trials is not None:
			optim_args.optimize_kwargs['callbacks'] = (
				optim_args.optimize_kwargs.get('callbacks', []) + 
				[optuna.study.MaxTrialsCallback(n_trials=optim_args.max_trials, states=None)]
			)
		
		optim_args.study.optimize(
			# we need to wrap finetune_lm in this function since optuna
			# doesn't support passing additional arguments to the optimized function,
			# and it will only pass the trial object as a positional argument in the
			# first slot. In general, we don't want to require using only kwargs in
			# in the `finetune_lm` function when we're not running a trial, so this is 
			# how we get around that
			lambda trial: finetune_lm(
				model_args=model_args, data_args=data_args, optim_args=optim_args, trial=trial
			), 
			**optim_args.optimize_kwargs
		)
	
	logger.info(f'Best parameters: {optim_args.study.best_params}')
	logger.info(f'Best result: {optim_args.study.best_value}')
	save_optimization_results(study=optim_args.study)
	
	logger.info('Optimization study complete.')
	return optim_args.study

if __name__ == '__main__':
	model_args, data_args, optim_args = parse_arguments(ModelArguments, DataTrainingArguments, OptimizationArguments)
	if not optim_args.do_optimize:
		_ = finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
	else:
		_ = optimize_finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
