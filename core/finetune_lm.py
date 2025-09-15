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
import gzip
import json
import types
import torch
import random
import optuna
import tempfile
import loss_classes
import transformers
transformers.utils.logging.set_verbosity_error()

import data_preprocessing

import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from tqdm import tqdm, trange
from PyPDF2 import PdfMerger
from typing import *
from dataset import Dataset
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
	from data_evaluation import get_model_task
	from model_arguments import ModelArguments
	from optimization_arguments import OptimizationArguments
	from data_training_arguments import DataTrainingArguments
else:
	from .parser import parse_args_into_dataclasses
	from .constants import *
	from .data_evaluation import get_model_task
	from .model_arguments import ModelArguments
	from .optimization_arguments import OptimizationArguments
	from .data_training_arguments import DataTrainingArguments

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
		use_auth_token=model_args.token,
		**model_args.config_kwargs
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
		**model_args.model_kwargs,
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

def finetune_model(
	model: Union[AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM],
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
	
	model_callbacks = {}
	for when_to_run_callbacks in model_args.model_callbacks:
		model_callbacks[when_to_run_callbacks] = model_args.model_callbacks[when_to_run_callbacks]
		if model_callbacks[when_to_run_callbacks] and not isinstance(model_callbacks[when_to_run_callbacks], list):
			model_callbacks[when_to_run_callbacks] = [model_args.model_callbacks[when_to_run_callbacks]]
		
		if model_callbacks[when_to_run_callbacks]:
			model_callbacks[when_to_run_callbacks] = [
				callback(
					model=model, tokenizer=tokenizer,
					**model_args.model_callbacks_kwargs.get(when_to_run_callbacks, {}).get(callback.__name__, {})
				)
				for callback in model_callbacks[when_to_run_callbacks]
			]
	
	for dataset_type in data_args.evaluation_fns:
		if not isinstance(data_args.evaluation_fns[dataset_type], list):
			data_args.evaluation_fns[dataset_type] = [data_args.evaluation_fns[dataset_type]]
	
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
			for callback in model_callbacks.get('begin_epoch', []):
				callback(epoch=epoch, batch=None)
			
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
				for callback in model_callbacks.get('pre_train_batch', []):
					callback(epoch=epoch, batch=i)
				
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
						model=model,
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
				
				# for gpt-bert. this is recorded in the results so we know what
				# mode the model was using for the batch whose results we're recording
				if get_model_task(model_name_or_path=model.name_or_path) == 'LM+MLM':
					model_mode = 'decoder' if model.config.is_decoder else 'encoder'
				
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
				
				for callback in model_callbacks.get('pre_train_step', []):
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
					
					if get_model_task(model_name_or_path=model.name_or_path) == 'LM+MLM':
						addl_kwargs.update(dict(model_mode=model_mode))
					
					for fn in data_args.evaluation_fns.get('train', []):
						metrics.extend(
							fn(
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
								**addl_kwargs,
								**data_args.evaluation_fns_kwargs.get('train', {}).get(fn.__name__, {})
							)
						)
				
				epoch_train_losses.append(outputs.loss.item())
				for k in outputs.losses:
					each_train_losses[k] = each_train_losses.get(k, [])
					each_train_losses[k].append(outputs.losses[k].item())
				
				for callback in model_callbacks.get('post_train_batch', []):
					callback(epoch=epoch, batch=i)
			
			for callback in model_callbacks.get('pre_validation', {}):
				callback(epoch=epoch, batch=i)
			
			# pass through the validation set
			n_observed_examples = 0
			epoch_validation_losses = []
			each_validation_losses = {}
			for i, inputs in enumerate(dataloader['validation']):
				for callback in model_callbacks.get('pre_validation_batch', {}):
					callback(epoch=epoch, batch=i)
				
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
						model=model,
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
				
				# for gpt-bert. this is recorded in the results so we know what
				# mode the model was using for the batch whose results we're recording
				if get_model_task(model_name_or_path=model.name_or_path) == 'LM+MLM':
					model_mode = 'decoder' if model.config.is_decoder else 'encoder'
				
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
					
					if get_model_task(model_name_or_path=model.name_or_path) == 'LM+MLM':
						addl_kwargs.update(dict(model_mode=model_mode))
					
					for fn in data_args.evaluation_fns.get('validation', []):
						metrics.extend(
							fn(
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
								**data_args.evaluation_fns_kwargs.get('validation', {}).get(fn.__name__, {})
							)
						)
				
				epoch_validation_losses.append(dev_loss.item())
				for k in outputs.losses:
					each_validation_losses[k] = each_validation_losses.get(k, [])
					each_validation_losses[k].append(outputs.losses[k].item())
				
				for callback in model_callbacks.get('post_validation_batch', []):
					callback(epoch=epoch, batch=i)
			
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
			
			for callback in model_callbacks.get('end_epoch', []):
				callback(epoch=epoch, batch=None)
			
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

def extract_surprisals(
	model: Union[AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM],
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
	
	model_callbacks = {}
	for when_to_run_callbacks in model_args.model_callbacks:
		model_callbacks[when_to_run_callbacks] = model_args.model_callbacks[when_to_run_callbacks]
		if model_callbacks[when_to_run_callbacks] and not isinstance(model_callbacks[when_to_run_callbacks], list):
			model_callbacks[when_to_run_callbacks] = [model_args.model_callbacks[when_to_run_callbacks]]
		
		if model_callbacks[when_to_run_callbacks]:
			model_callbacks[when_to_run_callbacks] = [
				callback(
					model=model, tokenizer=tokenizer,
					**model_args.model_callbacks_kwargs.get(when_to_run_callbacks, {}).get(callback.__name__, {})
				)
				for callback in model_callbacks[when_to_run_callbacks]
			]
	
	for dataset_type in data_args.evaluation_fns:
		if not isinstance(data_args.evaluation_fns[dataset_type], list):
			data_args.evaluation_fns[dataset_type] = [data_args.evaluation_fns[dataset_type]]
	
	n_observed_examples = 0
	metrics = []
	for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
		for callback in model_callbacks.get('pre_test_batch', []):
			callback(epoch=None, batch=i)
		
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
				model=model,
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
		
		if get_model_task(model_name_or_path=model.name_or_path) == 'LM+MLM':
			model_mode = 'decoder' if model.config.is_decoder else 'encoder'
			addl_kwargs = dict(model_mode=model_mode)
		else:
			addl_kwargs = {}
		
		with torch.no_grad():
			batch_outputs = model(**inputs)
		
		for fn in data_args.evaluation_fns.get('test', []):
			results = fn(
				model=model,
				tokenizer=tokenizer,
				inputs=inputs,
				input_texts=batch_texts,
				input_labels=batch_labels,
				input_nums=input_nums,
				batch_outputs=batch_outputs,
				batch_metadata=batch_metadata,
				dataset_type='test',
				**addl_kwargs,
				**additional_metadata,
				**data_args.evaluation_fns_kwargs.get('test', {}).get(fn.__name__, {}),
			)
		
			if data_args.save_tmp_test_files:
				with gzip.open(f'tmpbatch{i}.json.gz', 'at', encoding='utf8') as out_file:
					for i, result in enumerate(results):
						_ = out_file.write(json.dumps(result))
						_ = out_file.write('\n')
			
			metrics.extend(results)
		
		for callback in model_callbacks.get('post_test_batch', []):
			callback(epoch=None, batch=i)
	
	if data_args.save_tmp_test_files:
		for file in glob('tmpbatch*.json.gz'):
			os.remove(file)
	
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
			model=model,
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
			model=model,
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
				model=model,
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
	model_args, data_args, optim_args = parse_args_into_dataclasses(
		ModelArguments, DataTrainingArguments, OptimizationArguments
	)
	if not optim_args.do_optimize:
		_ = finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
	else:
		_ = optimize_finetune_lm(model_args=model_args, data_args=data_args, optim_args=optim_args)
