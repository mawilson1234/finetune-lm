import re
import gzip
import json
import torch
import logging
logger = logging.getLogger(__name__)

import datasets
datasets.logging.set_verbosity_error()

import data_preprocessing

from typing import Callable
from transformers import AutoTokenizer, AutoModel

def _expand_rows(
	original_texts: list[str],
	original_labels: list[str],
	original_metadata: list[dict],
	expanded_lengths: torch.tensor
) -> None:
	'''
	If we add additional rows to the dataset when processing it,
	here we expand the texts, labels, and metadata to match.
	'''
	# remove all the marker values, so that we have an array that
	# tells us for the text, label, and metadata and the corresponding
	# index, how many times to repeat it.
	reduced = expanded_lengths[expanded_lengths != -1]
	
	# ensure we have the same number of values in the reduced
	# list as we do examples pre-expansion
	if any(len(l) != len(reduced) for l in [texts, labels, metadata]):
		raise ExpansionError(
			'Expected there to be as many numbers indicating expanded example lengths '
			'as examples in the original dataset without expansion. But there are '
			f'{len(texts)} examples in the original inputs and {len(reduced)} '
			'values for new expanded example lengths!'
		)
	
	new_texts = []
	new_labels = []
	new_metadata = []
	for i, l in enumerate(reduced):
		new_texts.extend([texts[i] for _ in range(l)])
		new_labels.extend([labels[i] for _ in range(l)])
		new_metadata.extend([metadata[i] for _ in range(l)])
	
	# ensure that the new numbers match the new number of rows in the dataset
	if any(len(l) != len(expanded_lengths) for l in [texts, labels, metadata]):
		raise ExpansionError(
			'Expected there to be as many texts, labels, and metadata entries for '
			'expanded examples as there are expanded examples. But there are '
			f'{len(expanded_lengths)} examples in the expanded dataset and '
			f'{len(texts)}, {len(labels)}, {len(metadata)} values for the new '
			'texts, labels, and metadata!'
		)
	
	return new_texts, new_labels, new_metadata

class ExpansionError(Exception):
	pass

class Dataset:
	'''
	Class that extends a dataset by associating it with
	metadata in a way that doesn't get in the way of
	using it for training/inference.
	'''
	def __init__(
		self,
		file: str,
		model: AutoModel,
		tokenizer: AutoTokenizer,
		split_name: str,
		max_samples: int = None,
		max_length: int = None,
		preprocessing_num_workers: int = None,
		overwrite_cache: bool = False,
		data_preprocessing_fn: Callable = data_preprocessing.identity,
		data_preprocessing_fn_kwargs: dict = None,
		data_preprocessing_fn_strategy: str = 'once',
	) -> None:
		if data_preprocessing_fn_kwargs is None:
			data_preprocessing_fn_kwargs = {}
		
		self.file = file
		self.split_name = split_name
		self.data_format = re.findall(r'.*?\.(txt|json)\.gz$', self.file)[-1].replace('txt', 'text')
		self.unformatted_dataset = datasets.load_dataset(self.data_format, data_files={split_name: self.file})
		self.labels = self.unformatted_dataset[split_name]['text']
		if 'labels' in self.unformatted_dataset[split_name].features:
			self.labels = self.unformatted_dataset[split_name]['labels']
		
		self.texts = self.unformatted_dataset[split_name]['text']
		self.load_metadata()
		
		self.dataset = self.preprocess_dataset(
			model=model,
			tokenizer=tokenizer,
			split=split_name,
			max_samples=max_samples,
			max_length=max_length,
			preprocessing_num_workers=preprocessing_num_workers,
			overwrite_cache=overwrite_cache,
			data_preprocessing_fn=data_preprocessing_fn,
			data_preprocessing_fn_kwargs=data_preprocessing_fn_kwargs,
			data_preprocessing_fn_strategy=data_preprocessing_fn_strategy,
		)
		self.num_rows = len(self)
	
	def load_metadata(self) -> list[dict[str,str]]:
		try:
			with gzip.open(
				re.sub(r'\.(txt|json)\.gz$', '_metadata.json.gz', self.file), 'rt', encoding='utf-8'
			) as in_file:
				self.metadata = [json.loads(l) for l in in_file.readlines()]
		except FileNotFoundError:
			logger.warning(
				f'No metadata file found for {self.file}. Using empty metadata.'
			)
			with gzip.open(self.file, 'rt', encoding='utf-8') as in_file:
				n_lines = len(in_file.readlines())
			
			self.metadata = [{} for _ in range(n_lines)]
	
	def preprocess_dataset(
		self,
		model: AutoModel,
		tokenizer: AutoTokenizer,
		split: str = 'train',
		max_samples: int = None,
		max_length: int = None,
		preprocessing_num_workers: int = None,
		overwrite_cache: bool = False,
		data_preprocessing_fn: Callable = data_preprocessing.identity,
		data_preprocessing_fn_kwargs: dict = None,
		data_preprocessing_fn_strategy: str = 'once',
	) -> datasets.Dataset:
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
		if data_preprocessing_fn_kwargs is None:
			data_preprocessing_fn_kwargs = {}
		
		drop_cols = self.unformatted_dataset.get(split).column_names
		
		def preprocess_function(examples: list[str]) -> dict:
			'''Tokenizes a batch of string inputs.'''
			model_inputs = tokenizer(
				examples['text'],
				max_length=max_length,
				padding=True,
				truncation=True,
			)
			
			# tokenize the labels if we have them
			if 'labels' in examples:
				model_inputs['labels'] = tokenizer(
					examples['labels'],
					max_length=max_length,
					padding=True,
					truncation=True,
				)['input_ids']
			
			start_ids = [tokenizer.bos_token_id, tokenizer.cls_token_id]
			start_ids = [token_id for token_id in start_ids if token_id is not None]
			if len(start_ids) != 0:
				start_id = start_ids[0]
				for k in [k for k in model_inputs if k in ['input_ids', 'labels']]:
					for i in range(len(model_inputs[k])):
						# add the cls/bos token to models that don't automatically include it
						# such as gpt2. we also need to ensure the other keys are the same length.
						# this should be done for fine-tuning, and also for evaluation with surprisals.
						# it's just not done automatically because it's not needed for open-ended
						# text generation (see https://github.com/huggingface/transformers/issues/3311)
						if model_inputs[k][i][0] != start_id:
							model_inputs[k][i].insert(0, start_id)
							for k2 in model_inputs:
								if k2 == 'attention_mask':
									model_inputs[k2][i].insert(0, 1)
								
								if k2 == 'token_type_ids':
									model_inputs[k2][i].insert(0, 0)
			
			return model_inputs
		
		dataset = self.unformatted_dataset.get(split)
		
		if max_samples is not None:
			dataset = dataset.select(range(max_samples))
		
		dataset = dataset.map(
			function=preprocess_function,
			batched=True,
			num_proc=preprocessing_num_workers,
			remove_columns=drop_cols,
			load_from_cache_file=not overwrite_cache,
		)
		dataset.set_format(type='torch')
		
		if data_preprocessing_fn_strategy == 'once' and data_preprocessing_fn is not None:
			dataset = dataset.map(
				function=data_preprocessing_fn,
				batched=True,
				num_proc=preprocessing_num_workers,
				load_from_cache_file=not overwrite_cache,
				fn_kwargs=dict(
					model=model,
					tokenizer=tokenizer,
					**data_preprocessing_fn_kwargs
				)
			)
			
			if 'expanded_length' in dataset.features:
				self.texts, self.labels, self.metadata = data_preprocessing._expand_rows(
					self.texts,
					self.labels,
					self.metadata,
					expanded_lengths=dataset['expanded_length']
				)
				dataset = dataset.remove_columns(column_names='expanded_length')
			
			# update the labels if needed. This is for span denoising objectives.
			if data_preprocessing_fn in data_preprocessing.UPDATE_LABELS_FNS:
				self.labels = [tokenizer.decode(label[label != -100]) for label in dataset['labels']]
		
		return dataset
	
	def __str__(self) -> str:
		return self.__repr__()
	
	def __repr__(self) -> str:
		return (
			'Dataset({' +
			'\n    dataset=' + '\n'.join(
				('    ' if i != 0 else 'datasets.') + line 
				for i, line in enumerate(str(self.dataset).split('\n'))
			) + ',' +
			f'\n    texts=list with {len(self.texts)} entries,' +
			f'\n    labels=list with {len(self.labels)} entries,' +
			f'\n    metadata=list with {len(self.metadata)} entries' +
			'\n})'
		)
	
	def __getitem__(self, indices) -> dict:
		if not isinstance(indices, tuple):
			indices = tuple([indices])
		
		d = self.dataset[indices]
		d['texts'] = [self.texts[i] for i in indices]
		d['labels_text'] = [self.labels[i] for i in indices]
		d['metadata'] = [self.labels[i] for i in indices]
		
		return d
	
	def __len__(self) -> int:
		return len(self.dataset)
