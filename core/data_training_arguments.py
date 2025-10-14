import os
import re
import sys
import torch
import random
import logging
import loss_classes
import data_evaluation
import data_preprocessing

from typing import Optional, Union, Callable, Any
from datetime import datetime
from dataclasses import (
	field,
	fields,
	dataclass,
)

logger = logging.getLogger(__name__)

DATASET_TYPES = ('train', 'validation', 'test')

@dataclass
class DataTrainingArguments:
	"""Arguments pertaining to what data we are going to input our model for training and eval."""
	train_file: Optional[str] = field(
		default=None, 
		metadata={"help": "The input training data file (a txt.gz file)."}
	)
	
	validation_file: Optional[str] = field(
		default=None,
		metadata={"help": "An evaluation data file to evaluate model performance on (a txt.gz file)."},
	)
	
	test_file: Optional[str] = field(
		default=None,
		metadata={"help": "A test data file or files to evaluate model performance on (a txt.gz file)."},
	)
	
	overwrite_cache: bool = field(
		default=False, 
		metadata={"help": "Overwrite the cached training and evaluation sets"}
	)
	
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	
	max_length: Optional[int] = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": "Whether to pad all samples to model maximum sentence length. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
			"efficient on GPU but very bad for TPU."
		},
	)
	
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	
	per_device_train_batch_size: Optional[int] = field(
		default=32,
		metadata={
			"help": "Number of training examples per batch per device."
		}
	)
	
	max_validation_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
			"value if set."
		},
	)
	
	per_device_validation_batch_size: Optional[int] = field(
		default=32,
		metadata={
			"help": "Number of validation examples per batch per device."
		}
	)
	
	max_test_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
			"value if set."
		},
	)
	
	per_device_test_batch_size: Optional[int] = field(
		default=32,
		metadata={
			"help": "Number of test examples per batch per device."
		}
	)
	
	ignore_pad_token_for_loss: Optional[bool] = field(
		default=True,
		metadata={
			"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
		},
	)
	
	data_preprocessing_fn: Optional[Union[Callable, dict[str,Callable]]] = field(
		default_factory=lambda: {
			'train': data_preprocessing.identity,
			'validation': data_preprocessing.identity,
			'test': data_preprocessing.identity,
		},
		metadata={
			"help": 'For each type of dataset, a function to use for additional data preprocessing '
			'(for instance, to mask inputs). Default is the identity function, which sets labels to '
			'the input_ids. `mask_bert` and `mask_span` functions are provided in '
			'"core/data_preprocessing.py".'
		}
	)
	
	data_preprocessing_fn_strategy: Optional[Union[str, dict[str,str]]] = field(
		default_factory=lambda: {},
		metadata={
			"help": 'How often to run the data preprocessing function for each type of dataset. '
			'Possible options are "once" (preprocess the whole dataset once) and "per_batch" '
			'(rerun the preprocessing every batch)". "once" is useful if you want to run BERT-'
			'style masking, where mask indices are identical for each example every epoch. '
			'"per_batch" is useful if you want to run RoBERTa style masking, where the mask indices '
			'for an example may vary every epoch.'
		}
	)
	
	data_preprocessing_fn_kwargs: Optional[Union[dict, dict[str,dict]]] = field(
		default_factory=lambda: {},
		metadata={
			'help': 'Additional keyword arguments to pass to the data preprocessing functions.'
		}
	)
	
	epochs: Optional[int] = field(
		default=250,
		metadata={
			"help": "How many (max) epochs to train for."
		}
	)
	
	min_epochs: Optional[int] = field(
		default=0,
		metadata={
			"help": "The minimum number of epochs to train for. Overrides patience."
		}
	)
	
	patience: Optional[int] = field(
		default=None,
		metadata={
			"help": "How many epochs to wait for improvement on validation loss before ending training. 'None' trains until "
			"all epochs are finished."
		}
	)
	
	delta: Optional[float] = field(
		default=0.,
		metadata={
			"help": "How much improvement on validation loss is needed to reset the patience counter. '0' means"
			"any reduction in loss, no matter how small, resets the counter."
		}
	)
	
	train_optimizer: Optional[torch.optim] = field(
		default=torch.optim.AdamW,
		metadata={
			"help": "Class to use as optimizer for fine-tuning. Takes model parameters, plus kwargs in "
			"`data_args.train_optimizer_kwargs`."
		}
	)
	
	train_optimizer_kwargs: Optional[Union[dict,list[str]]] = field(
		default_factory=lambda: {
			'lr': 2e-6,
			'weight_decay': 0,
		},
		metadata={
			"help": "Kwargs to pass to the train optimizer. If passed as a dictionary, "
			"kwargs will be set to instance attributes of this dataclass to facilitate "
			"hyperparameter optimization with optuna. If passed as a list of str, "
			"attributes already in DataTrainingArguments with the same names will be "
			"passed as kwargs to the train optimizer."
		}
	)
	
	gradient_accumulation_steps: Optional[int] = field(
		default=1,
		metadata={
			"help": 'How many batches to accumulate loss over before doing a step of gradient descent.'
		}
	)
	
	loss_classes: Optional[dict] = field(
		default_factory=lambda: {
			'train': [loss_classes.OutputsDefaultLoss],
			'validation': [loss_classes.OutputsDefaultLoss],
		},
		metadata={
			"help": "A dict of classes to use to compute loss for each type of dataset."
			"Not used for test datasets."
		},
	)
	
	loss_reduction_fns: Optional[dict] = field(
		default_factory=lambda: {
			'train': torch.sum,
			'validation': torch.sum,
		},
		metadata={
			"help": "How to combine the values for the losses when using multiple "
			"losses during fine-tuning."
		}
	)
	
	loss_classes_kwargs: Optional[dict[str,Any]] = field(
		default_factory=lambda: {
			'train': {
				'OutputsDefaultLoss': {},
				'KLBaselineLoss': {
					'dataset': None,
					'batch_size': 32,
					'scaleby': 2.5,
					'n_examples_per_batch': 20,
					'max_samples_per_dataset': None,
				},
			},
			'validation': {
				'OutputsDefaultLoss': {},
			}
		},
		metadata={
			"help": "Kwargs to use with the loss classes for each type of dataset. Classes are identified "
			"by name in the dict. Kwargs are added to the dataclass instance attributes under "
			"`DatasetType_ClassName_Key`, and then pulled from there to facilitate optimization with optuna."
		}
	)
	
	evaluation_fns: Optional[dict] = field(
		default_factory=lambda: {
			'train': [data_evaluation.evaluate_batch],
			'validation': [data_evaluation.evaluate_batch],
			'test': [data_evaluation.evaluate_batch],
		},
		metadata={
			'help': 'A dictionary specifying which evaluation functions to run on the model outputs. '
			'An evaluation function is passed at least the following for each batch of data for each epoch: '
			'`model`, `tokenizer`, `inputs`, `input_texts`, `input_labels`, `batch_outputs`, `input_nums`, '
			'`batch_metadata`, `epoch` (fine-tuning only), `batch_number` (fine-tuning only), `dataset_type`, '
			'`loss` (fine-tuning only), and additional kwargs by any names (these contain the '
			'individual losses during fine-tuning, the current model_mode for gpt-bert, and the dataset_name). '
			'They must return a list of dictionaries. All results will be collected into a single dataframe and '
			'saved as a csv.gz file in the output directory. Default is to always use `evaluate_batch`, a provided '
			'function that dispatches to a function appropriate to the model type (LM, MLM, or Seq2Seq). Functions '
			'are not run during an optimization study, since saving all results for all trials would be prohibitive.'
		}
	)
	
	evaluation_fns_kwargs: Optional[dict] = field(
		default_factory=lambda: {
			'train': {},
			'validation': {},
			'test': {},
		},
		metadata={
			"help": 'Additional kwargs to pass to evaluation functions when they are called, as a dict mapping the '
			'function name to a dict containing the kwargs for the appropriate evaluation function (`train`, '
			'`validation`, or `test`).'
		}
	)
	
	output_dir: Optional[str] = field(
		default=None,
		metadata={
			"help": "Used to store the output directory name. Set automatically if not provided. "
			'${now} will be replaced with the date and time when the script is run.'
		}	
	)
	
	test_output_file_prefix: str = field(
		default=None,
		metadata={
			'help': 'What the prefix of the output file for the test datasets should be. '
			'Default uses the model name, with `/`, replaced by `-`.'
		}
	)
	
	seed: int = field(
		default_factory=lambda: random.randint(0, 2**32-1),
		metadata={
			"help": "The random seed to use. Set before optimization if using optuna, otherwise set "
			"prior to anything else. Default is a random number between 0 and 2**32-1."
		}
	)
	
	save_tmp_test_files: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Whether to store temporary files for each batch of results as they "
			"are computed for the test sets. These can be used to resume evaluation if "
			"it is interrupted. Useful for large models which may not be able to finish "
			"on time within cluster resource limits."
		},
	)
	
	save_best_model_state_to_disk: Optional[bool] = field(
		default=True,
		metadata={
			"help": "Whether to save the best model state to disk if a training dataset is provided. "
			"If no training dataset is provided or `--do_optimize` is set, this is ignored."
		}
	)
	
	def _set_output_dir(self, model_name: str) -> None:
		if self.output_dir is not None:
			if not '{now}' in self.output_dir:
				return
			
			output_dir = self.output_dir.replace(
				'{now}',
				datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
			)
			while os.path.isdir(output_dir):
				output_dir = self.output_dir.replace(
					'{now}',
					datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
				)
			
			self.output_dir = output_dir
			return
		
		if self.train_file is not None:
			baseline = re.sub(r'\.(txt|json)\.gz$', '', os.path.split(self.train_file)[-1])
			self.output_dir = os.path.join(
				'outputs',
				baseline,
				model_name.replace('/', '-'),
				datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f') if self.train_file is not None else '',
			)
			while os.path.isdir(self.output_dir):
				self.output_dir = os.path.join(
					'outputs',
					baseline,
					model_name.replace('/', '-'),
					datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
				)
		else:
			self.output_dir = os.path.join(
				'outputs',
				'test_only',
				model_name.replace('/', '-')
			)
			while os.path.isdir(self.output_dir):
				self.output_dir = os.path.join(
					'outputs',
					'test_only',
					model_name.replace('/', '-')
				)
	
	def __post_init__(self):
		if self.train_file is None and self.validation_file is None and self.test_file is None:
			raise ValueError("Need a training/validation file or a test file.")
		
		if self.train_file is not None:
			extension = self.train_file.split(".")[-1]
			if extension == 'gz':
				extension = self.train_file.split('.')[-2]
			
			if not extension in ['txt', 'json']:
				raise ValueError("`train_file` should be a txt or json file.")
		
		if self.validation_file is not None:
			extension = self.validation_file.split('.')[-1]
			if extension == 'gz':
				extension = self.validation_file.split('.')[-2]
			
			if not extension in ['txt', 'json']:
				raise ValueError("`validation_file` should be a txt or json file.")
		
		if self.test_file is not None:
			test_file = self.test_file
			if isinstance(test_file, str):
				test_file = [test_file]
			
			for file in test_file:
				extension = file.split('.')[-1]
				if extension == 'gz':
					extension = file.split('.')[-2]
				
				if not extension in ['txt', 'json']:
					raise ValueError("All test files should be a txt or json file.")
		
		if self.min_epochs > self.epochs:
			raise ValueError(
				f'`min_epochs` {self.min_epochs} cannot be greater than `epochs` {self.epochs}.'
			)
		
		# if self.output_dir is not None and self.train_file:
		# 	raise ValueError(
		# 		'`output_dir` is used internally for training. It should not be set manually unless running '
		# 		'on test files only.'
		# 	)
		
		if isinstance(self.data_preprocessing_fn, Callable):
			self.data_preprocessing_fn = {
				k: self.data_preprocessing_fn for k in DATASET_TYPES
			}
		
		if (
			isinstance(self.data_preprocessing_fn, dict) and 
			not all(x in self.data_preprocessing_fn for x in DATASET_TYPES)
		):
			missing_keys = [k for k in DATASET_TYPES if k not in self.data_preprocessing_fn]
			for k in missing_keys:
				self.data_preprocessing_fn[k] = data_preprocessing.identity
		
		# if we have a dict of kwargs, if we don't have all the keys we expect,
		# fill things out. We do this by the following strategy:
		# - if we have some but not all keys corresponding to dataset types,
		#   set the missing keys to empty dicts
		# - if we have NO keys corresponding to dataset types,
		#   set the entire dict to the kwargs for each dataset type
		if not all(k in self.data_preprocessing_fn_kwargs for k in DATASET_TYPES):
			if not any(k in self.data_preprocessing_fn_kwargs for k in DATASET_TYPES):
				self.data_preprocessing_fn_kwargs = {
					k: self.data_preprocessing_fn_kwargs for k in DATASET_TYPES
				}
			else:
				missing_keys = [k for k in DATASET_TYPES if k not in self.data_preprocessing_fn_kwargs]
				for k in missing_keys:
					self.data_preprocessing_fn_kwargs[k] = {}
		
		if isinstance(self.data_preprocessing_fn_strategy, str):
			self.data_preprocessing_fn_strategy = {
				k: self.data_preprocessing_fn_strategy for k in DATASET_TYPES
			}
		
		if (
			isinstance(self.data_preprocessing_fn_strategy, dict) and 
			not all(x in self.data_preprocessing_fn_strategy for x in DATASET_TYPES)
		):
			missing_keys = [k for k in DATASET_TYPES if k not in self.data_preprocessing_fn_strategy]
			for k in missing_keys:
				self.data_preprocessing_fn_strategy[k] = 'once'
		
		# this makes it easier to do optimization
		# if we have a dict, set the instance's attributes
		# to the keys in the optimizer kwargs so that
		# it's easier to write the logic that optimizes
		# the hyperparameters for the optimizer.
		if isinstance(self.train_optimizer_kwargs, dict):
			if any(self.train_optimizer_kwargs) and not hasattr(self, '_added_in_post_init'):
				self._added_in_post_init = []
			
			for k in self.train_optimizer_kwargs:
				setattr(self, k, self.train_optimizer_kwargs[k])
				self._added_in_post_init.append(k)
		
		# need to do this for the loss function parameters, too
		# we'll later retrieve them from the dataclass itself,
		# so this will allow easier optimization.
		for k in self.loss_classes_kwargs:
			if not hasattr(self, '_added_in_post_init'):
				self._added_in_post_init = []
			
			for klass in self.loss_classes_kwargs[k]:
				for k2 in self.loss_classes_kwargs[k][klass]:
					setattr(self, f'{k}_{klass}_{k2}', self.loss_classes_kwargs[k][klass][k2])
					self._added_in_post_init.append(f'{k}_{klass}_{k2}')
		
		if self.patience is None:
			self.patience = self.epochs
	
	# this lets us view attributes added during post_init
	# that we want in the repr for reproduceability during
	# optimization.
	def __str__(self):
		return self.__repr__()
	
	def __repr__(self):
		flds = [f for f in fields(self) if f.repr]
		# taken from dataclasses.py
		parent_repr = (
			f'{self.__class__.__qualname__}(' +
			', '.join([f'{f.name}={getattr(self, f.name)!r}' for f in flds]) +
			')'
		)
		
		if hasattr(self, '_added_in_post_init') and any(self._added_in_post_init):
			parts = [parent_repr[:-1], parent_repr[-1]]
			additional_info = ''
			for added in self._added_in_post_init:
				additional_info += f', {added}={getattr(self, added)}'
			
			parts[-1] = additional_info + parts[-1]
			parent_repr = ''.join(parts)
		
		return parent_repr