import os
import torch
import logging
logger = logging.getLogger(__name__)

import model_modifiers

from typing import Optional, Any, Callable
if __name__ == '__main__':
	from .constants import *
else:
	from constants import *

from dataclasses import field
from dataclasses import dataclass

@dataclass
class ModelArguments:
	'''Arguments pertaining to which model/config/tokenizer we are going to evaluate.'''
	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
	)
	
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	
	use_fast_tokenizer: Optional[bool] = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizers (backed by the tokenizers library) or not."},
	)
	
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	
	token: str = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
		repr=False,
	)
	
	use_gpu: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Set to use the GPU if available. If no GPU is available, the CPU will be used instead."
		}
	)
	
	model_task: Optional[str] = field(
		default=None,
		metadata={
			'help': 'Allows manually setting model task to one of `lm`, `mlm`, `seq2seq` on the command line.'
		}
	)
	
	model_modifier_fns: Optional[list[Callable]] = field(
		default_factory=lambda: [],
		metadata={
			"help": "A list of functions to apply to the model and tokenizer before fine-tuning/testing."
		}
	)
	
	model_modifier_fn_kwargs: Optional[dict[str,Any]] = field(
		default_factory=lambda: {},
		metadata={
			"help": "A dict of kwargs to apply to the model modifier functions. Keys should be the name "
			"of the function to which the corresponding kwargs should be passed."
		}
	)
	
	model_pre_train_step_callbacks: Optional[list] = field(
		default_factory=lambda: [],
		metadata={
			"help": "Callbacks, implemented as classes that take the model + kwargs when "
			"initialized and implement a __call__ function. To be run in sequence before "
			"`optimizer.step()` following each training step."
		}
	)
	
	model_pre_train_step_callbacks_kwargs: Optional[dict] = field(
		default_factory=lambda: {},
		metadata={
			"help": "Kwargs, besides the model, to be passed to the model_train_step_callback "
			"initializer of the corresponding name."
		},
	)
	
	def __post_init__(self):
		self.config_name = self.config_name or self.model_name_or_path
		self.tokenizer_name = self.tokenizer_name or self.model_name_or_path
		self.tokenizer_kwargs = get_tokenizer_kwargs(self.tokenizer_name)
		
		self.tokenizer_name = self.tokenizer_kwargs['pretrained_model_name_or_path']
		del self.tokenizer_kwargs['pretrained_model_name_or_path']
		
		if 'use_fast' in self.tokenizer_kwargs:
			self.use_fast_tokenizer = self.tokenizer_kwargs['use_fast']
			del self.tokenizer_kwargs['use_fast']
		
		self.token = None if not self.token else self.token
		if self.token is not None and os.path.isfile(os.path.expanduser(self.token)):
			with open(os.path.expanduser(self.token), 'rt') as in_file:
				self.token = in_file.read().strip()
		
		self.from_flax = self.model_name_or_path in MUELLER_T5_MODELS
		if not torch.cuda.is_available() and self.use_gpu:
			self.use_gpu = False
			logger.warning('`use_gpu` was set, but no GPU was found. Defaulting to CPU.')
		
		if self.model_task is not None:
			if self.model_task.lower() == 'lm':
				self.model_task = 'LM'
			elif self.model_task.lower() == 'mlm':
				self.model_task = 'MLM'
			elif self.model_task.lower() == 'seq2seq':
				self.model_task = 'Seq2Seq'
			else:
				raise ValueError(
					'`--model_task`, when provided, must be one of `lm`, `mlm`, or '
					f'`seq2seq` (case insensitive), but {self.model_task} was provided '
					'instead!'
				)
			
			set_model_task(model_name_or_path=self.model_name_or_path, model_task=self.model_task)
			# we need to set this, too, since it may be different if this is being set manually
			# (e.g., for a local model + tokenizer, which may be in different directories)
			if self.tokenizer_name != self.model_name_or_path:
				set_model_task(model_name_or_path=self.tokenizer_name, model_task=self.model_task)