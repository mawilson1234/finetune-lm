import os

from typing import Optional
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