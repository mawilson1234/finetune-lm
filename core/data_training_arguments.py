import gzip
import json

from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import field
from dataclasses import dataclass

def load_metadata(dataset_path: str) -> list[dict]:
	'''
	Loads the metadata file for a dataset.
	'''
	with gzip.open(dataset_path.replace('.txt.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
		metadata = [json.loads(l) for l in in_file.readlines()]	
	
	return metadata

@dataclass
class DataTrainingArguments:
	"""Arguments pertaining to what data we are going to input our model for training and eval."""
	# dataset_name: Optional[str] = field(
	# 	default=None, 
	# 	metadata={"help": "The name of the dataset to use (via the datasets library)."}
	# )
	
	# dataset_config_name: Optional[str] = field(
	# 	default=None,
	# 	metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	# )
	
	train_file: Optional[str] = field(
		default=None, 
		metadata={"help": "The input training data file (a txt.gz file)."}
	)
	
	validation_file: Optional[str] = field(
		default=None,
		metadata={"help": "An evaluation data file to evaluate model performance on (a txt.gz file)."},
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
	
	max_val_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
			"value if set."
		},
	)
	
	ignore_pad_token_for_loss: Optional[bool] = field(
		default=True,
		metadata={
			"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
		},
	)
	
	epochs: Optional[int] = field(
		default=250,
		metadata={
			"help": "How many epochs to train for."
		}
	)
	
	patience: Optional[int] = field(
		default=None,
		metadata={
			"help": "How many epochs to wait for improvement on validation loss before ending training. 'None' trains until "
			"all epochs are finished."
		}
	)
	
	delta: Optional[float]: field(
		default=0.,
		metadata={
			"help": "How much improvement on validation loss is needed to reset the patience counter. '0' means"
			"any reduction in loss, no matter how small, resets the counter."
		}
	)
	
	def __post_init__(self):
		if self.dataset_name is None and self.train_file is None and self.validation_file is None:
			raise ValueError("Need either a dataset name or a training/validation file.")
		
		if self.train_file is not None:
			extension = self.train_file.split(".")[-1]
			if extension == 'gz':
				extension = self.train_file.split('.')[-2]
			
			if not extension in ['txt']:
				raise ValueError("`train_file` should be a txt file.")
		
		if self.validation_file is not None:
			extension = self.validation_file.split('.')[-1]
			if extension == 'gz':
				extension = self.validation_file.split('.')[-2]
			
			if not extension in ['txt']:
				raise ValueError("`validation_file` should be a txt file.")
			
			self.validation_dataset = load_dataset('text', data_files={'test': self.validation_file})
			self.validation_metadata = load_metadata(self.validation_file)
		
		if self.val_max_target_length is None:
			self.val_max_target_length = self.max_target_length