import os

from typing import Optional
from datetime import datetime
from dataclasses import field
from dataclasses import dataclass

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
	
	max_val_samples: Optional[int] = field(
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
	
	lr: Optional[float] = field(
		default=2e-6,
		metadata={
			"help": "Learning rate."
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
	
	use_kl_baseline_loss: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Whether to add the KL divergence between the fine-tuned model's predictions and the original "
			"model's predictions to the loss term."
		}
	)
	
	kl_dataset: Optional[str] = field(
		default=None,
		metadata={
			"help": "If using the KL baseline loss term, the path to the dataset used to compute it."
		}
	)
	
	kl_batch_size: Optional[int] = field(
		default=32,
		metadata={
			"help": "If using the KL baseline loss term, how many examples to run per batch for it."
		}
	)
	
	kl_n_examples_per_batch: Optional[int] = field(
		default=20,
		metadata={
			"help": "If using the KL baseline loss term, how many examples to use per weight update "
			"to compute it. Keeping this smaller is generally better even if using GPU, since otherwise "
			"a lot of time is wasted computing outputs for pad tokens."
		}
	)
	
	kl_scaleby: Optional[float] = field(
		default=250,
		metadata={
			"help": "If using the KL baseline loss term, how much to scale it by. "
			"Note that the value returned for the KL baseline loss term is the average "
			"KL divergence per token."
		}
	)
	
	kl_max_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of total examples in the "
			"KL baseline loss term dataset to this value if set."
		},
	)
	
	kl_reduction: Optional[str] = field(
		default='none',
		metadata={
			"help": "Which reduction strategy to use for the KL baseline loss term. Default is 'none'; see "
			"`torch.nn.KLDivLoss` documentation for details. It is *highly* recommended you don't change this, "
			"since doing so could lead to loss being included for pad tokens."
		}
	)
	
	output_dir: Optional[str] = field(
		default=None,
		metadata={
			"help": "Used to store the output directory name. Do not set manually."
		}	
	)
	
	test_output_file_prefix: str = field(
		default=None,
		metadata={
			'help': 'What the prefix of the output file for the test datasets should be. '
			'Default uses the model name, with `/`, replaced by `-`.'
		}
	)
	
	def _set_output_dir(self, model_name: str) -> None:
		if self.output_dir is not None:
			return
		
		self.output_dir = os.path.join(
			'outputs',
			os.path.split(self.train_file)[-1].replace('.txt.gz', ''),
			model_name.replace("/", "-"),
			datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
		)
		while os.path.isdir(self.output_dir):
			self.output_dir = os.path.join(
				'outputs',
				os.path.split(self.train_file)[-1].replace('.txt.gz', ''),
				model_name.replace("/", "-"),
				datetime.now().strftime('%Y-%m-%d_%I-%M-%S.%f'),
			)
	
	def __post_init__(self):
		if self.train_file is None and self.validation_file is None and self.test_file is None:
			raise ValueError("Need a training/validation file or a test file.")
		
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
		
		if self.test_file is not None:
			test_file = self.test_file
			if isinstance(test_file, str):
				test_file = [test_file]
			
			for file in test_file:
				extension = file.split('.')[-1]
				if extension == 'gz':
					extension = file.split('.')[-2]
				
				if not extension in ['txt']:
					raise ValueError("All test files should be a txt file.")
		
		if self.min_epochs > self.epochs:
			raise ValueError(
				f'`min_epochs` {min_epochs} cannot be greater than `epochs` {epochs}.'
			)
		
		if self.output_dir is not None and self.train_file:
			raise ValueError(
				'`output_dir` is used internally for training. It should not be set manually unless running '
				'on test files only.'
			)