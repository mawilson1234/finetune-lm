# Implements custom loss classes
import logging

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss

from typing import *
from constants import *

from transformers import (
	AutoModelForCausalLM,
	AutoModelForMaskedLM,
	AutoModelForSeq2SeqLM,
	AutoTokenizer
)
from transformers import logging as lg
lg.set_verbosity_error()

from dataset import Dataset
from datasets import disable_caching
# otherwise a cache file is saved for every time KLBaselineLoss is called,
# which we don't want
disable_caching()

from datasets.utils import disable_progress_bar
from datasets.utils import logging as dataset_utils_logging
disable_progress_bar()
dataset_utils_logging.set_verbosity_error()

import data_preprocessing

logger = logging.getLogger(__name__)

class OutputsDefaultLoss:
	def __init__(self, *args, **kwargs):
		pass
	
	def __call__(self, outputs: 'LMOutput', labels: torch.Tensor = None) -> torch.Tensor:
		return outputs.loss

class KLBaselineLoss(KLDivLoss):
	'''
	Calculates the loss on a dataset based on the KL divergence between the predictions
	of a model being fine-tuned and a pretrained version of that model. Higher KL divergences
	mean the model's predictions have deviated more from the baseline. This penalizes deviation
	from the baseline model to avoid catastrophic forgetting. You should add the KLBaselineLoss
	to another loss against the fine-tuning targets during fine-tuning. Otherwise, you would
	just be optimizing for the behavior of the original model and not striking the intended
	balance between performance on tuning data and the original model's behavior.
	
	Adapted from torch.nn.modules.loss.KLDivLoss
	'''
	def __init__(
		self,
		model: 'PreTrainedModel',
		tokenizer: 'PreTrainedTokenizer',
		dataset: Union[str, list[str]],
		batch_size: int = 1,
		scaleby: float = 1.,
		n_examples_per_batch: int = None,
		model_kwargs: dict = {},
		tokenizer_kwargs: dict = {},
		config_kwargs: dict = {},
		size_average = None,
		reduce = None,
		reduction: str = 'none',
		split_name: Union[str, list[str]] = 'train',
		max_samples_per_dataset: Union[int,list[int]] = None,
		max_sample_length: int = None,
		preprocessing_num_workers: int = None,
		overwrite_cache: bool = None,
		data_preprocessing_fn: Callable = data_preprocessing.identity,
		data_preprocessing_fn_kwargs: dict = None,
		data_preprocessing_fn_strategy: str = 'once',
		return_all: bool = False,
		model_callbacks: dict[str,list[callable]] = None,
		model_callbacks_kwargs: dict[str,dict] = None,
		baseline_model_callbacks: dict[str,list[callable]] = None,
		baseline_model_callbacks_kwargs: dict[str,dict] = None,
		baseline_model_modifier_fns: list[callable] = None,
		baseline_model_modifier_fn_kwargs: dict[str,dict] = None,
	) -> None:
		'''
		Creates a KL divergence loss object that codes divergence of a fine-tuned
		model compared to a baseline model on a specified dataset.
		
			params:
				model (PreTrainedModel)				: a huggingface pretrained model
				tokenizer (PreTrainedTokenizer)		: a huggingface pretrained tokenizer (should match the model)
				dataset (str)						: a str or list of strs indicating the files to use for
													  the KL baseline loss.
				batch_size (int)					: the number of sentences to run through the models at a single time.
													  KL divergence is computed per sentence and averaged
				scaleby (float)						: returned loss is multiplied by this
				n_examples_per_batch (int)			: it may be too time-consuming to calculate the KLBaselineLoss on
													  the basis of the entire dataset, if the dataset is large.
													  you can use this to set how many random samples to draw
													  from dataset to use when calculating loss. If not set,
													  all examples will be used each time.
													  this is per dataset passed.
				model_kwargs (dict)					: used to create a baseline version of the passed model
				tokenizer_kwargs (dict)				: used to create a baseline version of the passed tokenizer
				size_average						: passed to KLDivLoss
				reduce 								: passed to KLDivLoss
				reduction							: 
				data_preprocessing_fn 				: a function used to preprocess the datasets.
													  must accept at least `inputs` (dict[str,torch.Tensor]) and 
													  `tokenizer` arguments. Additional kwargs are passed via
													  `data_preprocessing_fn_kwargs`.
				data_preprocessing_fn_kwargs		: additional kwargs passed to data_preprocessing_fn.
				data_preprocessing_fn_strategy		: whether to preprocess all datasets `once`, or reprocess
													  per batch (=`epoch`).
				return_all 							: whether to return the kl div loss for each example.
				model_callbacks: dict[str,list[callable]]: dict mapping 'pre_batch', 'post_batch', 'pre_dataset',
													  'post_dataset' to a list of
													  callbacks to run for the model being fine-tuned.
				model_callbacks_kwargs: dict[str,dict]: dict mapping the same keys to a dict mapping
													  to dicts mapping the callback names to dicts of kwargs
													  to pass to their init method.
				baseline_model_callbacks: dict[str,list[callable]]: same, but for the baseline model.
				baseline_model_callbacks_kwargs: dict[str,dict]: same, but for the baseline model.
				baseline_model_modifier_fns: list[callable]: a list of functions to use to modify the baseline model
													  and tokenizer after they are initialized.
				baseline_model_modifier_fn_kwargs: dict[str,dict]: a dict mapping baseline model modifier function 
													 names to kwargs to pass to that function.
		'''
		super(KLBaselineLoss, self).__init__(size_average, reduce, reduction)
		self.model = model
		self.tokenizer = tokenizer
		self.device	= self.model.device
		self.batch_size = batch_size
		
		model_callbacks = {} if model_callbacks is None else model_callbacks
		model_callbacks_kwargs = {} if model_callbacks_kwargs is None else model_callbacks_kwargs
		baseline_model_callbacks = {} if baseline_model_callbacks is None else baseline_model_callbacks
		baseline_model_callbacks_kwargs = {} if baseline_model_callbacks_kwargs is None else baseline_model_callbacks_kwargs
		baseline_model_modifier_fns = [] if baseline_model_modifier_fns is None else baseline_model_modifier_fns
		baseline_model_modifier_fn_kwargs = {} if baseline_model_modifier_fn_kwargs is None else baseline_model_modifier_fn_kwargs
		
		# logger.debug(f'Initializing Baseline Tokenizer for KLBaselineLoss: {self.tokenizer.name_or_path}')
		# self.baseline_tokenizer = load_tokenizer(self.tokenizer.name_or_path, **tokenizer_kwargs)
		
		# logger.debug(f'Initializing Baseline Model for KLBaselineLoss: {self.model.name_or_path}')
		# self.baseline_model	= load_model(self.model.name_or_path, **model_kwargs).to(self.device)
		
		logger.debug(f'Initializing baseline tokenizer and model for KLBaselineLoss: {self.model.name_or_path}')
		self.baseline_tokenizer, self.baseline_model = load_tokenizer_and_model(
			name_or_path=self.model.name_or_path,
			model_kwargs=model_kwargs,
			tokenizer_kwargs=tokenizer_kwargs,
			config_kwargs=config_kwargs,
			use_gpu=self.model.device.type == 'cuda',
			model_modifier_fns=baseline_model_modifier_fns,
			model_modifier_fn_kwargs=baseline_model_modifier_fn_kwargs,
		)
		
		for fn in baseline_model_modifier_fns:
			kwargs = baseline_model_modifier_fns_kwargs.get(fn.__name__, {})
			self.baseline_tokenizer, self.baseline_model = fn(
				model=self.baseline_model, tokenizer=self.tokenizer, **kwargs
			)
		
		# we're not fine-tuning this
		# _ = is to prevent printing
		_ = self.baseline_model.eval()
		
		# set up the dataset
		self.dataset = dataset
		if not isinstance(self.dataset, list):
			self.dataset = [self.dataset]
		
		if not isinstance(split_name, list):
			split_name = [split_name]
		
		if len(split_name) == 1:
			split_name *= len(self.dataset)
		
		if not isinstance(max_samples_per_dataset, list):
			max_samples_per_dataset = [max_samples_per_dataset]
		
		if len(max_samples_per_dataset) == 1:
			max_samples_per_dataset *= len(self.dataset)
		
		self.dataset = [
			Dataset(
				file=file,
				model=self.baseline_model,
				tokenizer=self.baseline_tokenizer,
				split_name=split_name,
				max_samples=max_samples,
				max_length=max_sample_length,
				preprocessing_num_workers=preprocessing_num_workers,
				overwrite_cache=overwrite_cache,
				data_preprocessing_fn=data_preprocessing_fn,
				data_preprocessing_fn_kwargs=data_preprocessing_fn_kwargs,
				data_preprocessing_fn_strategy=data_preprocessing_fn_strategy,
			)
			for file, split_name, max_samples in zip(self.dataset, split_name, max_samples_per_dataset)
		]
		
		# if the number of examples is a single value, broadcast it so that 
		# all datasets get that many examples pulled
		if isinstance(n_examples_per_batch, int):
			n_examples_per_batch = [n_examples_per_batch] * len(self.dataset)
		elif len(n_examples_per_batch) != len(self.dataset):
			# if the number of examples is a list to correspond to each dataset,
			# ensure there are the right number of entries
			raise ValueError(
				f'{len(n_examples_per_batch)} values were provided for the number of examples, '
				f'but only {len(self.dataset)} were provided! You should provide as many numbers '
				'as datasets, or else provide one number to be used for all datasets.'
			)
		
		# can't use more examples for each dataset than we've got
		self.n_examples = [
			d.num_rows if n is None else min(n, d.num_rows)
			for d, n in zip(self.dataset, n_examples_per_batch)
		]
		
		# broadcast this, too
		if isinstance(scaleby, float) or isinstance(scaleby, int):
			scaleby = [scaleby] * len(self.dataset)
		elif len(scaleby) != len(self.dataset):
			# if the scaleby is a list to correspond to each dataset,
			# ensure there are the right number of entries
			raise ValueError(
				f'{len(kl_scaleby)} values were provided for the scaling terms, '
				f'but only {len(datasets)} were provided! You should provide as many numbers '
				'as datasets, or else provide one number to be used for all datasets.'
			)
		
		self.scaleby = scaleby
		
		self.data_preprocessing_fn = data_preprocessing_fn
		self.data_preprocessing_fn_kwargs = data_preprocessing_fn_kwargs if data_preprocessing_fn_kwargs else {}
		self.data_preprocessing_fn_strategy = data_preprocessing_fn_strategy if data_preprocessing_fn_strategy is not None else ''
		self.return_all = return_all
		
		self.model_callbacks = {}
		for when_to_run_callbacks in model_callbacks:
			self.model_callbacks[when_to_run_callbacks] = model_callbacks[when_to_run_callbacks]
			if self.model_callbacks[when_to_run_callbacks] and not isinstance(model_callbacks[when_to_run_callbacks], list):
				self.model_callbacks[when_to_run_callbacks] = [model_callbacks[when_to_run_callbacks]]
			
			if self.model_callbacks[when_to_run_callbacks]:
				self.model_callbacks[when_to_run_callbacks] = [
					callback(
						model=self.model, tokenizer=self.tokenizer,
						**model_callbacks_kwargs.get(when_to_run_callbacks, {}).get(callback.__name__, {})
					)
					for callback in self.model_callbacks[when_to_run_callbacks]
				]
		
		self.baseline_model_callbacks = {}
		for when_to_run_callbacks in baseline_model_callbacks:
			self.baseline_model_callbacks[when_to_run_callbacks] = baseline_model_callbacks[when_to_run_callbacks]
			if self.baseline_model_callbacks[when_to_run_callbacks] and not isinstance(baseline_model_callbacks[when_to_run_callbacks], list):
				self.baseline_model_callbacks[when_to_run_callbacks] = [baseline_model_callbacks[when_to_run_callbacks]]
			
			if self.baseline_model_callbacks[when_to_run_callbacks]:
				self.baseline_model_callbacks[when_to_run_callbacks] = [
					callback(
						model=self.baseline_model, tokenizer=self.baseline_tokenizer,
						**baseline_model_callbacks_kwargs.get(when_to_run_callbacks, {}).get(callback.__name__, {})
					)
					for callback in self.baseline_model_callbacks[when_to_run_callbacks]
				]
	
	def forward(
		self, 
		outputs: 'LMOutput',
		labels: torch.Tensor = None
	) -> torch.Tensor:
		'''
		Computes KLBaselineLoss between the predictions of the baseline model
		and the predictions of the fine-tuned model on the basis of self.n_examples
		from self.dataset. Samples are randomized with each call.
		
			params:
				outputs (LMOutput)		: ignored, included for compatibility with general architecture.
			
			returns:
				kl_div (torch.Tensor)	: the mean KL divergence between the model and the baseline model
										  across n_examples of the dataset, multiplied by the scaling factor
				kl_divs (list[torch.Tensor]): the individual KL divergence for each example in each dataset
										  returned if self.return_all=True.
		'''
		# construct a comparison dataset for this call with n random examples from each dataset
		comp_datasets = [
			d.dataset.shuffle().select(range(n)) 
			for d, n in zip(self.dataset, self.n_examples)
		]
		dataloaders = [
			torch.utils.data.DataLoader(d, batch_size=self.batch_size, collate_fn=data_preprocessing.pad_batch)
			for d in comp_datasets
		]
		
		total_kl_divs = [torch.Tensor([0.]).to(self.model.device) for dataloader in dataloaders]
		kl_divss = [[] for dataloader in dataloaders]
		for kl_divs, total_kl_div, dataloader in zip(kl_divss, total_kl_divs, dataloaders):
			for callback in self.model_callbacks.get('pre_dataset', []):
				callback(epoch=None, batch=None)
			
			for callback in self.baseline_model_callbacks.get('pre_dataset', []):
				callback(epoch=None, batch=None)
			
			for i, batch in enumerate(dataloader):
				for callback in self.model_callbacks.get('pre_batch', []):
					callback(epoch=None, batch=i)
				
				for callback in self.baseline_model_callbacks.get('pre_batch', []):
					callback(epoch=None, batch=i)
				
				if self.data_preprocessing_fn_strategy == 'per_batch':
					batch = self.data_preprocessing_fn(
						inputs=batch,
						model=self.model,
						tokenizer=self.baseline_tokenizer,
						**self.data_preprocessing_fn_kwargs,
					)
					
					# we don't need to bother doing anything like expanding the texts
					# and labels here, since we're not saving these results.
					if 'expanded_length' in batch:
						del batch['expanded_length']
				
				batch_inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
				outputs = self.model(**batch_inputs).logits
				
				# we're not training the baseline model, so no need to get gradients for it
				with torch.no_grad():
					baseline_outputs = self.baseline_model(**batch_inputs).logits
				
				# KLDivLoss expects one of these in log space and the other not in log space,
				# so we just use log softmax for the fine-tuned model's predictions.
				outputs = F.log_softmax(outputs, dim=-1)
				baseline_outputs = F.softmax(baseline_outputs, dim=-1)
				
				# summing gives us the KL divergence for each token for each input
				kl_div = super(KLBaselineLoss, self).forward(outputs, baseline_outputs).sum(dim=-1)
				
				divisor = torch.ones(kl_div.shape, dtype=int).to(kl_div.device)
				# remove pad tokens from the overall loss
				# this retains the values for the non-pad tokens since the
				# attention mask for those is 1, while the attention mask for
				# pad tokens is 0, so it removes those values.
				# we need to check the shapes in case we're doing this with a seq2seq model,
				# where the outputs and attention mask won't match up.
				if 'attention_mask' in batch_inputs and kl_div.shape == batch_inputs['attention_mask'].shape:
					kl_div *= batch_inputs['attention_mask']
					divisor *= batch_inputs['attention_mask']
				
				# remove labels marked as -100 from the loss
				# these correspond to non-masked tokens in MLM
				# objectives where we only want to compute loss 
				# on the mask token indices
				if 'labels' in batch:
					dont_ignore = (batch['labels'].detach().clone() != -100).int().to(kl_div.device)
					kl_div *= dont_ignore
					divisor *= dont_ignore
				
				if self.return_all:
					kl_divs.append(kl_div)
				
				# get mean for each example and accumulate the total for the whole set
				kl_div = kl_div.sum(dim=-1)/divisor.sum(dim=-1)
				total_kl_div += kl_div.sum(dim=-1)
				
				for callback in self.model_callbacks.get('post_batch', []):
					callback(epoch=None, batch=i)
				
				for callback in self.baseline_model_callbacks.get('post_batch', []):
					callback(epoch=None, batch=i)
			
			for callback in self.model_callbacks.get('post_dataset', []):
				callback(epoch=None, batch=None)
			
			for callback in self.baseline_model_callbacks.get('post_dataset', []):
				callback(epoch=None, batch=None)
		
		total_kl_divs = [
			(total_kl_div/n) * scaleby 
			for total_kl_div, n, scaleby in zip(total_kl_divs, self.n_examples, self.scaleby)
		]
		
		# the average of the averages, since we want each dataset to be of equal importance
		total_kl_div = torch.sum(torch.cat(total_kl_divs))/len(total_kl_divs)
		
		if self.return_all:
			# pad so we can return all KL divs in a single tensor,
			# instead of a list of tensors
			max_dim = [max(kl_div.shape[-1] for kl_div in kl_divs) for kl_divs in kl_divss]
			kl_divs = [
				[
					F.pad(input=kl_div, pad=(0, max_dim - kl_div.shape[-1]), value=0)
					for kl_div in kl_divs
				]
				for kl_divs in kl_divss
			]
			kl_divs = [torch.cat(kl_divs) for kl_divs in kl_divss]
			
			return total_kl_div, kl_divs
		else:
			return total_kl_div
