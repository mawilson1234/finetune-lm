# Implements a loss that combines loss on the new data
# with the KL divergence between the updated model's predictions
# and the pretrained model's predictions
import logging

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss

from tqdm import tqdm
from typing import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as lg
lg.set_verbosity_error()

from datasets import (
	load_dataset, 
	Dataset, 
	DatasetDict, 
	disable_caching
)
# otherwise a cache file is saved for every time KLBaselineLoss is called,
# which we don't want
disable_caching()
from datasets.utils import logging as dataset_utils_logging
from datasets.utils import disable_progress_bar
disable_progress_bar()
dataset_utils_logging.set_verbosity_error()

log = logging.getLogger(__name__)

def pad_tensor(t: torch.Tensor, pad: int, dim: int = -1) -> torch.Tensor:
	'''
	Pads a tensor to length pad in dim dim.
	From https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8?u=mawilson
	
		params:
			t (torch.Tensor): tensor to pad
			pad (int)		: the size to pad to
			dim (int)		: dimension to pad
		
		returns:
			a new torch.Tensor padded to 'pad' in dimension 'dim'
	'''
	pad_size = list(t.shape)
	pad_size[dim] = pad - t.size(dim)
	return torch.cat([t, torch.zeros(*pad_size, dtype=t.dtype, device=t.device)], dim=dim)

def pad_batch(batch: tuple) -> tuple:
	'''Pads examples in a batch to the same length.'''
	# Use attention mask to trim examples first, since otherwise
	# preprocessing the dataset into tensors pads everything to max length.
	# this wastes a lot of resources on running the model on pad tokens for
	# no benefit (especially in the KL dataset). If we have an attention mask, 
	# we use it to trim off any trailing ignored (pad) tokens, then pad examples
	# if still needed.
	if all('attention_mask' in ex for ex in batch):
		# add two so we predict the end of text token
		max_attn_mask = max(map(lambda ex: torch.where(ex['attention_mask'] == 1)[0][-1], batch)) + 2
		for ex in batch:
			for k in ex:
				if isinstance(ex[k], torch.Tensor):
					ex[k] = ex[k][:max_attn_mask]
	
	max_len = max(map(lambda ex: ex['input_ids'].size(-1), batch))
	batch = list(map(lambda ex: {k: pad_tensor(ex[k], pad=max_len, dim=-1) for k in ex}, batch))
	batch = {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}
	return batch

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
		dataset: Union[Dataset, list[Dataset]],
		batch_size: int = 1,
		scaleby: float = 1.,
		n_examples_per_batch: int = None,
		model_kwargs: Dict = {},
		tokenizer_kwargs: Dict = {},
		size_average = None,
		reduce = None,
		reduction: str = 'none',
	) -> None:
		'''
		Creates a KL divergence loss object that codes divergence of a fine-tuned
		model compared to a baseline model on a specified dataset.
		
			params:
				model (PreTrainedModel)				: a huggingface pretrained model
				tokenizer (PreTrainedTokenizer)		: a huggingface pretrained tokenizer (should match the model)
				dataset (Dataset)					: a dataset in huggingface's datasets format that
													  has been pretokenized for use with the same kind of tokenizer
													  as passed, or a list of such datasets
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
				reduction							: passed to KLDivLoss
		'''
		super(KLBaselineLoss, self).__init__(size_average, reduce, reduction)
		self.model = model
		self.tokenizer = tokenizer
		self.device	= self.model.device
		self.batch_size = batch_size
		
		log.debug(f'Initializing Baseline Model for KLBaselineLoss: {self.model.name_or_path}')
		self.baseline_model	= (
			AutoModelForCausalLM
				.from_pretrained(self.model.name_or_path, **model_kwargs)
				.to(self.device)
		)
		
		# we're not fine-tuning this
		# _ = is to prevent printing
		_ = self.baseline_model.eval()
		
		log.debug(f'Initializing Baseline Tokenizer for KLBaselineLoss: {self.tokenizer.name_or_path}')
		self.baseline_tokenizer = AutoTokenizer.from_pretrained(
			self.tokenizer.name_or_path, 
			**tokenizer_kwargs
		)
		
		# set up the dataset
		self.dataset = dataset
		if not isinstance(self.dataset, list):
			self.dataset = [self.dataset]
		
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
	
	def forward(
		self, 
		return_all: bool = False,
	) -> torch.Tensor:
		'''
		Computes KLBaselineLoss between the predictions of the baseline model
		and the predictions of the fine-tuned model on the basis of self.n_examples
		from self.dataset. Samples are randomized with each call.
		
			params:
				return_all (bool)		: whether to return a list containing every individual KL divergence
										  in a list in addition to the mean
			
			returns:
				kl_div (torch.Tensor)	: the mean KL divergence between the model and the baseline model
										  across n_examples of the dataset, multiplied by the scaling factor
				kl_divs (list[torch.Tensor]): the individual KL divergence for each example in each dataset
										  returned if return_all=True.
		'''
		# construct a comparison dataset for this call with n random examples from each dataset
		comp_datasets = [
			d.shuffle().select(range(n)) 
			for d, n in zip(self.dataset, self.n_examples)
		]
		dataloaders = [
			torch.utils.data.DataLoader(d, batch_size=self.batch_size, collate_fn=pad_batch)
			for d in comp_datasets
		]
		
		total_kl_divs = [torch.Tensor([0.]).to(self.model.device) for dataloader in dataloaders]
		kl_divss = [[] for dataloader in dataloaders]
		for kl_divs, total_kl_div, dataloader in zip(kl_divss, total_kl_divs, dataloaders):
			for batch in dataloader:
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
				
				# remove pad tokens from the overall loss
				# this retains the values for the non-pad tokens since the
				# attention mask for those is 1, while the attention mask for
				# pad tokens is 0, so it removes those values.
				kl_div *= batch_inputs['attention_mask']
				
				if return_all:
					kl_divs.append(kl_div)
				
				# get mean for each example and accumulate the total for the whole set
				kl_div = kl_div.sum(dim=-1)/batch_inputs['attention_mask'].sum(dim=-1)
				total_kl_div += kl_div.sum(dim=-1)
		
		total_kl_divs = [
			(total_kl_div/n) * scaleby 
			for total_kl_div, n, scaleby in zip(total_kl_divs, self.n_examples, self.scaleby)
		]
		
		# the average of the averages, since we want each dataset to be of equal importance
		total_kl_div = torch.sum(torch.cat(total_kl_divs))/len(total_kl_divs)
		
		if return_all:
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