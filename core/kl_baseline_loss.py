# Implements a loss that combines loss on the new data
# with the KL divergence between the updated model's predictions
# and the pretrained model's predictions
import logging

import numpy as np

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
# otherwise a cache file is saved for every time KL loss is run,
# which we don't want
disable_caching()
from datasets.utils import logging as dataset_utils_logging
from datasets.utils import disable_progress_bar
disable_progress_bar()
dataset_utils_logging.set_verbosity_error()

log = logging.getLogger(__name__)

def sem(x: Union[List,np.ndarray,torch.Tensor], na_rm: bool = True) -> float:
	'''
	Calculate the standard error of the mean for a list of numbers
	
		params:
			x (list) 		: a list of numbers for which to calculate the standard error of the mean
			na_rm (bool)	: exclude nas?
		
		returns:
			sem_x (float)	: the standard error of the mean of x
	'''
	namespace = torch if isinstance(x,torch.Tensor) else np
	if na_rm:
		x = [v for v in x if not namespace.isnan(v)]
		if namespace == torch:
			x = torch.tensor(x)
	
	if len(x) == 0:
		return namespace.nan
	else:	
		return namespace.std(x)/sqrt(len(x))

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
		dataset: Dataset,
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
													  as passed
				batch_size (int)					: the number of sentences to run through the models at a single time.
													  KL divergence is computed per sentence and averaged
				scaleby (float)						: returned loss is multiplied by this
				n_examples_per_batch (int)			: it may be too time-consuming to calculate the KLBaselineLoss on
													  the basis of the entire dataset, if the dataset is large.
													  you can use this to set how many random samples to draw
													  from dataset to use when calculating loss. If not set,
													  all examples will be used each time.
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
		self.baseline_model	= AutoModelForCausalLM.from_pretrained(self.model.name_or_path, **model_kwargs).to(self.device)
		
		# we're not fine-tuning this
		# _ = is to prevent printing
		_ = self.baseline_model.eval()
		
		log.debug(f'Initializing Baseline Tokenizer for KLBaselineLoss: {self.tokenizer.name_or_path}')
		self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path, **tokenizer_kwargs)
		
		# set up the dataset
		self.dataset = dataset
		
		# can't use more examples than we've got
		self.n_examples = self.dataset.num_rows if n_examples_per_batch is None else min(n_examples_per_batch, self.dataset.num_rows)
		
		self.scaleby = scaleby
		
	def forward(
		self, 
		progress_bar: bool = False,
		return_all: bool = False,
	) -> torch.Tensor:
		'''
		Computes KLBaselineLoss between the predictions of the baseline model
		and the predictions of the fine-tuned model on the basis of self.n_examples
		from self.dataset. Samples are randomized with each call.
		
			params:
				progress_bar (bool)		: whether to display a progress bar while iterating through
										  the chosen examples
				return_all (bool)		: whether to return a list containing every individual KL divergence
										  in a list in addition to the mean
			
			returns:
				kl_div (torch.Tensor)	: the mean KL divergence between the model and the baseline model
										  across n_examples of the dataset, multiplied by the scaling factor
				kl_divs (torch.Tensor)	: the individual KL divergence for each example
										  returned if return_all=True.
		'''
		# construct a comparison dataset for this call with n random examples
		comp_dataset = self.dataset.shuffle().select(range(self.n_examples))
		dataloader = enumerate(torch.utils.data.DataLoader(comp_dataset, batch_size=self.batch_size, collate_fn=pad_batch))
		
		if progress_bar:
			dataloader = tqdm(dataloader, total=num_batches)
		
		total_kl_div = torch.Tensor([0.]).to(self.model.device)
		kl_divs = []
		for i, batch in dataloader:
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
		
		if return_all:
			# pad so we can return all KL divs in a single tensor,
			# instead of a list of tensors
			max_dim = max(kl_div.shape[-1] for kl_div in kl_divs)
			kl_divs = [
				F.pad(input=kl_div, pad=(0, max_dim - kl_div.shape[-1]), value=0)
				for kl_div in kl_divs
			]
			kl_divs = torch.cat(kl_divs)
			
			return (total_kl_div/self.n_examples) * self.scaleby, kl_divs
		else:
			return (total_kl_div/self.n_examples) * self.scaleby
