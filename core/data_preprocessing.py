import re
import torch
import random
import logging
logger = logging.getLogger(__name__)

import numpy as np

from typing import Callable, Any
from collections import Counter
from transformers import AutoTokenizer, AutoModel

def pad_tensor(t: torch.Tensor, pad: int, pad_id: int = 0, dim: int = -1) -> torch.Tensor:
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
	return torch.cat([t, torch.full(size=pad_size, fill_value=pad_id, dtype=t.dtype, device=t.device)], dim=dim)

def pad_batch(batch: list[dict[str,torch.Tensor]]) -> dict[str,torch.Tensor]:
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
	
	# this ensures that the following line will find all the keys for all examples
	assert all(batch[0].keys() == ex.keys() for ex in batch)
	
	# this allows us to pad each tensor out separately, since sometimes labels
	# won't be the same length as the inputs, and we don't want to pad those
	# with zeros (they should already be padded with -100).
	max_lens = {k: max(ex[k].size(-1) for ex in batch) for k in batch[0]}
	
	# do this so we ignore the pads in the labels for loss.
	pad_ids = {k: 0 if k != 'labels' else -100 for k in batch[0]}
	batch = [{k: pad_tensor(t=ex[k], pad=max_lens[k], pad_id=pad_ids[k], dim=-1) for k in ex} for ex in batch]
	batch = {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}
	return batch

def _expand_rows(
	*args,
	expanded_lengths: torch.tensor,
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
	if any(len(l) != len(reduced) for l in args):
		raise ExpansionError(
			'Expected there to be as many numbers indicating expanded example lengths '
			'as examples in the original dataset without expansion. But there are '
			f'{[len(l) for l in args]} examples in the original inputs and {len(reduced)} '
			'values for new expanded example lengths!'
		)
	
	new_args = []
	for arg in args:
		new_args.append([])
		for i, l in enumerate(reduced):
			new_args[-1].extend([arg[i] for _ in range(l)])
	
	# ensure that the new numbers match the new number of rows in the dataset
	if any(len(l) != len(expanded_lengths) for l in new_args):
		raise ExpansionError(
			'Expected there to be as many texts, labels, and metadata entries for '
			'expanded examples as there are expanded examples. But there are '
			f'{len(expanded_lengths)} examples in the expanded dataset and '
			f'{[len(l) for l in new_args]} values for the new lists!'
		)
	
	return new_args

# the first two arguments of data_preprocessing functions must be 
# inputs and tokenizer in that order. Additional
# arguments may be added after this, and are passed in 
# `data_preprocessing_fn_kwargs`.

def identity(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
) -> dict[str,torch.Tensor]:
	'''
	Set labels to inputs as a default proprocessing function.
	If labels are already provided, return inputs unchanged.
	'''
	if not 'labels' in inputs:
		inputs['labels'] = inputs['input_ids'].detach().clone()
	
	# we want to keep the first pad token id for each example in the loss
	# since that means that the model correctly predicts EOS where it should.
	# after that, we don't care what it does.
	inputs['labels'][[x[1:] for x in torch.where(inputs['labels'] == tokenizer.pad_token_id)]] = -100
	
	return inputs

def remove_non_mask_tokens_from_labels(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
) -> dict[str,torch.Tensor]:
	'''
	Replaces non-mask tokens in the labels with -100,
	so they don't affect the loss. This is done in the 
	provided masking functions already, but it can be
	useful to use this if you are using a dataset
	with pre-basked mask tokens and want to compute loss
	only on those. If you are using a span denoising objective,
	you *want* loss on the non-mask tokens, since you want
	to predict the correct sentinel token for the specific
	span that was noised over.
	'''
	for inp, lab in zip(inputs['input_ids'], inputs['labels']):
		if inp.shape != lab.shape:
			raise ValueError(
				'`remove_non_mask_tokens_from_labels` is intended to be used for '
				'masked language modeling datasets with pre-baked mask tokens. '
				'This means that the number of tokens in the inputs and the labels '
				'should be the same, since the output shape must match the input '
				'shape in this task. However, at least one input was found for which '
				f"this wasn't the case: {tokenizer.convert_tokens_to_ids(inp)} != "
				f'{tokenizer.convert_tokens_to_ids(lab)}. You should revise your '
				'dataset to avoid this issue.'
			)
		
		non_mask_indices_mask = (inp != tokenizer.mask_token_id).to(inp.device)
		lab[non_mask_indices_mask] = -100
	
	return inputs

def _get_mask_indices(
	input_ids: torch.Tensor,
	tokenizer: AutoTokenizer,
	mask_prop: float = 0.15,
) -> list[torch.Tensor]:
	'''
	Helper function to get mask indices according to `mask_prop`.
	'''
	# exclude pad tokens, CLS, BOS, EOS, etc.
	special_token_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens))
	candidate_tokens_for_masking = [
		torch.isin(ex, special_token_ids, invert=True)
			.nonzero(as_tuple=True)[0] 
		for ex in input_ids
	]
	
	masked_indices = []
	for candidates in candidate_tokens_for_masking:
		# randomly get `mask_prop` proportion of indices per example to mask.
		# mask at least one position per example (otherwise, why
		# are we including it in the dataset?)
		n_tokens_to_mask = max(1, round(candidates.shape[0] * mask_prop))
		
		# get a random set of numbers in the right shape, and argsort it.
		# this gives us a randomized list of numbers between 0 and len(candidates).
		to_mask = torch.argsort(torch.rand(candidates.shape[0]))
		
		# get the first n_tokens_to_mask indices
		to_mask = to_mask[:n_tokens_to_mask]
		
		# get the candidates at the random indices we chose of the candidate positions
		to_mask = candidates[to_mask]
		# [0] returns the sorted values, dropping their original indices.
		# ([1] contains the original indices of the sorted values, which we
		# don't care about)
		to_mask = torch.sort(to_mask)[0]
		
		masked_indices.append(to_mask)
	
	return masked_indices	

def mask_random_tokens(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
	mask_prop: float = 0.15,
	replacement_props: dict[str, float] = None
) -> dict[str,torch.Tensor]:
	'''
	Masks input examples like in the BERT pre-training regimen.
	`mask_prop` of tokens are randomly selected from all tokens
	in `inputs['input_ids']`'. By default,
	of these:
		- 80% are replaced with `tokenizer.mask_token_id`
		- 10% are replaced with a random token
		- 10% are left untouched
	
	The specific %s can be set by passing the `replacement_props`
	argument, which should have the following keys:
		- 'mask'
		- 'random'
		- 'original'
	Corresponding to the proportion of time to use the mask token, a random token,
	or the original token. The values associated with these keys
	must sum to 1.
	
	returns:
		the masked inputs with labels for each masked position (and -100
		elsewhere)
	'''
	if replacement_props is None:
		replacement_props = {
			'mask': 0.8,
			'random': 0.1,
			'original': 0.1
		}
	
	if sum(replacement_props.values()) != 1:
		raise ValueError(
			f'The replacement proportions must sum to 1 {replacement_props!r}.'
		)
	
	# replace the proportions with cumulative sum
	cumsum = np.cumsum(list(replacement_props.values()))
	replacement_threshes = {k: float(cumsum[i]) for i, k in enumerate(replacement_props)}
	
	# replace masked positions in inputs with mask tokens
	masked_indices = _get_mask_indices(
		input_ids=inputs['input_ids'],
		tokenizer=tokenizer,
		mask_prop=mask_prop
	)
	masked_input_ids = inputs['input_ids'].detach().clone()
	labels = masked_input_ids.detach().clone() if not 'labels' in inputs else inputs['labels']
	for indices, input_ids, label in zip(masked_indices, masked_input_ids, labels):
		
		# find the indices we want to replace in the labels
		# to ignore for CE loss, and replace them with a dummy value
		label_mask = torch.ones(label.shape, dtype=bool)
		
		for index in indices:
			label_mask[index] = False
			# get a random value, and then mask according to what it is
			r = np.random.random()
			for k in reversed(replacement_threshes):
				if r < replacement_threshes[k]:
					match k:
						case 'mask':
							replacement = tokenizer.mask_token_id
						case 'random':
							replacement = np.random.choice(len(tokenizer))
						case 'original':
							replacement = input_ids[index]
						case _:
							raise ValueError(
								f'Unknown key provided in replacement_props {k!r}.'
							)
					
					input_ids[index] = replacement
		
		label[label_mask] = -100
	
	return_inputs = {
		'input_ids': masked_input_ids.to(inputs['input_ids'].device),
		'labels': labels.to(inputs['input_ids'].device),
	}
	return_inputs.update({k: v for k, v in inputs.items() if k not in ['input_ids', 'labels']})
	
	return return_inputs

def mask_random_spans(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
	mask_prop: float = 0.15,
	mean_noise_span_length: float = 3.,
) -> dict[str,torch.Tensor]:
	'''
	Masks tokens according to the T5 pre-training task.
	Randomly selects `mask_prop` of tokens (at least 1 token)
	per example, and sets them to be masked. Consecutive mask
	tokens are replaced with a single sentinel token.
	
	Mean span length determines the average length of a masked
	span. Note that mask tokens are not assigned to places in 
	the input sequence truly randomly, but in a way consistent 
	with Google's implementation of T5's pretrained objective. 
	See https://github.com/google-research/text-to-text-transfer-
	transformer/blob/d72bd861de901d3269f45ec33c6ca6acd18b10b8/
	t5/data/preprocessors.py#L1864, `random_spans_noise_mask`.
	Essentially, inputs are always guaranteed to start with
	a non-noised span, and the average length of a noise span is 
	mean_noise_span_length.
	
	If you want to truly randomly pick mask_prop of tokens to mask,
	and then collapse adjacent positions into a single sentinel,
	use `mask_truerandom_spans`.
	
	If labels are provided, they are returned unchanged. If not,
	the labels consist of each sentinel token, followed by the
	sequence replaced by mask tokens, followed by a final sentinel
	token.
	'''
	special_token_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)).int()
	
	def _get_noise_span_ranges(
		inp: torch.Tensor,
		mask_prop: float,
		mean_noise_span_length: float,
	) -> list[list[int]]:
		'''
		Randomly gets span ranges to mask with spans according
		to T5's pre-training objective for each input in input
		ids. Does not consider masking special tokens.
		'''
		# exclude pad tokens, CLS, BOS, EOS, etc.
		non_special_tokens = torch.isin(inp, special_token_ids, invert=True).nonzero(as_tuple=True)[0]
		starting_offset = [t not in special_token_ids for t in inp].index(True)
		
		# add one because we want the number of tokens instead of the range
		# between them.
		length = max(non_special_tokens).item() - min(non_special_tokens).item() + 1
		
		# to avoid degeneracy, we want all examples to have
		# at least two tokens: one mask and one non-mask.
		# (just a mask token wouldn't give us any basis for
		# prediction, so the result would be meaningless).
		length = max(length, 2)
		
		num_noise_tokens = int(round(length * mask_prop))
		num_noise_spans = int(round(num_noise_tokens)/mean_noise_span_length)
		
		# to avoid degeneracy, we want to mask at least one span
		# (all inputs should contribute to the loss somehow).
		num_noise_spans = max(num_noise_spans, 1)
		num_nonnoise_tokens = length - num_noise_tokens
		
		def _random_segmentation(num_items: int, num_segments: int) -> list[int]:
			'''
			Randomly segments a list of num_items into num_segments, and
			returns the starting indices (excluding 0) of each segment.
			'''
			first_in_segment = np.pad([int(i < num_segments - 1) for i in range(num_items - 1)], [[1, 0]])
			random.shuffle(first_in_segment)
			segment_id = np.cumsum(first_in_segment)
			segment_length = list(Counter(segment_id).values())
			return segment_length
		
		noise_span_lengths = _random_segmentation(
			num_items=num_noise_tokens, num_segments=num_noise_spans,
		)
		nonnoise_span_lengths = _random_segmentation(
			num_items=num_nonnoise_tokens, num_segments=num_noise_spans,
		)
		interleaved_span_lengths = [x for v in zip(nonnoise_span_lengths, noise_span_lengths) for x in v]
		interleaved_span_lengths = [l + starting_offset for l in interleaved_span_lengths]
		
		span_starts = np.cumsum(interleaved_span_lengths)[:-1].tolist()
		span_ranges = []
		prev_start = 0 + starting_offset
		for start in span_starts:
			span_ranges.append([prev_start, start - 1])
			prev_start = start
		
		span_ranges.append([span_starts[-1], length + starting_offset])
		
		# odd indices are the noise ranges
		noise_spans = [span_range for i, span_range in enumerate(span_ranges) if i % 2 == 1]
		return noise_spans
	
	sentinel_token_ids = tokenizer.convert_tokens_to_ids(
		tokenizer.additional_special_tokens
	)
	
	# we need to build the labels as we mask
	# if labels are provided, we'll return those.
	# otherwise, we'll return these.
	new_labels = []
	for inp in inputs['input_ids']:
		# for the first pass, we just replace each token in each span
		# with the same sentinel token, without collapsing them into
		# one.
		new_labels.append([])
		noise_spans = _get_noise_span_ranges(
			inp=inp, mask_prop=mask_prop, mean_noise_span_length=mean_noise_span_length
		)
		for (min_idx, max_idx), sentinel in zip(noise_spans, sentinel_token_ids):
			new_labels[-1].append(sentinel)
			new_labels[-1].extend(inp[min_idx:max_idx].tolist())
			inp[min_idx:max_idx] = sentinel
		
		new_labels[-1].append(sentinel_token_ids[sentinel_token_ids.index(sentinel)+1])
	
	# now, we collapse the adjacent mask span tokens into a single token
	collapsed = []
	for inp in inputs['input_ids']:
		collapsed.append([])
		for j, token in enumerate(inp):
			# can't check the previous token if we're at the beginning
			if j == 0:
				collapsed[-1].append(token.item())
				continue
			
			# if the previous token is a sentinel and the current
			# one is a sentinel, skip it
			if token in sentinel_token_ids and collapsed[-1][-1] in sentinel_token_ids:
				continue
			
			# skip pad tokens, we'll re-add them later
			if token == tokenizer.pad_token_id:
				continue
			
			collapsed[-1].append(token.item())
	
	return_inputs = {}
	# we need to rebuild this to match the new input lengths
	if 'attention_mask' in inputs:
		max_len = max(len(ids) for ids in collapsed)
		new_attention_mask = [[1] * len(l) for l in collapsed]
		new_attention_mask = [l + ([0] * (max_len - len(l))) for l in new_attention_mask]
		return_inputs['attention_mask'] = torch.stack([
			torch.tensor(mask, dtype=int) for mask in new_attention_mask
		])
	
	# we need to pad these so they're the same length before making them tensors again.
	max_len = max(len(ids) for ids in collapsed)
	collapsed = [l + ([tokenizer.pad_token_id] * (max_len - len(l))) for l in collapsed]
	masked_input_ids = torch.stack([torch.tensor(ids, dtype=int) for ids in collapsed])
	
	max_len = max(len(ids) for ids in new_labels)
	new_labels = [l + ([-100] * (max_len - len(l))) for l in new_labels]
	new_labels = torch.stack([torch.tensor(ids, dtype=int) for ids in new_labels])
	
	return_inputs.update({
		'input_ids': masked_input_ids,
		'labels': new_labels if 'labels' not in inputs else inputs['labels'],
	})
	return_inputs.update({
		k: v for k, v in inputs.items() if k not in ['input_ids', 'labels', 'attention_mask']
	})
	return_inputs = {
		k: v.to(inputs['input_ids'].device) 
		for k, v in return_inputs.items() if isinstance(v, torch.Tensor)
	}
	
	return return_inputs

def mask_truerandom_spans(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
	mask_prop: float = 0.15,
) -> dict[str,torch.Tensor]:
	'''
	Randomly selects `mask_prop` of tokens (at least 1 token)
	per example, and sets them to be masked. Consecutive mask
	tokens are replaced with a single sentinel token.
	
	Note that this is not entirely consistent with how T5's
	pretraining objective is originally implemented. For that,
	use `mask_span`. See https://github.com/google-research/
	text-to-text-transfer-transformer/blob/d72bd861de901d326
	9f45ec33c6ca6acd18b10b8/t5/data/preprocessors.py#L1864, 
	`random_spans_noise_mask`.
	
	If labels are provided, they are returned unchanged. If not,
	the labels consist of each sentinel token, followed by the
	sequence replaced by mask tokens, followed by a final sentinel
	token.
	'''
	masked_indices = _get_mask_indices(
		tokenizer=tokenizer, 
		input_ids=inputs['input_ids'], 
		mask_prop=mask_prop
	)
	sentinel_token_ids = tokenizer.convert_tokens_to_ids(
		tokenizer.additional_special_tokens
	)
	
	masked_input_ids = inputs['input_ids'].clone().detach()
	# we need to build the labels as we mask
	# if labels are provided, we'll return those.
	# otherwise, we'll return these.
	new_labels = []
	for i, (indices, input_ids) in enumerate(zip(masked_indices, masked_input_ids)):
		# for the first pass, we just replace indices with mask span tokens
		# if the previous index has a mask span token, we use the same one
		current_sentinel_num = 0
		current_sentinel = sentinel_token_ids[current_sentinel_num]
		new_labels.append([])
		for index in indices:
			# can't check the previous position if we're at 0
			if index == 0:
				new_labels[i].extend([current_sentinel, input_ids[index].item()])
				indices[index] = current_sentinel
				continue
			
			if input_ids[index - 1] != current_sentinel:
				current_sentinel_num += 1
				current_sentinel = sentinel_token_ids[current_sentinel_num]
				new_labels[i].append(current_sentinel)
			
			new_labels[i].append(input_ids[index].item())
			input_ids[index] = current_sentinel
		
		new_labels[i].append(sentinel_token_ids[current_sentinel_num + 1])
	
	# now, we collapse the mask span tokens into a single token
	collapsed = []
	for inp in masked_input_ids:
		collapsed.append([])
		for j, token in enumerate(inp):
			# can't check the previous token if we're at the beginning
			if j == 0:
				collapsed[-1].append(token.item())
				continue
			
			# if the previous token is a sentinel and the current
			# one is a sentinel, skip it
			if token in sentinel_token_ids and collapsed[-1][-1] in sentinel_token_ids:
				continue
			
			# skip pad tokens, we'll re-add them later
			if token == tokenizer.pad_token_id:
				continue
			
			collapsed[i].append(token.item())
	
	return_inputs = {}
	# we need to rebuild this to match the new inputs
	if 'attention_mask' in inputs:
		max_len = max(len(ids) for ids in collapsed)
		new_attention_mask = [[1] * len(l) for l in collapsed]
		new_attention_mask = [l + ([0] * (max_len - len(l))) for l in new_attention_mask]
		return_inputs['attention_mask'] = torch.stack([
			torch.tensor(mask, dtype=int) for mask in new_attention_mask
		])
	
	# we need to pad these so they're the same length before making them tensors again.
	max_len = max(len(ids) for ids in collapsed)
	collapsed = [l + ([tokenizer.pad_token_id] * (max_len - len(l))) for l in collapsed]
	masked_input_ids = torch.stack([torch.tensor(ids, dtype=int) for ids in collapsed])
	
	max_len = max(len(ids) for ids in new_labels)
	new_labels = [l + ([-100] * (max_len - len(l))) for l in new_labels]
	new_labels = torch.stack([torch.tensor(ids, dtype=int) for ids in new_labels])
	
	return_inputs.update({
		'input_ids': masked_input_ids,
		'labels': new_labels if 'labels' not in inputs else inputs['labels'],
	})
	return_inputs.update({
		k: v for k, v in inputs.items() if k not in ['input_ids', 'labels', 'attention_mask']
	})
	return_inputs = {
		k: v.to(inputs['input_ids'].device) 
		for k, v in return_inputs.items() if isinstance(v, torch.Tensor)
	}
	
	return return_inputs

def expand_with_masks(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
) -> dict[str,torch.Tensor]:
	'''
	Expands a batch of inputs by generating a new set of inputs
	where each token is replaced, in order, with a mask token.
	E.g.,:
		[['Look', 'at', 'this']]
	
	becomes
		[['[MASK]', 'at', 'this'],
		 ['Look', '[MASK]', 'this'],
		 ['Look', 'at', '[MASK]']]
	
	[MASK] is not used to replace any special tokens in the input
	(e.g., [PAD], [CLS], [BOS], etc.).
	
	If a model with no mask token is passed, assume
	we are dealing with a T5 model and format according
	to the T5 pre-training objective, with one token
	masked by the sentinel token at a time.
	'''
	special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)
	if tokenizer.mask_token_id is not None:
		mask_type = 'bert'
		special_token_ids = [t for t in special_token_ids if t != tokenizer.mask_token_id]
		mask_token_id = tokenizer.mask_token_id
	else:
		mask_type = 'span_denoising'
		sentinel_tokens = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
		special_token_ids = [
			t for t in special_token_ids 
			if t not in sentinel_tokens
		]
		mask_token_id = sentinel_tokens[0]
	
	# this expands a single example with masks
	def _expand_with_masks(example: dict[str,torch.Tensor]) -> torch.Tensor:
		'''
		Expands a single example to a list of examples, constructed
		by replacing each non-special token with the mask token id.
		This allows us to get surprisal for each position in the sentence from
		a masked language model without telling it what the token in that
		position is.
		'''
		ignore_mask = [t not in special_token_ids for t in example['input_ids']]
		expanded_length = sum(ignore_mask)
		
		# expand all tensors in the example
		expanded = dict()
		for k in example:
			expanded[k] = example[k].expand(1, example[k].shape[0]).repeat(expanded_length, 1)
		
		# add labels if they're not already there,
		# before we overwrite the input ids with the
		# mask token
		had_labels_already = True
		if 'labels' not in expanded:
			had_labels_already = False
			if mask_type == 'bert':
				expanded['labels'] = expanded['input_ids'].detach().clone()
		
		# set this so that we know which input ids correspond to the 
		# original example. We'll use this later to tell us how much
		# to expand the labels, text, and metadata of each example.
		expanded['expanded_length'] = expanded_length
		
		# get the first non-special token in our input as the starting
		# position
		starting_id = ignore_mask.index(True)
		
		token_index = starting_id
		if mask_type == 'span_denoising':
			labels = []
		
		for inp in expanded['input_ids']:
			# we need to use a sentinel variable here (token_index)
			# instead of enumerating and using the position index
			# since we don't want to mask the position corresponding
			# to the row we're in (which enumerate would give),
			# but the position corresponding to an offset from the
			# first non-special token position (the starting_id).
			index_to_mask = token_index		
			if inp[index_to_mask] not in special_token_ids:
				# construct the t5 style labels before replacing
				# the target with the mask token: 
				# pad token, mask span, target, mask span.
				if mask_type == 'span_denoising':
					labels.append([
						tokenizer.pad_token_id, mask_token_id, 
						inp[index_to_mask].item(), sentinel_tokens[sentinel_tokens.index(mask_token_id)+1]
					])
				
				inp[index_to_mask] = mask_token_id
			
			token_index += 1
		
		# remove all but the mask token from the labels for the loss,
		# if we didn't already have labels that were passed.
		if not had_labels_already:
			if mask_type == 'bert':
				non_mask_indices_mask = (expanded['input_ids'] != tokenizer.mask_token_id).to(expanded['input_ids'].device)
				expanded['labels'][non_mask_indices_mask] = -100
			else:
				# we know these should all be the same length: 4.
				# that's because of how the span denoising objective
				# is defined, and the fact that we are only masking
				# a single token at a time. We don't need to do any
				# padding here or removing labels for the loss in this
				# case since we know the length.
				expanded['labels'] = torch.tensor(labels, dtype=int)
		
		# fill up the rest of this with -1. We'll later use the -1
		# as a signal value to tell us where each example starts so
		# we can line them up with the original texts, labels, and metadata
		# -1 is safe since no length can be < 0
		expanded['expanded_length'] = torch.tensor([expanded_length] + ([-1] * (expanded_length - 1)), dtype=int)
		
		return {
			k: v.to(example['input_ids'].device) 
			if isinstance(v, torch.Tensor) else v 
			for k, v in expanded.items()
		}
	
	# we'll use the original example number later to tell us
	# how many times each metadata, input text, and input label
	# should be duplicated.
	expanded = {k: [] for k in list(inputs.keys()) + ['expanded_length']}
	
	if 'labels' not in inputs:
		expanded['labels'] = []
	
	# loop through each row. since this is a dict
	# of lists, we need to deparse the values and zip
	# them, and them pack them back into a dict to
	# send to the inner function
	for i, example in enumerate(zip(*inputs.values())):
		example = dict(zip(inputs.keys(), example))
		example_expanded = _expand_with_masks(example=example)
		for k in expanded:
			expanded[k].extend(example_expanded[k])
	
	for k in expanded:
		expanded[k] = torch.stack(expanded[k])
	
	return expanded

def mask_word_tokens(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
	word_tokens_to_mask: list[str],
) -> dict[str,torch.Tensor]:
	'''
	Masks the specific word tokens in the dataset,
	and removes non-mask tokens from the labels
	(if labels are not already provided)
	using a masked language model.
	
	If a T5 model, masks the specific words
	using consecutive sentinel tokens, and
	constructs labels according to T5's pre-training
	span denoising objective.
	
	If you want to mask words that may contain
	multiple tokens, use `mask_words` instead.
	'''
	if not isinstance(word_tokens_to_mask, list):
		word_tokens_to_mask = [word_tokens_to_mask]
	
	try:
		tokens_to_mask = tokenizer.convert_tokens_to_ids([
			t[0] for t in [
				tokenizer.tokenize(w) for w in word_tokens_to_mask
			] 
			if len(t) == 1 and re.search(r'\w', t[0])
		])
	except TypeError as e:
		# if we want to support this later, we need to check
		# for contiguous subsequences of tokens, which is more
		# difficult. Not impossible, but we can implement
		# this later if we need it.
		raise TypeError(
			f'At least one word in {words_to_mask} has more than one '
			'token in it. Use `mask_words` instead if this is what you '
			'want.'
		) from e
		
	assert not any(t == tokenizer.unk_token_id for t in tokens_to_mask)
	
	if tokenizer.mask_token_id is not None:
		# we're doing bert-style masking here.
		# this one is easy since we have one mask
		# token and we don't need to programmatically
		# construct the labels.
		
		# add labels if we don't have them.
		had_labels_already = True
		if 'labels' not in inputs:
			had_labels_already = False
			inputs['labels'] = inputs['input_ids'].detach().clone()
		
		idx = [(inputs['input_ids'] == x) for x in tokens_to_mask]
		idx = reduce(lambda x, y: torch.logical_or(x, y), idx)
		inputs['input_ids'][idx] = tokenizer.mask_token_id
		
		# fix up the labels so that only the mask indices
		# are retained for loss.
		if not had_labels_already:
			# whereever the input doesn't now have a mask token,
			# replace it with the "ignore_loss" value.
			inputs['labels'][~idx] = -100
		
		return inputs
	
	# if we're here, we're doing T5-style masking.
	# we need to progressively use increasing sentinels
	# to mark mask positions, and construct new labels.
	sentinel_tokens = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
	new_labels = []
	for inp in inputs['input_ids']:
		current_sentinel_idx = 0
		new_labels.append([tokenizer.pad_token_id])
		for i, tok in enumerate(inp):
			if tok in tokens_to_mask:
				new_labels[-1].append(tok.item())
				inp[i] = sentinel_tokens[current_sentinel_idx]
				current_sentinel_idx += 1
		
		new_labels[-1].append(sentinel_tokens[current_sentinel_idx])
	
	if 'labels' not in inputs:
		# get all labels to the same length. They might be
		# different lengths if we have different numbers of
		# the masked tokens in different inputs.
		max_len_label = max(len(l) for l in new_labels)
		new_labels = [l + ([-100] * (max_len_label - len(l))) for l in new_labels]
		new_labels = torch.stack(
			[torch.tensor(ids, dtype=int) for ids in new_labels]
		).to(inputs['input_ids'].device)
		inputs['labels'] = new_labels
	
	return inputs

def mask_words(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
	words_to_mask: list[str],
) -> dict[str,torch.Tensor]:
	'''
	Masks the words in the dataset,
	and removes non-masked tokens from the labels
	(if labels are not already provided)
	using a masked language model.
	
	If using a MLM, replaces each token in 
	each word with a separate [MASK] token,
	and removes non-mask tokens from labels
	if labels are not already provided.
	
	If using a T5 model, replaces each word
	with a single sentinel token, and adds
	all tokens of the word after the sentinel
	token in the labels (if labels are not 
	provided).
	
	If you want to ensure that masked words
	constitute a single token, use `mask_word_tokens`
	instead.
	'''
	if not isinstance(words_to_mask, list):
		words_to_mask = [words_to_mask]
	
	token_sequences_to_mask = []
	for word in words_to_mask:
		# the regex would ensure that we don't include 
		# non-word tokens that sometimes arise as
		# artifacts in certain tokenizers (like the
		# '__' special underscore). But on closer
		# thought, we probably want that if we're using
		# this function.
		token_sequence = tokenizer.convert_tokens_to_ids(
			[t for t in tokenizer.tokenize(word)] # if re.search(r'\w', t)]
		)
		assert not any(t == tokenizer.unk_token_id for t in token_sequence)
		token_sequences_to_mask.append(token_sequence)
	
	# if what we have are all single-token words,
	# we can just fall back to the simple case here.
	if all(len(seq) == 1 for seq in token_sequences_to_mask):
		return mask_word_tokens(
			inputs=inputs, tokenizer=tokenizer, word_tokens_to_mask=words_to_mask
		)
	
	# we need to check longer sequences first, so that we
	# don't end up replacing part of a longer word with
	# masks and then blocking it from being checked later.
	# if we sort so that longer sequences are first, we
	# avoid this problem.
	token_sequences_to_mask = sorted(
		token_sequences_to_mask, key=lambda l: len(l), reverse=True
	)
	
	if tokenizer.mask_token_id is not None:
		# we're doing bert-style masking here.
		# We need to find the contiguous sequences
		# that match any of our token sequences
		# and replace each token with a mask token.
		# add labels if we don't have them.
		
		# add labels if we don't have them.
		had_labels_already = True
		if 'labels' not in inputs:
			had_labels_already = False
			inputs['labels'] = inputs['input_ids'].detach().clone()
		
		for inp in inputs['input_ids']:
			for i, tok in enumerate(inp):
				for seq in token_sequences_to_mask:
					# don't waste time checking if 
					# the remaining space isn't enough
					if i + len(seq) > len(inp):
						continue
					
					if torch.all(inp[i:i+len(seq)] == torch.tensor(seq).to(inp.device)):
						inp[i:i+len(seq)] = tokenizer.mask_token_id
		
		# fix up the labels so that only the mask indices
		# are retained for loss.
		if not had_labels_already:
			# whereever the input doesn't now have a mask token,
			# replace it with the "ignore_loss" value.
			inputs['labels'][inputs['input_ids'] != tokenizer.mask_token_id] = -100
		
		return inputs
	
	# if we're here, we're doing T5-style masking.
	# we need to replace all tokens in each word sequence
	# with a single sentinel token, replace subsequent
	# words with the next sentinel token, repad, and
	# construct new labels. We also need to rebuild
	# the attention mask.
	
	# in the first pass, just find the spans we want to mask,
	# and accumulate their start and ending positions.
	spans_to_mask = []
	for i, inp in enumerate(inputs['input_ids']):
		spans_to_mask.append([])
		for j, tok in enumerate(inp):
			for seq in token_sequences_to_mask:
				# don't waste time checking this one
				# if the remaining space isn't enough
				if j + len(seq) > len(inp):
					continue
				
				if torch.all(inp[j:(j + len(seq))] == torch.tensor(seq).to(inp.device)):
					# start, end
					spans_to_mask[-1].extend([j, j + len(seq)])
	
	# collapse adjacent spans so that we replace them
	# with a single sentinel. Since these are increasing
	# indices, duplicates will be removed in pairs,
	# and this will collapsed adjacent spans into one.
	spans_to_mask = [[p for p in span if span.count(p) < 2] for span in spans_to_mask]
	spans_to_mask = [list(batched(span, n=2)) for span in spans_to_mask]
	
	# now we have a list that for each input contains
	# the non-contiguous start and end positions of
	# each span. Replace each token of those spans 
	# with the sentinel token and construct the labels.
	sentinel_token_ids = tokenizer.convert_tokens_to_ids(
		tokenizer.additional_special_tokens
	)
	new_labels = []
	for inp, span_list in zip(inputs['input_ids'], spans_to_mask):
		new_labels.append([])
		current_sentinel_num = 0
		current_sentinel = sentinel_token_ids[current_sentinel_num]
		for start, end in span_list:
			new_labels[-1].append(current_sentinel)
			new_labels[-1].extend(inp[start:end].tolist())
			inp[start:end] = current_sentinel
			current_sentinel_num += 1
			current_sentinel = sentinel_token_ids[current_sentinel_num]
		
		new_labels[-1].append(current_sentinel)
	
	# now, we collapse the adjacent mask span tokens into a single token
	collapsed = []
	for inp in inputs['input_ids']:
		collapsed.append([])
		for j, token in enumerate(inp):
			# can't check the previous token if we're at the beginning
			if j == 0:
				collapsed[-1].append(token.item())
				continue
			
			# if the previous token is a sentinel and the current
			# one is a sentinel, skip it
			if token in sentinel_token_ids and collapsed[-1][-1] in sentinel_token_ids:
				continue
			
			# skip pad tokens, we'll re-add them later once
			# we know the new max length.
			if token == tokenizer.pad_token_id:
				continue
			
			collapsed[-1].append(token.item())
	
	return_inputs = {}
	# we need to rebuild this to match the new inputs
	if 'attention_mask' in inputs:
		max_len = max(len(ids) for ids in collapsed)
		new_attention_mask = [[1] * len(l) for l in collapsed]
		new_attention_mask = [l + ([0] * (max_len - len(l))) for l in new_attention_mask]
		return_inputs['attention_mask'] = torch.stack([
			torch.tensor(mask, dtype=int) for mask in new_attention_mask
		])
	
	# we need to pad these so they're the same length before making them tensors again.
	max_len = max(len(ids) for ids in collapsed)
	collapsed = [l + ([tokenizer.pad_token_id] * (max_len - len(l))) for l in collapsed]
	masked_input_ids = torch.stack([torch.tensor(ids, dtype=int) for ids in collapsed])
	
	max_len = max(len(ids) for ids in new_labels)
	new_labels = [l + ([-100] * (max_len - len(l))) for l in new_labels]
	new_labels = torch.stack([torch.tensor(ids, dtype=int) for ids in new_labels])
	
	return_inputs.update({
		'input_ids': masked_input_ids,
		'labels': new_labels if 'labels' not in inputs else inputs['labels'],
	})
	return_inputs.update({
		k: v for k, v in inputs.items() if k not in ['input_ids', 'labels', 'attention_mask']
	})
	return_inputs = {
		k: v.to(inputs['input_ids'].device) 
		for k, v in return_inputs.items() if isinstance(v, torch.Tensor)
	}
	
	return return_inputs

# these are functions that require
# us to update the labels after
# being applied. this takes place
# in the dataset constructor after
# the function is called, as long
# as the function is registered
# here.
UPDATE_LABELS_FNS: set[Callable[[dict[str,torch.Tensor],AutoTokenizer,Any],dict[str,torch.Tensor]]] = {
	mask_random_spans,
	mask_truerandom_spans,
}

def identity_if_decoder_mask_if_encoder(
	inputs: dict[str,torch.Tensor],
	model: AutoModel,
	tokenizer: AutoTokenizer,
	mask_fn: callable = mask_random_tokens,
	mask_fn_kwargs: dict = None,
) -> dict[str,torch.Tensor]:
	'''
	If the models' `model.config.is_decoder`
	attribute is set to false, calls mask_fn
	with inputs, model, tokenizer, and mask_kwargs,
	and returns the result.
	
	If the models' `model.config.is_decoder`
	attribute is True, calls identity with
	inputs, model, tokenizer, and mask_kwargs,
	and returns the result.
	'''
	if model.config.is_decoder:
		try:
			UPDATE_LABELS_FNS.remove(identity_if_decoder_mask_if_encoder)
		except KeyError:
			pass
		
		return identity(inputs=inputs, model=model, tokenizer=tokenizer)
	
	if not model.config.is_decoder:
		if mask_fn in UPDATE_LABELS_FNS:
			UPDATE_LABELS_FNS.update({identity_if_decoder_mask_if_encoder})
		
		mask_fn_kwargs = mask_fn_kwargs if mask_fn_kwargs else {}
		return mask_fn(inputs=inputs, model=model, tokenizer=tokenizer, **mask_fn_kwargs)
