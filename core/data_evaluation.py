import re
import torch
import torch.nn.functional as F

from typing import Callable
from constants import *
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	AutoModelForMaskedLM,
	AutoModelForSeq2SeqLM,
)
from transformers.utils.generic import ModelOutput

def align_words_to_subword_tokens(
	tokenizer: AutoTokenizer, 
	words: list[str], 
	tokens: list[int]
) -> list[list[int]]:
	'''
	Aligns words to subword tokens.
	
	params:
		tokenizer: AutoTokenizer: the tokenizer used to generate `tokens`
		words (list[str]): a list of words to align
		tokens (list[int]): a list of tokens generated from the sequence of words.
	
	returns:
		list of list of ints, which is the same length as `words`.
		Each sublist contains the tokens corresponding to the word
		at the same position as the sublist in `words`.	
	
	raises:
		IndexError, AssertionError: if the words and tokens cannot be aligned.
	'''
	# pop works backward, so reverse first
	num_words = len(words)
	tokens = tokens[::-1]
	words = words[::-1]
	
	# these should be in every tokenizer that has cased letters,
	# so we can use them to test
	uncased = tokenizer.tokenize('A') == tokenizer.tokenize('a')
	if uncased:
		# sometimes uncased tokenizer have uppercase special tokens, so we need
		# to deal with that and NOT uncase special tokens if they're already
		# in the inputs.
		if any(t in w for t in tokenizer.all_special_tokens for w in words):
			special_words = []
			for w in words:
				# this handles a case where we have a word without a special
				# token and still want to lower case it.
				if not any(t in w for t in tokenizer.all_special_tokens):
					w = w.lower()
				
				for t in tokenizer.all_special_tokens:
					if t in w:
						w = w.split(t)
						# this is needed in case we have multiple special tokens in the same word
						w = [part.lower() for part in w if not part in tokenizer.all_special_tokens]
						w = t.join(w)
					
				special_words.append(w)
			
			words = special_words
		else:
			words = [w.lower() for w in words]
	
	aligned = []
	while tokens:
		aligned_tokens = [tokens.pop()]
		# some tokenizers apparently have tokens that evaluate
		# to an empty string. that won't show up in the word list,
		# of course (they only appear during open-ended generation).
		# Let's never consider these to be part of a word, so we'll 
		# just continue on and grab the next word in the next iteration
		if tokenizer.decode(aligned_tokens[-1]) == '':
			aligned.append(aligned_tokens)
			continue
		
		word = words.pop()
		
		# we need to replace all spaces here rather than
		# just stripping because some tokenizers don't handle
		# words with punctuation in the middle correctly
		# e.g, 'bert-large-cased' tokenizes 're-wrapped' as
		# [1231, 118, 4293], but decodes that sequence as
		# 're - wrapped', with spaces in the middle.
		if 'babyt5' in tokenizer.name_or_path:
			# babyt5 doesn't tokenize commas correctly, but
			# as its <unk> token. in general, an <unk> token
			# should not be used to identify a word, since
			# not all <unk> tokens have the same source.
			# in this case, we build in a very specific hack.
			# we don't want a more general solution, since
			# that could mask an actually problematic case
			while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != re.sub('[0-9,]', tokenizer.unk_token, word):
				aligned_tokens += [tokens.pop()]
		else:
			while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != word:
				aligned_tokens += [tokens.pop()]
		
		aligned.append(aligned_tokens)
	
	assert len([l for l in aligned if tokenizer.decode(l) != '']) == num_words, (
		f'Unable to find {num_words} words in text.'
	)
	
	return aligned

def get_model_task(model_name_or_path: str) -> str:
	'''Returns the model task based on the name.'''
	# for gpt-bert
	if model_name_or_path in NEXT_WORD_MODELS and model_name_or_path in MASKED_LANGUAGE_MODELS:
		return 'LM+MLM'
	
	if model_name_or_path in NEXT_WORD_MODELS:
		return 'LM'
	
	if model_name_or_path in MASKED_LANGUAGE_MODELS:
		return 'MLM'
	
	if model_name_or_path in SEQ2SEQ_MODELS:
		return 'Seq2Seq'
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def get_model_eval_function(model_name_or_path: str) -> Callable:
	'''
	Returns the appropriate function for eval based on the kind of 
	model.
	'''
	# for gpt-bert. We'll just get the surprisal for each token, like
	# causal lm.
	if get_model_task(model_name_or_path=model_name_or_path) == 'LM+MLM':
		return evaluate_LM_batch
	
	if get_model_task(model_name_or_path=model_name_or_path) == 'LM':
		return evaluate_LM_batch
	
	if get_model_task(model_name_or_path=model_name_or_path) == 'MLM':
		return evaluate_MLM_batch
	
	if get_model_task(model_name_or_path=model_name_or_path) == 'Seq2Seq':
		return evaluate_S2S_batch
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def tokenize_texts(tokenizer: AutoTokenizer, text: list[str]) -> list[list[int]]:
	'''
	Tokenize a list of examples without special tokens for use during evaluation.
	'''
	tokenized = tokenizer(text, add_special_tokens=False)['input_ids']
	return tokenized

def evaluate_batch(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	batch_outputs: ModelOutput,
	input_nums: list[int] = None,
	batch_metadata: list[dict] = None,
	**additional_metadata
) -> list[dict]:
	'''Record metrics for a single batch of inputs, depending on the model type.'''
	if input_nums is None:
		input_nums = range(len(inputs.get('labels', inputs['input_ids']).shape[0]))
	
	if batch_metadata is None:
		batch_metadata = {}
	
	model_eval_function = get_model_eval_function(model_name_or_path=model.name_or_path)
	
	return model_eval_function(
		model=model, 
		tokenizer=tokenizer, 
		inputs=inputs, 
		input_texts=input_texts,
		input_labels=input_labels,
		input_nums=input_nums, 
		batch_outputs=batch_outputs,
		batch_metadata=batch_metadata,
		additional_metadata=additional_metadata,
	)

def evaluate_LM_batch(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	input_nums: list[int],
	batch_outputs: 'CausalLMOutput',
	batch_metadata: list[dict],
	additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a batch of examples for a Language Model.
	For each input, determines the surprisal of each eval token
	as a prediction for the next token.
	'''
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs.logits, dim=-1)
	
	next_word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	
	metrics = []
	records = zip(
		input_nums, input_texts, inputs['input_ids'], input_labels, 
		next_word_ids, batch_surprisals, batch_metadata,
		strict=True
	)
	for (
		input_num, input_text, input_ids, input_label, 
		next_word_tokens, surprisal, example_metadata
	) in records:
		# get this here in case the original text that was passed
		# was modified by preprocessing (i.e., lower-casing, masking, etc.)
		actual_input_text = [
			t for t in input_ids 
			if tokenizer.convert_ids_to_tokens(t.item()) not in tokenizer.all_special_tokens
		]
		actual_input_text = tokenizer.decode(actual_input_text)
		
		input_words = input_text.split()
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer, 
			words=input_words, 
			tokens=next_word_tokens
		)
		
		tokens_seen = 0
		for word_num, tokens in enumerate(aligned_tokens):
			for token_num, token in enumerate(tokens):
				predicted_token_id = torch.argmin(surprisal[tokens_seen,:], dim=-1).item()
				predicted_token = tokenizer.decode(predicted_token_id)
				metrics.extend([{
					'item': input_num,
					'original_text': input_text,
					'input_text': actual_input_text,
					'input_label': input_label,
					'word_num': word_num,
					'token_num_in_word': token_num,
					'token': tokenizer.decode(token),
					'token_id': token,
					'token_is_start_of_word': token_num == 0,
					'token_is_word': len(tokens) == 1,
					'surprisal': surprisal[tokens_seen,token].item(),
					'predicted_token': predicted_token,
					'predicted_token_id': predicted_token_id,
					'predicted_token_surprisal': surprisal[tokens_seen,predicted_token_id].item(),
					**example_metadata,
					**additional_metadata,
				}])
				tokens_seen += 1
	
	return metrics

def evaluate_MLM_batch(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	input_nums: list[int],
	batch_outputs: 'MaskedLMOutput',
	batch_metadata: list[dict],
	additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a batch of examples for a Masked Language Model.
	For each input and label, for each mask position in the input, 
	determines the surprisal of the token at that position in the label.
	'''
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs.logits, dim=-1)
	
	# these are the positions we want to get predictions for
	mask_locations = torch.nonzero(inputs['input_ids'] == tokenizer.mask_token_id, as_tuple=True)
	batch_surprisals = batch_surprisals[mask_locations]
	
	# we need to repeat the text, num, and metadata associated with each input
	# for each time a mask token occurs in that input mask_locations[0] does 
	# this, since it repeats the example number for each mask token in it.
	# this will also have the effect of excluding any sentences with no masks,
	# which is what we want.
	input_texts = [input_texts[ex_idx] for ex_idx in mask_locations[0]]
	input_labels = [input_labels[ex_idx] for ex_idx in mask_locations[0]]
	input_nums = [input_nums[ex_idx] for ex_idx in mask_locations[0]]
	batch_metadata = [batch_metadata[ex_idx] for ex_idx in mask_locations[0]]
	
	# we want to record whether this mask token is at the start of a word or not,
	# so we need to tokenize the whole sentence, and align the tokens to determine
	# this.
	word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	label_word_ids = tokenize_texts(tokenizer=tokenizer, text=input_labels)
	# this will help us align the words to the subword tokens
	# we ignore the mask token since it's the one we're getting
	# predictions for.
	special_token_ids = [
		t for t in tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens) 
		if t != tokenizer.mask_token_id
	]
	# for each input (for each mask token), tell us where to start looking for 
	# "real" (i.e., non-special) tokens.
	input_ids = [inputs['input_ids'][ex_idx] for ex_idx in mask_locations[0]]
	starting_ids = [[t not in special_token_ids for t in row].index(True) for row in input_ids]
	labels = [inputs['labels'][ex_idx] for ex_idx in mask_locations[0]]
	
	metrics = []
	records = zip(
		input_nums, input_texts, input_labels, labels, starting_ids, inputs['input_ids'],
		mask_locations[-1], word_ids, label_word_ids, batch_surprisals, batch_metadata,
		strict=True
	)
	for (
		input_num, input_text, input_label, label_ids, starting_id, input_ids,
		mask_location, word_tokens, label_tokens, surprisal, example_metadata
	) in records:
		# get the actual text the model was run on, in case it was
		# changed in preprocessing.
		actual_input_text = [
			t for t in input_ids 
			if tokenizer.convert_ids_to_tokens(t.item()) not in tokenizer.all_special_tokens or 
			t == tokenizer.mask_token_id
		]
		actual_input_text = tokenizer.decode(actual_input_text)
	
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer,
			words=input_text.split(),
			tokens=word_tokens,
		)
		aligned_label_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer,
			words=input_label.split(),
			tokens=label_tokens,
		)
		
		# make sure that for MLM, the number of tokens is the same
		# in the inputs and the labels, since this is required
		if sum(len(l) for l in aligned_tokens) != sum(len(l) for l in aligned_label_tokens):
			raise ValueError(
				'The mask token in at least one example is associated with more than one token '
				f'in the label: {input_text=} (n_tokens={sum(len(l) for l in aligned_tokens)}) '
				f'and {input_label=} (n_tokens={sum(len(l) for l in aligned_label_tokens)})!'
			)
		
		# we need to extract the aligned token set that has the token_num in it,
		# so we can determine whether it starts a word or not. Then, we need to get
		# the token at the corresponding position in the labels.
		
		# this tells us which token number starts each word
		starting_token_numbers = [
			sum(len(t) for t in aligned_tokens[:i]) for i, _ in enumerate(aligned_tokens)
		]
		
		# the aligned tokens are indexed without the special tokens,
		# so we need to adjust to find it in the starting_token_numbers,
		# by subtracting out the starting position of the non-special tokens
		starting_token_num_of_word = mask_location - starting_id
		# if the token doesn't start a word, we need to move backward till
		# we find it, so we know which token in the middle of the word
		# we want to get the surprisal for (the token at the actual
		# mask position in the word)
		while starting_token_num_of_word not in starting_token_numbers:
			starting_token_num_of_word -= 1
		
		# get the tokens corresponding to the word containing this mask token
		tokens = aligned_label_tokens[starting_token_numbers.index(starting_token_num_of_word)]
		
		# get the index of the mask token we're looking at in this word
		# subtract one since Python indexes start at 0, and we want to
		# use this as an index
		token_num_in_word = mask_location - starting_token_num_of_word - 1
		
		# get the token that we want the prediction for in the [MASK] location
		token = tokens[token_num_in_word]
		
		# get the number of the word that contains this mask token
		word_num = starting_token_numbers.index(starting_token_num_of_word.item())
		
		# get the predicted token and id, in case it's not the one
		# that's the actual target
		predicted_token_id = torch.argmin(surprisal, dim=-1).item()
		predicted_token = tokenizer.decode(predicted_token_id)
		
		metrics.append({
			'item': input_num,
			'original_text': input_text,
			'input_text': actual_input_text,
			'input_label': input_label,
			'word_num': word_num,
			'token_num_in_word': token_num_in_word.item(),
			'token': tokenizer.decode(token),
			'token_id': token,
			'token_is_start_of_word': (token_num_in_word == 0).item(),
			'token_is_word': len(tokens) == 1,
			'surprisal': surprisal[token].item(),
			'predicted_token': predicted_token,
			'predicted_token_id': predicted_token_id,
			'predicted_token_surprisal': surprisal[predicted_token_id].item(),
			**example_metadata,
			**additional_metadata,
		})
	
	return metrics

def evaluate_S2S_batch(
	model: AutoModelForSeq2SeqLM,
	tokenizer: AutoTokenizer,
	inputs: dict[str,torch.Tensor],
	input_texts: list[str],
	input_labels: list[str],
	input_nums: list[int],
	batch_outputs: 'Seq2SeqLMOutput',
	batch_metadata: list[dict],
	additional_metadata: dict,
) -> list[dict]:
	'''
	Evaluates a batch of examples for a Seq2Seq Language Model.
	For each input, determines the surprisal of each token of the
	generated sequence, as well as the surprisal of each token
	of the label corresponding to the input (using teacher-forcing).
	'''
	def fix_aligned_mask_span_tokens(
		aligned_tokens: list[list[int]], 
		single_word_tokens: list[int]
	) -> list[list[int]]:
		'''
		Ensures that each single word token is a single word.
		All other tokens will retain their relative positions
		in the aligned tokens list.
		'''
		fixed_aligned_tokens = [[]]
		for i, l in enumerate(aligned_tokens):
			for j, t in enumerate(l):
				if t in single_word_tokens:
					fixed_aligned_tokens.append([t])
					fixed_aligned_tokens.append([])
				else:
					fixed_aligned_tokens[-1].append(t)
		
		fixed_aligned_tokens = [l for l in fixed_aligned_tokens if l]
		return fixed_aligned_tokens
	
	def prefix_allowed_tokens_fn_factory(
		# we don't actually need to pass these, but we
		# need them here so that the inner function
		# can access them.
		labels: torch.Tensor = inputs['labels'], 
		pad_token_id: int = tokenizer.pad_token_id
	) -> Callable[[int, torch.Tensor], list[int]]:
		'''
		Returns a function that constrains the output generation to the label sequence.
		If we don't wrap this, then the inner function doesn't have access to the labels.
		'''
		def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
			'''
			Forces prediction of the label sequence corresponding to the
			batch we're evaluating.
			'''
			next_token = labels[batch_id][len(input_ids)].item()
			# this means we're ignoring this token for loss, so just force it
			# to be the pad token. We do this since the tokenizer won't be able
			# to decode -100.
			if next_token == -100:
				next_token = pad_token_id
			
			return [next_token]
		
		return prefix_allowed_tokens_fn
		
	# we have the batch_outputs corresponding to the generated sequence
	# already. But now we should get the outputs corresponding to the 
	# conditional probability of each token in the label sequence,
	# given the preceding tokens in that label. we do this with the
	# prefx_allow_tokens_fn and output_logits.
	with torch.no_grad():
		label_outputs = model.generate(
			inputs=inputs['input_ids'],
			prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_factory(),
			return_dict_in_generate=True,
			# allow generating only as many tokens as we need
			# to get the surprisals for the label sequence
			max_new_tokens=inputs['labels'].shape[-1] - 1,
			output_logits=True,
		)
		# transform these to be the same shape as the normal outputs
		# these will be one shorter than the batch_output logits,
		# since these are forced to start with a [0], so we don't
		# have the logits for that token.
		label_outputs.logits = torch.stack(label_outputs.logits, dim=1)
	
	surprisals = {
		'generated_sequence': batch_outputs.logits,
		'label_sequence': label_outputs.logits,
	}
	surprisals = {
		k: -(1/torch.log(torch.tensor(2.))) * F.log_softmax(logits, dim=-1) 
		for k, logits in surprisals.items()
	}
	
	metrics = []
	for generated_sequence_type, batch_surprisals in surprisals.items():
		if generated_sequence_type == 'generated_sequence':
			label_word_ids = torch.argmin(batch_surprisals, dim=-1).tolist()
			label_texts = tokenizer.batch_decode(torch.argmin(batch_surprisals, dim=-1))
		else:
			label_word_ids = inputs['labels'].detach().clone().tolist()
			
			# we need to remove the first value from this, since we don't
			# have predictions for the pad tokens for these ones
			label_word_ids = [l[1:] if l[0] == tokenizer.pad_token_id else l for l in label_word_ids]
			
			# remove this since the tokenizer won't be able to decode the 
			# "ignore_for_loss" value in the labels.
			label_word_ids = [[t for t in l if t != -100] for l in label_word_ids]
			label_texts = [re.sub(fr'^{tokenizer.pad_token}', '', l) for l in input_labels]
		
		records = zip(
			input_nums, input_texts, inputs['input_ids'], label_texts, 
			label_word_ids, batch_surprisals, batch_metadata,
			strict=True
		)
		for (
			input_num, input_text, input_ids, label_text, 
			label_word_ids, surprisal, example_metadata
		) in records:
			actual_input_text = [
				t for t in input_ids 
				if tokenizer.convert_ids_to_tokens(t.item()) not in tokenizer.all_special_tokens or 
				t in tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
			]
			actual_input_text = tokenizer.decode(actual_input_text)
			
			aligned_tokens = align_words_to_subword_tokens(
				tokenizer=tokenizer,
				words=label_text.split(),
				tokens=label_word_ids,
			)
			
			# for the mask span tokens, we need to fix this up here,
			# since we don't want to count them as word parts in the 
			# labels (even though they could be in the inputs).
			aligned_tokens = fix_aligned_mask_span_tokens(
				aligned_tokens=aligned_tokens, 
				single_word_tokens=tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens),
			)
			
			tokens_seen = 0
			for word_num, tokens in enumerate(aligned_tokens):
				for token_num, token in enumerate(tokens):
					predicted_token_id = torch.argmin(surprisal[tokens_seen,:], dim=-1).item()
					predicted_token = tokenizer.decode(predicted_token_id)
					metrics.append({
						'item': input_num,
						'original_text': input_text,
						'input_text': actual_input_text,
						'generated_sequence_type': generated_sequence_type,
						'label_text': label_text,
						'word_num': word_num,
						'token_num_in_word': token_num,
						'token': tokenizer.decode(token),
						'token_id': token,
						'token_is_start_of_word': token_num == 0,
						'token_is_word': len(tokens) == 1,
						'surprisal': surprisal[tokens_seen,token].item(),
						'predicted_token': predicted_token,
						'predicted_token_id': predicted_token_id,
						'predicted_token_surprisal': surprisal[tokens_seen,predicted_token_id].item(),
						**example_metadata,
						**additional_metadata,
					})
					tokens_seen += 1
	
	return metrics