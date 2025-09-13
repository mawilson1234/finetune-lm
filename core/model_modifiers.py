import os
import re
import json
import gzip
import logging

logger = logging.getLogger(__name__)

from copy import deepcopy
from dataset import Dataset
from tempfile import TemporaryDirectory
from transformers import AutoModel, AutoTokenizer

class PartiallyFrozenModelCallback:
	def __init__(
		self, model: AutoModel, tokenizer: AutoTokenizer,
		frozen_params: list[str] = None, unfrozen_params: list[str] = None
	):
		if frozen_params is None and unfrozen_params is None:
			raise ValueError(
				'At least one of `frozen_params`, `unfrozen_params` must be provided!'
			)
		
		if frozen_params is not None and unfrozen_params is not None:
			raise ValueError(
				'At most one of `frozen_params`, `unfrozen_params` must be provided!'
			)
		
		if not isinstance(frozen_params, list) and frozen_params is not None:
			frozen_params = [frozen_params]
		
		if not isinstance(unfrozen_params, list) and unfrozen_params is not None:
			unfrozen_params = [unfrozen_params]
		
		self.model = model
		
		params_to_check = self.resolve_params(params=frozen_params or unfrozen_params, frozen=not unfrozen_params)
		self.frozen_params = params_to_check.get('frozen', {})
		self.unfrozen_params = params_to_check.get('unfrozen', {})
		
		# if we're only specifying unfrozen params, save time
		# by removing grad on all the others. We'll need to deal
		# with the grads on the non-unfrozen indices separately
		# later, but this means we don't need to worry about
		# zeroing out the grads for all other parameters every
		# time the callback is invoked.
		for k in self.unfrozen_params:
			for name, param in self.model.named_parameters():
				if k != name:
					param.requires_grad = False
	
	def recursive_getattr(self, obj, attr, default = None):
		attr = attr.split('.', 1)
		if len(attr) == 1:
			return getattr(obj, attr[0], default)
		else:
			return self.recursive_getattr(getattr(obj, attr[0]), attr[1], default)
	
	def parse_tuple_slice(self, s: str) -> tuple:
		stack = []
		s = list(s)
		try:
			while s:
				c = s.pop(0)
				if c == '(':
					# this is safe because the tuples won't
					# be nested. I mean, they could be,
					# but it's not useful so the right answer
					# is "just don't do that".
					stack.append('')
					while (c := s.pop(0)) != ')':
						stack[-1] += c
				else:
					if c != ',':
						stack.append(c)
					
					while (c := s.pop(0)) != '(':
						stack[-1] += c
		except IndexError:
			pass
		
		for i, _ in enumerate(stack):
			if ':' in stack[i]:
				stack[i] = slice(*[int(x.strip()) if x.strip() else None for x in stack[i].split(':')])
			elif ',' in stack[i]:
				stack[i] = tuple([int(x) for x in stack[i].split(',')])
			else:
				stack[i] = int(stack[i])
		
		stack = tuple(stack)
		return stack
	
	def resolve_params(self, params: list[str], frozen: bool) -> dict:
		# check if we got an index for each param
		slices = [re.findall(r'\[([0-9\-,():]*)\]$', p) for p in params]
		
		# if we didn't get a slice, pass ':' so that we (un)freeze everything
		# for that parameter
		slices = [s[0] if s else ':' for s in slices]
		
		# pytorch uses tuples as slices, so we need to make a slice for
		# thing in the tuple
		slices = [self.parse_tuple_slice(s=s) for s in slices]
		
		# remove any slices
		params = [re.sub(r'\[.*\]$', '', p) for p in params]
		
		# map the string identifying the attribute to the slice to be 
		# frozen/unfrozen
		params_dict = {k: v for k, v in zip(params, slices)}
		
		params = {}
		params['frozen' if frozen else 'unfrozen'] = params_dict
		
		return params
	
	def find_word_embeddings_param_name(self, model: AutoModel) -> str:
		'''
		Attempt to find the name of the word embeddings parameter in the model.
		'''
		name = [
			name for name, _ in model.named_parameters()
			if any(x in name for x in ['word_embeddings', 'tok_embeddings'])
		]
		
		assert len(name) == 1, f'Found multiple possible parameters for word embeddings! {name}'
		
		return name[0]
	
	def __call__(self, epoch: int = None, batch: int = None) -> None:
		# replace the frozen parameter grads with zeros at the
		# corresponding indices. This will ensure that they
		# won't be updated during the backward pass.
		for name, idx in self.frozen_params.items():
			to_freeze = self.recursive_getattr(self.model, f'{name}.grad.data', None)[idx]
			to_freeze.fill_(0)
		
		for name, idx in self.unfrozen_params.items():
			# get the grads of the parameter whose element(s) we
			# want to retain the grad for
			to_unfreeze = self.recursive_getattr(self.model, f'{name}.grad.data', None)
			# save the original grads at the indices we don't want to freeze
			original_grads = to_unfreeze[idx].clone()
			
			# replace all grads in the unfrozen parameter with zeros.
			# this may seem counterintuitive, since we're supposed to
			# be unfreezing parameters, but this is so that any
			# values NOT at the indices we specifically want to 
			# unfreeze are frozen instead.
			to_unfreeze.fill_(0)
			
			# replace with the original grads
			to_unfreeze[idx] = original_grads

class UnfreezeWordTokensCallback(PartiallyFrozenModelCallback):
	'''
	Callback to unfreeze specific word tokens.
	'''
	def __init__(
		self, model: AutoModel, tokenizer: AutoTokenizer,
		words_to_unfreeze: list[str], word_embeddings_name: str = None,
	):
		super(PartiallyFrozenModelCallback, self).__init__()
		self.words_to_unfreeze = words_to_unfreeze
		self.model = model
		
		# this is to deal with tokens that have spaces before them
		# sometimes, we want to keep those. other times, the tokenizer
		# converts them into some bizarre symbol. We only want to unfreeze
		# word tokens with this, so this is okay. If you want to unfreeze
		# special tokens or non-word tokens, just look up the indices and
		# use the PartiallyFrozenModelCallback directly.
		self.tokens_to_unfreeze = tokenizer.convert_tokens_to_ids(
			[t for t in tokenizer.tokenize(self.words_to_unfreeze) if re.search(r'\w', t)]
		)
		if not isinstance(self.tokens_to_unfreeze, list):
			self.tokens_to_unfreeze = [self.tokens_to_unfreeze]
		
		assert not any(t == tokenizer.unk_token_id for t in self.tokens_to_unfreeze)
		
		self.word_embeddings_name = word_embeddings_name
		if self.word_embeddings_name is None:
			self.word_embeddings_name = self.find_word_embeddings_param_name(model=self.model)
		
		s = str(self.tokens_to_unfreeze)
		s = re.sub(r'\s', '', s)
		s = re.sub(r'^\[|\]$', '', s)
		if ',' in s:
			s = f'({s})'
		
		self.frozen_params = {}
		self.unfrozen_params = self.resolve_params(
			params=[
				f'{self.word_embeddings_name}[{s}]'
			],
			frozen=False,
		)['unfrozen']	

class FreezeWordTokensCallback(PartiallyFrozenModelCallback):
	'''
	Callback to unfreeze specific word tokens.
	'''
	def __init__(
		self, model: AutoModel, tokenizer: AutoTokenizer,
		words_to_freeze: list[str], word_embeddings_name: str = None,
	):
		super(PartiallyFrozenModelCallback, self).__init__()
		self.words_to_freeze = words_to_unfreeze
		self.model = model
		
		# this is to deal with tokens that have spaces before them
		# sometimes, we want to keep those. other times, the tokenizer
		# converts them into some bizarre symbol. We only want to freeze
		# word tokens with this, so this is okay. If you want to freeze
		# special tokens or non-word tokens, just look up the indices and
		# use the PartiallyFrozenModelCallback directly.
		self.tokens_to_freeze = tokenizer.convert_tokens_to_ids(
			[t for t in tokenizer.tokenize(self.words_to_freeze) if re.search(r'\w', t)]
		)
		# if not isinstance(self.tokens_to_freeze, list):
		# 	self.tokens_to_freeze = [self.tokens_to_freeze]
		
		assert not any(t == tokenizer.unk_token_id for t in self.tokens_to_freeze)
		
		self.word_embeddings_name = word_embeddings_name
		if self.word_embeddings_name is None:
			self.word_embeddings_name = self.find_word_embeddings_param_name(model=self.model)
		
		s = str(self.tokens_to_freeze)
		s = re.sub(r'\s', '', s)
		s = re.sub(r'^\[|\]$', '', s)
		if ',' in s:
			s = f'({s})'
		
		self.unfrozen_params = {}
		self.frozen_params = self.resolve_params(
			params=[
				f'{self.word_embeddings_name}[{s}]'
			],
			frozen=True,
		)['frozen']

class SwitchEncoderDecoderModesCallback:
	def __init__(
		self, model: AutoModel, tokenizer: AutoTokenizer,
		switch_strategy: str = '', start: str = 'encoder',
		freq: int = 1, log_switch: bool = False,
	):
		if not switch_strategy or switch_strategy not in ['epoch', 'batch']:
			raise ValueError(
				f'At least one of `epoch`, `batch` must be provided as a '
				'switching strategy!'
			)
			
		self.strategy = switch_strategy
		self.freq = freq
		
		self.model = model
		self.start = start
		if self.start == 'encoder':
			self.model.config.is_decoder = False
		elif self.start == 'decoder':
			self.model.config.is_encoder = True
		
		# we don't want to switch on the first call, since
		# that is counterintuitive with the 'start' argument +
		# the freq argument.
		self._first_call = True
		self.log_switch = log_switch
		
		if self.log_switch:
			state = 'decoder' if self.model.config.is_decoder else 'encoder'
			logger.info(f'{self.__class__.__name__}: {model.name_or_path} starting in {state} mode.')
	
	def __call__(self, epoch: int, batch: int) -> None:
		# add once since we start at zero and don't want to 
		# switch on the 0th epoch.
		
		# return early on the first call so we don't switch
		if self._first_call:
			self._first_call = False
			return
		
		if epoch is not None and self.strategy == 'epoch' and epoch % self.freq == 0:
			self.model.config.is_decoder = not self.model.config.is_decoder
			if self.log_switch:
				state = 'decoder' if self.model.config.is_decoder else 'encoder'
				logger.info(f'{self.model.name_or_path} switched to {state} mode ({epoch=}).')
		
		if batch is not None and self.strategy == 'batch' and batch % self.freq == 0:
			self.model.config.is_decoder = not self.model.config.is_decoder
			if self.log_switch:
				state = 'decoder' if self.model.config.is_decoder else 'encoder'
				logger.info(f'{self.model.name_or_path} switched to {state} mode ({batch=}).')

class SetEncoderModeCallback:
	def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
		'''
		Sets the model to encoder mode when called.
		'''
		self.model = model
	
	def __call__(self, epoch: int, batch: int) -> None:
		self.model.config.is_decoder = False

class SetDecoderModeCallback:
	def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
		'''
		Sets the model to decoder mode when called.
		'''
		self.model = model
	
	def __call__(self, epoch: int, batch: int) -> None:
		self.model.config.is_decoder = True

def add_new_tokens(
	model: AutoModel, 
	tokenizer: AutoTokenizer, 
	added_tokens: list[str],
) -> tuple[AutoModel, AutoTokenizer]:
	'''
	Adds new tokens to a model in the *right* way if possible,
	depending on the tokenizer type.
	'''
	tokenizer_type = determine_tokenizer_type(tokenizer=tokenizer)
	tokenizer.add_tokens(added_tokens)
	model.resize_token_embeddings(len(tokenizer))
	
	return model, tokenizer