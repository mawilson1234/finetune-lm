import torch
from transformers import (
	AutoModel,
	AutoConfig,
	AutoTokenizer,
	AutoModelForCausalLM,
	AutoModelForMaskedLM,
	AutoModelForSeq2SeqLM,
)

GPT2_MODELS: set[str] = (
	{'gpt2'} |
	{f'gpt2-{s}' 
		for s in 
		{'medium', 'large', 'xl'}
	}
)

OPT_MODELS: set[str] = (
	{f'facebook/opt-{i}m' for i in 
		{125, 350}
	} |
	{f'facebook/opt-{i}b' for i in 
		{1.3, 2.7, 6.7, 13, 30, 66, 175}
	}
)

HF_LLAMA_MODELS: set[str] = (
	{f'facebook/llama-hf/{i}B' for i in
		{7, 13, 30, 65}
	} |
	{f'meta-llama/Llama-2-{i}b-hf' for i in
		{7, 13, 70}
	} |
	{f'meta-llama/Meta-Llama-3-{i}B' for i in
		{8, 70}
	} |
	{f'meta-llama/Llama-3.1-{i}B' for i in
		{8, 70, 405}
	} |
	{f'meta-llama/Llama-3.2-{i}B' for i in
		{1, 3}
	}
)

NON_HF_LLAMA_MODELS: set[str] = (
	{f'facebook/llama/{i}B' for i in
		{7, 13, 30, 65}
	} |
	{f'facebook/llama-2/llama-2-{i}b' for i in
		{7, 13, 70}
	} |
	{f'facebook/llama-3/Meta-Llama-3-{i}B' for i in
		{8, 70}
	} |
	{f'facebook/llama-3.1/Llama-3.1-{i}B{j}' for i, j in 
		{(8, ''), (70, ''), (405, '-MP16')}
	} |
	{f'facebook/llama-3.2/Llama3.2-{i}B' for i in 
		{1, 3}
	}
)

NON_HF_LLAMA_TOKENIZERS: set[str] = (
	{
		'facebook/llama/tokenizer.model',
		'facebook/llama-2/tokenizer.model',
	} |
	{f'facebook/llama-3/Meta-Llama-3-{i}B/tokenizer.model' for i in
		{8, 70}
	} |
	{f'facebook/llama-3.1/Meta-Llama-3.1-{i}B{j}/tokenizer.model' for i, j in
		{(8, ''), (70, ''), (405, '-MP16')}
	} |
	{f'facebook/llama-3.2/Llama3.2-{i}B/tokenizer.model' for i in
		{1, 3}
	}
)

LLAMA_MODELS: set[str] = (
	HF_LLAMA_MODELS |
	NON_HF_LLAMA_MODELS
)

MAMBA_MODELS: set[str] = (
	{f'state-spaces/mamba-{s}-hf' for s in 
		{'130m', '370m', '790m', '1.4b', '2.8b'}
	}
)

PYTHIA_MODELS: set[str] = (
	{f'EleutherAI/pythia-{size}'
		for size in {'14m'} | {
			name for duped in {'', '-deduped'} for name in
			{f'{size}m{duped}' for size in [70, 160, 410]} |
			{f'{size}b{duped}' for size in [1, 1.4, 2.8, 6.9, 12]}
		}
	}
)

OLMO_MODELS: set[str] = (
	{f'allenai/OLMo-{i}' for i in {
			'1B', '1B-hf', '1B-0724-hf',
			'7B', '7B-hf', '7B-0424', 
			'7B-0424-hf', '7B-0724-hf', 
			'7B-Twin-2T', '7B-Twin-2T-hf',
		}
	}
)

OLMO_2_MODELS: set[str] = (
	{f'allenai/OLMo-2-{i}' for i in {
			'0425-1B', '1124-7B', '1124-13B', 
			'0325-32B', 
		}
	}
)

GPT_BERT_MODELS: set[str] = (
	{f'ltg/gpt-bert-babylm-{s}' for s in {
			'small', 'base',
		}
	}
)

NEXT_WORD_MODELS: set[str] = (
	OPT_MODELS |
	GPT2_MODELS |
	LLAMA_MODELS |
	MAMBA_MODELS |
	PYTHIA_MODELS |
	OLMO_MODELS |
	OLMO_2_MODELS | 
	GPT_BERT_MODELS
)

HF_AUTH_REQUIRED_MODELS: set[str] = (
	{model for model in 
		HF_LLAMA_MODELS 
			if any(name in model for name in {'Llama-2', 'Llama-3', 'Llama-3.1', 'Llama-3.2'})
	}
)

CASES: set[str] = {'uncased', 'cased'}

MASKED_LANGUAGE_MODELS: set[str] = (
	{f'distilbert-base-{case}' for case in CASES} |
	{f'bert-{size}-{case}' for case in CASES for size in {'base', 'large'}} |
	# {f'answerdotai/ModernBERT-{size}' for size in {'base', 'large'}} |
	{f'roberta-{size}' for size in {'base', 'large'}} |
	{f'phueb/BabyBERTa-{i}' for i in range(1,4)} |
	{'distilroberta-base'} |
	{
		f'albert-{size}-{version}' 
		for size in 
		{'base', 'large', 'xlarge', 'xxlarge'} 
			for version in 
			{'v1', 'v2'}
	} |
	# these models do not work (04/2023)
	# see https://github.com/huggingface/transformers/pull/18674
	# {
	#	f'microsoft/deberta-v3-{size}'
	#	for size in 
	#	{'xsmall', 'small', 'base', 'large'}
	# } |
	# {
	#	f'microsoft/deberta-{size}'
	#	for size in 
	#	{'base', 'large', 'xlarge'}
	# } |
	# {
	#	f'microsoft/deberta-v2-{size}' 
	#	for size in 
	#	{'xlarge', 'xxlarge'}
	# } |
	{
		f'google/electra-{size}-generator'
		for size in 
		{'small', 'base', 'large'}
	} |
	{
		f'google/multiberts-seed_{i}'
		for i in range(25)
	} |
	{
		f'google/multiberts-seed_{i}-step_{n}k'
		for i in range(5)
		for n in set(range(0,200,20)) | set(range(200,2001,100))
	} |
	{
		f'google/bert_uncased_L-{l}_H-{h}_A-{a}' 
		for l in {2, 4, 6, 8, 10, 12} 
		for h, a in zip((128, 256, 512, 768), (2, 4, 8, 12))
	} |
	{
		f'yanaiela/roberta-base-epoch_{n}'
		for n in range(84)
	} |
	{
		f'ltg/gpt-bert-babylm-{s}' for s in {
			'small', 'base',
		}
	}
)

MUELLER_T5_MODELS: set[str] = ({
	f'mueller/{m}' for m in
		{f'{pfx}-1m' for pfx in 
			{'babyt5', 'c4', 'wikit5', 'simplewiki'}} |
		{'babyt5-5m'} |
		{m for pfx in 
			{'c4', 'wikit5', 'simplewiki'}
				for m in 
				{f'{pfx}-{i}m' for i in {10, 100}}
		} |
		{m for pfx in 
			{'c4', 'wikit5'}
				for m in
				{f'{pfx}-{i}' for i in {'100m_withchildes', '1b'}}
		}
})

GOOGLE_T5_MODELS: set[str] = (
	{f'google/t5-efficient-{size}' 
		for size in 
		{'tiny', 'mini', 'small', 'base', 'large', 'xl', 'xxl'}
	} | 
	{f'google/t5-efficient-base-{ablation}'
		for ablation in 
		{f'dl{i}' for i in range(2,9,2)} |
		{f'el{i}' for i in range(2,9,2)} |
		{f'nl{i}' for i in (2**i for i in range(1,4,1))} |
		{f'nh{i}' for i in range(8, 33, 8)}
	} |
	{f'google/t5-efficient-mini-{ablation}'
		for ablation in 
		{f'nl{i}' for i in {6, 8, 12, 24}}
	}
)

T5_MODELS: set[str] = (
	MUELLER_T5_MODELS |
	GOOGLE_T5_MODELS
)

SEQ2SEQ_MODELS: set[str] = (
	T5_MODELS
)

ALL_MODELS: set[str] = (
	NEXT_WORD_MODELS |
	MASKED_LANGUAGE_MODELS |
	SEQ2SEQ_MODELS
)

import os
DATASETS: set[str] = set(
	os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data'))
)

def set_model_task(model_name_or_path: str, model_task: str) -> None:
	'''
	Used to set the model task manually. This is useful for fine-tuned models,
	for which we might not want to update constants.py, but instead provide the
	model task from the command line.
	'''
	# default case, no command line argument passed,
	# no update to ALL_MODELS needed.
	if model_task is None:
		return
	
	if model_task.lower() == 'lm+mlm':
		NEXT_WORD_MODELS.update({model_name_or_path})
		MASKED_LANGUAGE_MODELS.update({model_name_or_path})
	
	if model_task.lower() == 'lm':
		NEXT_WORD_MODELS.update({model_name_or_path})
	
	if model_task.lower() == 'mlm':
		MASKED_LANGUAGE_MODELS.update({model_name_or_path})
	
	if model_task.lower() == 'seq2seq':
		SEQ2SEQ_MODELS.update({model_name_or_path})
	
	ALL_MODELS.update(
		NEXT_WORD_MODELS |
		MASKED_LANGUAGE_MODELS |
		SEQ2SEQ_MODELS
	)

def is_huggingface_model(model_name_or_path: str) -> bool:
	'''
	Is the model compatible with the Hugging Face API or not?
	'''
	return (
		model_name_or_path in ALL_MODELS and
		model_name_or_path not in NON_HF_LLAMA_MODELS
	)

def get_tokenizer_kwargs(model_name_or_path: str) -> dict:
	# gets the appropriate kwargs for the tokenizer
	# this provides us a single place where we can deal
	# with idiosyncrasies of specific tokenizers
	tokenizer_kwargs = {'pretrained_model_name_or_path': model_name_or_path}
	if 'BabyBERTa' in model_name_or_path:
		tokenizer_kwargs = {**tokenizer_kwargs, 'add_prefix_space': True}
	
	# these ones can't be used fast,
	# since it causes problems
	if any(s in model_name_or_path for s in {'deberta-v3', 'opt-'}):
		tokenizer_kwargs = {**tokenizer_kwargs, 'use_fast': False}
	
	# opt-175b doesn't come with a HF tokenizer,
	# but it uses the same one as the smaller models
	# which are available on HF
	if 'opt-175b' in model_name_or_path:
		tokenizer_kwargs = {**tokenizer_kwargs, 'pretrained_model_name_or_path': 'facebook/opt-125m'}
	
	# padding on the left does not work correctly, but for some reason it is the default
	# for this tokenizer. So let's fix that.
	if 'llama-hf' in model_name_or_path:
		tokenizer_kwargs = {**tokenizer_kwargs, 'padding_side': 'right'}
	
	if model_name_or_path in NON_HF_LLAMA_MODELS:
		tokenizer_kwargs = {
			**tokenizer_kwargs,
			'pretrained_model_name_or_path': os.path.join(os.path.dirname(model_name_or_path), 'tokenizer.model')
		}
	
	return tokenizer_kwargs

def model_not_supported_message(model_name_or_path: str) -> str:
	return (
		f'{model_name_or_path!r} has not been classified in `constants.py`. '
		'If you would like to add it, you should add it to `SEQ2SEQ_MODELS` '
		'for T5 models (evaluated using a conditional generation task), '
		'`MASKED_LANGUAGE_MODELS` for models that should be evaluated using '
		'a masked language modeling task, or `NEXT_WORD_MODELS` for models that '
		'should be evaluated using a language modeling task. (Encoding-decoder models '
		'beside T5 models are not currently supported.)'
	)

def load_model(model_name_or_path: str, *args, **kwargs) -> 'AutoModel':
	'''
	Loads the model using the appropriate function.
	'''
	
	# ThE second clause is needed to deal with 
	# gpt-bert, which can be used as both a causal
	# lm or as a masked lm, but we want to load
	# it as a masked LM always (using it in decoder
	# model just means setting model.config.is_decoder to True)
	if (
		model_name_or_path in NEXT_WORD_MODELS and 
		not model_name_or_path in MASKED_LANGUAGE_MODELS
	):
		model = AutoModelForCausalLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
		# store any kwargs in the model so
		# we can pass them to the KL baseline loss later
		setattr(model, 'model_kwargs', kwargs)
		if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
			model.config.pad_token_id = model.config.eos_token_id
		
		if not hasattr(model.config, 'bos_token_id') or model.config.bos_token_id is None:
			model.config.bos_token_id = model.config.eos_token_id
		
		return model
	
	if model_name_or_path in MASKED_LANGUAGE_MODELS:
		model = AutoModelForMaskedLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
		setattr(model, 'model_kwargs', kwargs)
		return model
	
	if model_name_or_path in SEQ2SEQ_MODELS:
		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
		setattr(model, 'model_kwargs', kwargs)
		return model
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def load_tokenizer(tokenizer_name_or_path: str, *args, **kwargs) -> AutoTokenizer:
	'''
	Loads a tokenizer and adds pad token if needed.
	'''
	tokenizer = AutoTokenizer.from_pretrained(
		tokenizer_name_or_path,
		*args, **kwargs
	)
	# store the tokenizer kwargs for use later
	setattr(tokenizer, 'tokenizer_kwargs', kwargs)
	
	if tokenizer.name_or_path in HF_LLAMA_MODELS:
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	
	if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	
	if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
		tokenizer.bos_token = tokenizer.eos_token
		tokenizer.bos_token_id = tokenizer.eos_token_id
	
	return tokenizer

def load_tokenizer_and_model(
	model_args: 'ModelArguments' = None,
	name_or_path: str = None,
	model_kwargs: dict = None,
	tokenizer_kwargs: dict = None,
	config_kwargs: dict = None,
	use_gpu: bool = False,
	model_modifier_fns: list[callable] = None,
	model_modifier_fn_kwargs: dict[str,dict] = None,
) -> tuple[AutoTokenizer, AutoModel]:
	'''Loads the tokenizer and model as specified in model_args or by the passed arguments.'''
	if model_args is not None:
		model_name_or_path = model_args.model_name_or_path
		config_name = model_args.config_name
		tokenizer_name = model_args.tokenizer_name
		model_kwargs = model_args.model_kwargs
		tokenizer_kwargs = model_args.tokenizer_kwargs
		config_kwargs = model_args.config_kwargs
		model_kwargs.update(dict(
			from_flax=model_args.from_flax,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			token=model_args.token,
		))
		tokenizer_kwargs.update(dict(
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			token=model_args.token,
		))
		config_kwargs.update(dict(
			cache_dir=model_args.cache_dir,
			revision=model_args.cache_dir,
			use_auth_token=model_args.token,
		))
		use_gpu = model_args.use_gpu
		model_modifier_fns = model_args.model_modifier_fns
		model_modifier_fn_kwargs = model_args.model_modifier_fn_kwargs
	else:
		model_name_or_path = config_name = tokenizer_name = name_or_path
		model_kwargs = model_kwargs if model_kwargs is not None else {}
		tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
		config_kwargs = config_kwargs if config_kwargs is not None else {}
		model_modifier_fns = model_modifier_fns if model_modifier_fns is not None else []
		model_modifier_fn_kwargs = model_modifier_fn_kwargs if model_modifier_fn_kwargs is not None else {}
	
	if name_or_path in NON_HF_LLAMA_MODELS:
		raise ValueError(model_not_supported_message(name_or_path))
	
	config = AutoConfig.from_pretrained(
		config_name,
		**config_kwargs
	)
	
	tokenizer = load_tokenizer(
		tokenizer_name,
		**tokenizer_kwargs,
	)
	
	model = load_model(
		model_name_or_path,
		config=config,
		**model_kwargs,
	)
	
	if model.name_or_path in HF_LLAMA_MODELS:
		model.resize_token_embeddings(len(tokenizer))
	
	if use_gpu and torch.cuda.is_available():
		model.to('cuda')
	elif use_gpu:
		logger.warning('`use_gpu` was set, but no GPU was found. Defaulting to CPU.')
	
	for fn in model_modifier_fns:
		kwargs = model_modifier_fn_kwargs.get(fn.__name__, {})
		tokenizer, model = fn(model=model, tokenizer=tokenizer, **kwargs)
	
	return tokenizer, model
