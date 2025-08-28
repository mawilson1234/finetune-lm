# dataset maker
#
# use to make a corpus of random examples from huggingface datasets
# it is NOT recommended that you run this locally unless you want to take up
# a lot of disk space with datasets just to generate a small one
import os
import re
import json
import gzip
import hydra
import logging

from tqdm import tqdm
from typing import *
from random import random
from datasets import load_dataset, DatasetDict, Dataset
from omegaconf import OmegaConf, DictConfig

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'namer',
	lambda d: '-'.join([f'{d[0]}={d[1]}' for d in list(d.items())])
)	

@hydra.main(config_path='.', config_name='dataset_maker')
def create_save_dataset(cfg: DictConfig) -> None:
	'''
	Create a dataset of sentences randomly pulled from huggingface datasets.
	The dataset is saved in a text file with one sentence per line.
	
		params:
			cfg (DictConfig): a dict/dictconfig with the following parameters specified:
			
			n (int)					: the number of sentences to go in the dataset
			datasets (dict)			: a dictionary mapping a huggingface dataset name to
									  the approximate proportion of examples to pull from that dataset
			dataset_args (tuple) 	: additional arguments to pass to load_dataset for each dataset
			dataset_kwargs (dict)	: additional arguments to pass to load_dataset for each dataset
			name (str)				: what to name the dataset. if not provided, the dataset will be named
									  using information from the datasets dictionary
	'''
	
	# Collect configuration options
	datasets 		= cfg.datasets
	dataset_args 	= cfg.dataset_args
	dataset_kwargs 	= cfg.dataset_kwargs
	name 			= cfg.name
	
	assert sum(v for v in datasets.values()) == 1, 'Probabilities for all datasets must sum to 1!'
	
	loaded_datasets = dict.fromkeys(datasets.keys())
	for dataset in loaded_datasets:
		try:
			loaded_datasets[dataset] = load_dataset(dataset, *dataset_args[dataset], **dataset_kwargs[dataset])
		except Exception:
			raise ValueError(f'Unable to load dataset {dataset} on huggingface!')
	
	# we need to set up the probabilities as ranges for a random number
	# which means adding each previous probability to the next
	previous_prob = 0
	for dataset, prob in datasets.items():
		previous_prob += datasets[dataset]
		datasets[dataset] = previous_prob
	
	new_dataset = dict.fromkeys(cfg.splits)
	exs 	 	= [None for _ in range(sum(cfg.splits.values()))] # so we don't repeat sentences
	for split in cfg.splits:
		n = cfg.splits[split]
		new_dataset[split] = {}
		
		# preallocate
		new_dataset[split]['text'] = [None for _ in range(n)]
		new_dataset[split]['source'] = [None for _ in range(n)]
		n_chosen = 0
		
		# we don't just shuffle the dataset and choose the first n examples,
		# because some datasets contain multiple sentences per row. we want
		# n sentences, which means getting the row, and then splitting and getting a random (good)
		# sentence from that row. we also don't want repeats that are identical except for case
		with tqdm(total=n) as pbar:
			while n_chosen < n:
				r = random()
				for dataset, prob in datasets.items():
					if r < prob:
						current_dataset = dataset
						break
				
				# np.random.choice is sloooow with big lists
				r = int(round(random() * (len(loaded_datasets[current_dataset]['train'])-1),0))
				
				ex = loaded_datasets[current_dataset]['train'][r]['text']
				
				# do some formatting: split on periods, remove anything with newlines
				# newlines would sometimes be best replaced with commas, or bullet points, etc.
				# better to just leave them out entirely
				
				# we split this way to retain the delimiters
				ex = [s for s in re.sub(r'((\.) |$)|((\?) |$)|((\!) |$)', '\\2&&&', ex).split('&&&') if not '\n' in s]
				# remove empty strings and extra leading/trailing spaces
				ex = [s.strip() for s in ex if s.strip()]
				
				# if there's anything left, save an example
				if ex and not all(ex.lower() in exs for ex in ex):
					# get a random example from the retained sentences
					r = int(round(random() * (len(ex)-1),0))
					e = ex[r]
					while e.lower() in exs:
						r = int(round(random() * (len(ex)-1),0))
						e = ex[r]
					
					# use lower case here because we want sentences that are distinguished by more than case
					# this is because we are using some uncased models
					exs[n_chosen] = e.lower()
					new_dataset[split]['source'][n_chosen] = current_dataset
					new_dataset[split]['text'][n_chosen] = e
					
					n_chosen += 1
					pbar.set_postfix(split=split)
					pbar.update(1)
	
	new_dataset = DatasetDict({k: Dataset.from_dict(v) for k, v in new_dataset.items()})
	new_dataset.save_to_disk(name)
	log.info(f'Dataset saved in directory {name} in "{os.getcwd()}".')

if __name__ == '__main__':
	
	create_save_dataset()
