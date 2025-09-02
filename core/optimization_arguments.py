import os
import optuna

from typing import Optional
from dataclasses import dataclass
from dataclasses import field, fields

if __name__ == '__main__':
	from .data_training_arguments import DataTrainingArguments
else:
	from data_training_arguments import DataTrainingArguments

@dataclass
class OptimizationArguments:
	"""Arguments pertaining to hyperparameter optimization."""
	do_optimize: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Whether to run multiple training sessions to optimize hyperparameters."
		},
	)
	
	study_kwargs: Optional[dict] = field(
		default_factory = lambda: {
			"study_name": None,
			"sampler": optuna.samplers.GPSampler,
			"sampler_kwargs": {
				"deterministic_objective": True,
			},
			"pruner": optuna.pruners.PatientPruner,
			"pruner_kwargs": {
				"wrapped_pruner": optuna.pruners.WilcoxonPruner,
				"wrapped_pruner_kwargs": {
					"p_threshold": 0.1,
					"n_startup_steps": 100,
				},
				"patience": 30,
				"min_delta": 0.,
			},
			"direction": "minimize",
		},
		metadata={
			"help": "A dictionary of keyword arguments to pass to optuna.create_study. An argument named "
			"x that is a class will be instantiated with keyword arguments passed at the same level in the "
			"dictionary named 'x_kwargs'."
		}
	)
	
	optimize_kwargs: Optional[dict] = field(
		default_factory = lambda: {
			"n_trials": 2,
			"gc_after_trial": True,
			"show_progress_bar": False,
		},
		metadata={
			"help": "Dictionary of keyword arguments to pass to optuna.study.Study.optimize."
		}
	)
	
	params: Optional[dict] = field(
		default_factory = lambda: {
			'lr': {
				'values': [2e-6, 2e-5],
				'suggest_kwargs': {},
			}
		},
		metadata={
			"help": "A dictionary specifying which parameters to optimize, and how to optimize them. "
			"Each key should correspond to the name of a parameter in `DataTrainingArguments`, and map "
			'to a dictionary containing a "values" key that maps to the desired values (see optuna documentation). '
			"Any keyword arguments to be passed to the `suggest` function should be specified in the corresponding "
			'hyperparameters dictionary under the key "suggest_kwargs".'
			
		}
	)
	
	def __post_init__(self):
		if self.do_optimize:
			# this lets us override defaults in dictionaries by passing an empty dictionary or string
			# as the value
			for d in vars(self):
				if isinstance(d, dict):
					for p in d.copy():
						if not d[p]:
							del d[p]
			
			self._set_study_name()
			
			# used so that we can see the kwargs passed to object constructors
			# for reproduceability
			self.original_repr = repr(self)
			
			self._resolve_study_kwargs()
			self.study = optuna.create_study(**self.study_kwargs)
	
	def _set_study_name(self) -> None:
		'''
		Sets the study's name.
		'''
		# don't override anything that was set manually
		if self.study_kwargs['study_name'] is not None:
			return
		
		self.study_kwargs['study_name'] = '-'.join(
			[f'objective_{self.study_kwargs["direction"]}_loss'] +
			['_'.join([k] + ['_'.join([str(v2) for v2 in self.params[k]['values']])]) for k, v in self.params.items()]
		)
	
	def _resolve_study_kwargs(self) -> None:
		'''
		Returns a dictionary containing the study kwargs for optuna.
		This loads the pruner and sampler classes from a string identifier
		if one is provided, as well as functions they may need.
		'''
		st_kwargs = {}
		for k, v in self.study_kwargs.items():
			# these should just be strings, ints, or floats, so we're good
			if not any(k.startswith(x) for x in ['sampler', 'pruner', 'storage']) or v is None:
				st_kwargs[k] = v
				continue
			
			if k in ['sampler', 'pruner', 'storage']:
				sp_kwargs = {}
				for k2, v2 in self.study_kwargs.get(f'{k}_kwargs', {}).items():
					# these should just be strings, ints, or floats,
					# so we're good
					if not any(
						k2.startswith(x) for x in [
							'wrapped_pruner', 'gamma', 'weights',
							'log_storage',
						]
					):
						sp_kwargs[k2] = v2
						continue
					
					# a wrapped pruner is an object, so we need to instantiate it
					# with the appropriate kwargs
					if k2 in ['wrapped_pruner', 'independent_sampler']:
						sp_kwargs[k2] = v2(
							**self.study_kwargs
								.get(f'{k}_kwargs', {})
								.get(f'{k2}_kwargs', {})
						)
					
					# a gamma or weights kwarg is a function passed to a sampler,
					# so we need to convert the string identifier to the actual
					# Callable with the right parameters
					if k2 in ['gamma', 'weights', 'constraints_func']:
						from functools import partial
						callable_function = partial(
							v2,
							*self.study_kwargs
								.get(f'{k}_kwargs', {})
								.get(f'{k2}_args', []),
							**self.study_kwargs
								.get(f'{k}_kwargs', {})
								.get(f'{k2}_kwargs', {})
						)
						sp_kwargs[k2] = callable_function
					
					# deal with log storage in case it's an object
					if k2 in ['log_storage']:
						if not isinstance(v2, str):
							log_storage_kwargs = self.study_kwargs.get(f'{k}_kwargs', {}).get(f'{k2}_kwargs', {})
							if any(x and f'{x}_kwargs' in log_storage_kwargs for x in log_storage_kwargs):
								for k3, v3 in log_storage_kwargs.copy().items():
									if k3 and f'{k3}_kwargs' in log_storage_kwargs:
										log_storage_kwargs[k3] = v3(**log_storage_kwargs[f'{k3}_kwargs'])
										# we need to delete here since otherwise we end up passing
										# an extra argument to the storage handler that it won't know
										# how to deal with. Ideally we'd leave this in for visibility,
										# but I don't feel like dealing with that headache now.
										del log_storage_kwargs[f'{k3}_kwargs']
									
									# we have to make the directory since optuna doesn't know how to create
									# directories for us. sigh
									if k3 == 'file_path':
										os.makedirs(os.path.split(v3)[0], exist_ok=True)
							
							sp_kwargs[k2] = v2(
								**self.study_kwargs
									.get(f'{k}_kwargs', {})
									.get(f'{k2}_kwargs', {})
							)
				
				st_kwargs[k] = v(**sp_kwargs)
		
		self.study_kwargs = st_kwargs
	
	def set_suggested_values(self, data_args: DataTrainingArguments, trial: optuna.trial.Trial) -> None:
		'''
		Sets suggested values in data_args for the trial.
		'''
		for k, v in self.params.items():
			if not (isinstance(getattr(data_args, k), int) or isinstance(getattr(data_args, k), float)):
				setattr(data_args, k, trial.suggest_categorical(k, v['values']))
			else:
				if len(v['values']) != 2:
					raise ValueError(
						f'To suggest int or float for optimization requires exactly 2 values: '
						f'a max and a min. But {len(v["values"])} values were provided ({v["values"]!r}).'
					)
				
				suggest_kwargs = dict(
					name=k,
					low=min(v['values']),
					high=max(v['values']),
				)
				# the disjunction replaces 'None' with an empty dict,
				# so that we don't add anything if there are no kwargs
				# specified. The get call handles a case when suggest_kwargs
				# is missing entirely, rather than None.
				suggest_kwargs.update(v.get('suggest_kwargs', {}) or {})
			
			# we allow setting a 'type' entry in the value dict to
			# override behavior depending on whether the default
			# value for the hyperparamater is an int or a float
			if isinstance(getattr(data_args, k), int) or v.get('type') == 'int':
				setattr(data_args, k, trial.suggest_int(**suggest_kwargs))
			elif isinstance(getattr(data_args, k), float) or v.get('type') == 'float':
				setattr(data_args, k, trial.suggest_float(**suggest_kwargs))
			else:
				raise ValueError(
					f'Can only optimize parameters of type "categorical", "int", and "float", '
					f'but {type(getattr(data_args, k))!r} was provided.'
				)