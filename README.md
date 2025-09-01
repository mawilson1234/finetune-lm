# `finetune-lm`: a repository for fine-tuning language models

The goal of `finetune-lm` is to provide a flexible framework for fine-tuning and optimizing language models that work with Hugging Face's `transformers` library.

## Requirements

Set up the environment using `conda` or `mamba` by running `conda env create -f ./environment.yaml`.

## Usage

There are two things to keep in mind when using `finetune-lm`. The first is the format of the datasets, and the second is the command line arguments.

### Dataset format

Provided datasets are stored in a subdirectory of the `data` directory (you may store your own datasets wherever you like). Datasets consist of (i) a gzipped text file, with a single example per line (required); and (ii) a file with the same name, but which ends in `_metadata.json.gz` instead of `.txt.gz`, which consists of a gzipped text file with a single json dictionary per line. Models will be fine-tuned on the examples in the gzipped text file. Key-value pairs from the corresponding line of the metadata file will be added to the results for each example.

### Command line arguments

Command line arguments are defined in dataclasses in `core/model_arguments.py`, `core/data_training_arguments.py`, and `core/optimization_arguments.py`. They are parsed using a customized parser, defined in `core/parser.py`, and defined below.

#### `ModelArguments`

The `ModelArguments` dataclass accepts the following parameters:

- `model_name_or_path` (no default)
- `config_name`: default is `model_name_or_path`
- `cache_dir`: def; ault is Hugging Face's cache dir
- `tokenizer_name`: default is `model_name_or_path`
- `use_fast_tokenizer`: default is `True`
- `model_revision`: default is `"main"`
- `token`: your Hugging Face access token, or the name of a file where it is stored. default is `False` (meaning no token is used)
- `use_gpu`: default is `False`

#### `DataTrainingArguments`

The `DataTrainingArguments` dataclass accepts the following parameters:

- `train_file`: the path to the `.txt.gz` file described above
- `validation_file`: see `train_file`, but for the validation dataset
- `test_file`: see `train_file`, but for the test dataset(s). If a list is provided, evaluation will be run on all test datasets and saved in a single file, with each dataset's name in the 'dataset_name' column.
- `overwrite_cache`: default is `False`
- `preprocessing_num_workers`: default is `None`
- `max_length`: default is `1024`
- `pad_to_max_length`: default is `False`
- `max_train_samples`: default is `None` (uses all examples)
- `per_device_train_batch_size`: default is `32`
- `max_val_samples`: default is `None` (uses all examples)
- `per_device_validation_batch_size`: default is `32`
- `max_test_samples`: default is `None` (uses all examples). The same value applies to all test datasets.
- `per_device_validation_batch_size`: default is `32`. The same value applies to all test datasets.
- `ignore_pad_for_token_loss`: default is `True`
- `lr`: default is `2e-6`
- `epochs`: default is `250`
- `min_epochs`: even if `patience` is used, at least this many epochs will be completed before stopping. Default is `0`
- `patience`: the number of epochs to continue training without observing improved performance on the validation dataset. default is `None` (no early stopping)
- `delta`: the minimum amount of improvement on loss that resets the patience counter. Default is `0` (any improvement resets the patience counter)
- `use_kl_baseline_loss`: whether to add a loss term based on the KL divergence between the model being fine-tuned on the original version of that model. Used to help avoid overfitting to the fine-tuning dataset. Default is `False`
- `kl_dataset`: the path to the dataset used to compute the KL divergence loss term. Currently, this should be point to a directory containing the dataset in `.arrow` format. This may change later to be more consistent with other dataset formats. Default is `None`
- `kl_batch_size`: default is `32`
- `kl_n_examples_per_step`: how many examples from `kl_dataset` to use per weight update/batch to compute the KL divergence loss term. Default is `20`
- `kl_scaleby`: a multiplier for the KL divergence loss term. Default is `2.5`
- `kl_max_samples`: default is `None` (all samples may be used)
- `kl_reduction`: default is `'none'`. It is recommended you don't modify this.
- `output_dir`: Used to store the output directory name. This will be determined automatically.

#### `OptimizationArguments`

The `OptimizationArguments` dataclass accepts the following parameters:

- `do_optimize`: whether to run an optimization study. Default is `False`
- `study_kwargs`: a dictionary containing keyword arguments to be passed when the Optuna study is created. Default is:
	```
	{
		'study_name': None, # will be determined automatically if not set manually
		'sampler': optuna.samplers.GPSampler,
		'sampler_kwargs': {
			'deterministic_objective': True,
		},
		'pruner': optuna.pruners.PatientPruner,
		'pruner_kwargs': {
			'wrapped_pruner': optuna.pruners.WilcoxonPruner,
			'wrapped_pruner_kwargs': {
				'p_threshold': 0.1,
				'n_startup_steps': 100,
			},
			'patience': 30,
			'min_delta': 0.,
		},
		'direction': 'minimize',
	}
	```
	`sampler_kwargs`, `pruner_kwargs`, and `wrapped_pruner_kwargs`, if set, will be passed to the class constructor for the associated `sampler`, `pruner`, and `wrapped_pruner`. To set nested values in this dictionary and others, see below on how the custom parser works.
- `optimize_kwargs` a dictionary containing keyword arguments passed to the `study.optimize` call. Default is:
	```
	{
		'n_trials': 2,
		'gc_after_trial': True,
		'show_progress_bar': False,
	}
	```
- `params`: a dictionary specifying which parameters are to be optimized, and how. Valid keys are any parameter name in the `DataTrainingArguments` dataclass (see above). The value associated with each key must be a dictionary containing at least a `values` key, which should map to a list of possible values (for categorical hyperparameters) or a list containing the upper and lower bounds of the range of possible values (for integer or float hyperparameters). Default is:
	```
	{
		'lr': {
			'values': [2e-6, 2e-5],
			'suggest_kwargs': {},
		}
	}
	```
	Any keyword arguments provided in `suggest_kwargs` for a particular parameter are passed to the `suggest_` function of the Optuna `trial` object.

## Passing command line arguments

Command line arguments are specified using `--` before the name of a parameter found in one of the dataclasses detailed above, followed by the value. This is how Python's `argparse` library works, and it mostly works the same here. For example:

`--model_name_or_path gpt2`

Will set the `model_name_or_path` parameter of the `ModelArguments` class to `'gpt2'`.

However, in order to facilitate setting other parameters, a custom argument parser is defined in `core/parser.py`. It works as follows.

- If an argument value can be cast to `int` or `float`, it will be. To pass a number as a string, put quotes around it. Depending on your shell's behavior, you may need to escape the quotes with an additional set of quotes or backslashes for this to work.

- If an argument *name* has a `.` in it, it will be parsed as a dictionary. For instance `--parameter_name.key value` will associate the parameter named `parameter_name` with the dictionary `{'key': 'value'}`. Nested dictionaries add more dots for each level of nesting.

- If an argument *value* has a `.` in it, it will be treated as a module and imported. The object will be associated with the argument name, rather than the string naming the module. To pass an argument with a literal string `'.'` in it (e.g., a filename), put quotes around it. To pass an argument with quotes in the string, use two sets of quotes around it. If you want to import a module without a `.` in it to use as an argument value, put `import:` before the module name.

- If you want to use a predefined Python function as an argument, put `callable:` in front of it.

- If an argument value is `"True"` or `"False"`, it will be converted to boolean.

- If an argument value is `"None"`, it will be converted to `None`.

- If an argument name is not followed by a value (i.e., the next thing is an argument starting with `--`), the value of that argument will be boolean `True`.

- Default values for dictionary arguments in dataclasses will be merged with those provided on the command line. If you want to remove a key from one of the default dictionaries above instead of overwriting it with a new value, set the value of the corresponding key to `__delete_field__` in your script call, and the key will be removed entirely from the dictionary.

- To pass a list of values as an argument, put all values to be added to the list following the argument name, before the next argument name. For instance, `--params.lr.values 1e-5 1e-4 --use_gpu` will be parsed as `{'params': {'lr': {'values': [1e-5, 1e-4]}}, 'use_gpu': True}`.

## Outputs

The default output directory is 'outputs/${train_file}/${model_name_or_path}/${year-month-day_hour_minute_second.nanoseconds}'.

When running an optimization study, outputs will be `config.txt`, a log file displaying all the config options tried over the course of the study, as well as `optimization_results.csv.gz`, a CSV containing the results of the study. This has the loss value associated with each combination of hyperparameters tried during the study.

When not running an optimization study, outputs will be `config.txt`, a log file displaying the config options used to fine-tune the model. The `model` subdirectory stores the model state that performed best on the validation dataset, and `tokenizer` stores the tokenizer. `metrics.csv.gz` is a CSV with per-token surprisals for every sentence for the test and validation datasets for every batch in every epoch. `loss_curves.pdf` plots the training and validation loss (as well as the training + KL divergence loss, if that option is used) for all batches and epochs.