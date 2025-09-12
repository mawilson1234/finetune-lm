# `finetune-lm`: a repository for fine-tuning language models

The goal of `finetune-lm` is to provide a flexible framework for fine-tuning and optimizing language models that work with Hugging Face's `transformers` library.

## Requirements

Set up the environment using `conda` or `mamba` by running `conda env create -f ./environment.yaml`. To enable all plotting features, if you do not have Chrome installed on your system, you will need to install it by executing the following commands in Python:

- `from plotly.io import get_chrome`
- `get_chrome()`

This will enable saving plots of optimization studies.

## Usage

There are two main things to keep in mind when using `finetune-lm`. The first is the format of the datasets, and the second is the command line arguments.

### Dataset format and preprocessing

Provided datasets are stored in a subdirectory of the `data` directory (you may store your own datasets wherever you like). Datasets consist of (i) a gzipped text file with one example per line; or a gzipped `.json.gz` file, with a single dictionary per line that contains a single example under the keys `"text"` and `"labels"`; and (ii) a file with the same name, but which ends in `_metadata.json.gz` instead of `.(txt|json).gz`, which consists of a gzipped text file with a single json dictionary per line. Models will be fine-tuned on the examples in the gzipped text file. If a `.json.gz` file is provided as a data file, the labels in the `"labels"` key will be the targets for fine-tuning for masked language models and seq2seq models. For language models, labels will be ignored and the inputs (shifted left) serve as the labels as usual. Key-value pairs from the corresponding line of the metadata file will be added to the results for each example.

To facilitate running unsupervized learning fine-tuning, functions that can be used to modify the inputs before they go into the model are provided in `data_preprocessing.py`. Data preprocessing functions take, at minimum, a dictionary of inputs mapping string keys to PyTorch tensors, and the tokenizer used to create this dictionary. Additional kwargs may also be added to these (see below). This dictionary is assumed to have the inputs under the key `'input_ids'`, and (if there are any) the labels under `'labels'`. If labels are not provided (which is usually the case when these functions are useful), the function may create them. The functions that are provided are:

- `identity`: If no labels are provided in the dictionary, sets labels to a copy of the input ids. Otherwise, the inputs are returned unchanged.
- `remove_non_mask_tokens_from_labels`: Set the value of non-mask tokens in the labels to `-100`, so that the non-mask tokens are ignored when determining loss. Useful if you provide a dataset with pre-baked masking in the inputs for masked language modeling.
- `mask_random_tokens`: Implements the BERT mask filling pre-training objective. Accepts kwargs `mask_prop (float)` and `replacement_props` (a dictionary with keys `'mask'`, `'random'`, and `'original'`, which each map to a float whose values must all add to 1). For each example, selects `max(mask_prop * len(example), 1)` tokens as mask tokens. For each selected token, replaces it with a mask token `replacement_props['mask']` of the time, a random token from the tokenizer's vocabulary `replacement_props['random']` of the time, and the original token `replacement_props['original']` of the time. If no labels are provided, labels are constructed with the original tokens in the mask positions, and all other tokens mapped to `-100`.
- `mask_random_spans`: Implements the T5 span denoising pre-training objective. Accepts kwargs `mask_prop` and `mean_noise_span_length`. For each input, selects `max(1, (mask_prop * len(example))/mean_noise_span_length)` contiguous spans of tokens with a mean length of `mean_noise_span_length` as spans to replace with a single sentinel token. Consecutive spans that are corrupted in the same input are replaced with consecutive sentinel tokens. Note that the first span will never be a noised span. If no labels are provided, labels are constructed such that each sentinel token in an input is followed by the original tokens in the span corrupted by that sentinel, with a final unique sentinel token at the end. Labels are padded with `-100` so that pad tokens do not affect the loss calculation.
- `mask_truerandom_spans`: Accepts kwarg `mask_prop`. Randomly selects `max(mask_prop * len(example), 1)` tokens per example to corrupt. If selected tokens happen to be adjacent, they are corrupted by a single sentinel token (this is different from the `mask_random_spans` behavior and the T5 pre-training objective, which guarantees spans have a certain mean length). Following corruption, if labels are not provided, they are computed in the same way as for `mask_random_spans`.
- `expand_with_masks`: For each example in inputs, expands it by constructing the examples that differ from the example by replacing a single position at a time with a mask/sentinel token. If no labels are provided, they are constructed in the ways described for `mask_random_tokens` (for masked language models) or `mask_random_spans` (for span denoising seq2seq models).
- `mask_word_tokens`: Accepts kwarg `word_tokens_to_mask (list[str])`. Verifies that all tokens in `word_tokens_to_mask` consist of a single (non-unknown) token id, and replaces them with the mask/sentinel token in each input. If no labels are provided, they are constructed as described for `mask_random_tokens` (for masked language models) or `mask_random_spans` (for span denoising seq2seq models).
- `mask_words`: Accepts kwarg `words_to_mask (list[str])`. For each word, tokenizes it. Replaces each contiguous span of tokens in the inputs that matches the tokens of each word with mask/sentinel tokens, starting with the longest sequences of tokens. If a tokenizer for a span denoising model is passed, replaces consecutive sentinel tokens corresponding to a single word with a single sentinel token. If no labels are provided, they are constructed as described for `mask_random_tokens` (for masked language models) or `mask_random_spans` (for span denoising seq2seq models).

A custom dataset class is provided in `core/dataset.py`. The class contains a Hugging Face `datasets.Dataset`, but also includes the original text inputs used to make the tensors, the original text for the labels, and metadata (from the dataset file's corresponding `_metadata.json.gz` file). To access the original dataset, use `dataset.dataset`. Otherwise, indexing returns corresponding rows from the `dataset.dataset`, `dataset.texts`, `dataset.labels`, and `dataset.metadata`, as a single dictionary.

If you implement a data preprocessing function that maps a single example to multiple examples (for instance, `expand_with_masks`), you should add a key to the returned dictionary, `expanded_lengths`, that contains the expanded length of each example, in order. Since padding will be needed to ensure equal numbers of entries in all fields of the dataset, use `-1` as a padding token that will be removed. After preprocessing the dataset with the `data_preprocessing_fn` passed to the class constructor, the texts, labels, and metadata at each index will each be multiplied the number of times corresponding to the values in the `expanded_length` column when the pad values are dropped, ensuring that the texts, labels, and metadata correctly line up with the original example they correspond to.

If you implement a data preprocessing function that should require the labels to be regenerated afterward (e.g., span denoising), you should add it to the set `UPDATE_LABEL_FNS` in `data_preprocessing.py`. This will ensure that the dataset regenerates its labels attribute following application of the data preprocessing function.

In the training and test loops, using these attributes also ensures that the correct information is recorded with each example when the preprocessing strategy is `per_batch`.

### Command line arguments

Command line arguments are defined in dataclasses in `core/model_arguments.py`, `core/data_training_arguments.py`, and `core/optimization_arguments.py`. They are parsed using a customized parser, defined in `core/parser.py`, and defined below.

#### `ModelArguments`

The `ModelArguments` dataclass accepts the following parameters:

- `model_name_or_path` (no default)
- `config_name`: default is `model_name_or_path`
- `cache_dir`: default is Hugging Face's cache dir
- `tokenizer_name`: default is `model_name_or_path`
- `use_fast_tokenizer`: default is `True`
- `model_revision`: default is `"main"`
- `token`: your Hugging Face access token, or the name of a file where it is stored. default is `False` (meaning no token is used)
- `use_gpu`: default is `False`
- `model_task`: default is `None`, and this is pulled from sets defined in `constants.py`. If you want to run on a model not classified in `constants.py` (e.g., a local model), set to one of `lm`, `mlm`, or `seq2seq`, depending on what type of model you're using.
- `model_modifier_fns`: A list of functions that modify the model object before pre-training. These functions are defined in `model_modifiers.py`, and must accept at least the `model` and `tokenizer`, and may accept additional kwargs. Currently, the only function implemented for this, `add_new_tokens`, adds new tokens to the model vocabulary prior to fine-tuning, using the tokenizer's `add_tokens` method.
- `model_modifier_fn_kwargs`: a dictionary whose keys are the names of the model modifier function to pass the corresponding dictionary of kwargs to.
- `model_pre_train_step_callbacks`: a list of callback classes whose `__call__()` method will be run, in order, after `loss.backward()` but before `optimizer.step()`. Callback classes' `__init__()` method must accept the `model` and `tokenizer` as arguments, and may accept additional kwargs. The `__call__()` method of a callback must accept only `epoch` and `batch` parameters. Currently callbacks implemented in `model_modifiers.py` are `PartiallyFrozenModelCallback`, `UnfreezeWordTokensCallback`, and `FreezeWordTokensCallback`. `PartiallyFrozenModelCallback` accepts either `frozen_params` or `unfrozen_params` (one only), a list of strings naming model parameters and indices to freeze/leave unfrozen prior to updating weights. It is assumed that if a param is not specified as frozen, it is unfrozen; and vice versa. Params and indices that are frozen will have their gradients replaced with zero before `optimizer.step()` is called. `UnfreezeWordTokensCallback` and `FreezeWordTokensCallback` accept a list of either `words_to_unfreeze` or `words_to_freeze`, and optionally `word_embeddings_name` (a string naming the parameter where the word embeddings of the model are found; if not provided, the callback attempts to determine this automatically). All words must be single tokens. The word embeddings of the corresponding words are either frozen (gradients set to 0), or the only parameters left unfrozen (all other parameters and indices in the word embeddings gradients are set to 0).
- `model_pre_train_step_callbacks_kwargs`: a dictionary whose keys are the name of the callback to pass the corresponding kwargs dict to during initialization.

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
- `per_device_train_batch_size`: default is `32`. Note that if you use a data preprocessing function with the `per_batch` strategy that changes the number of examples, this will be the batch size that is fed into the preprocessing function, and the actual number of examples run may be more or fewer.
- `max_validation_samples`: default is `None` (uses all examples)
- `per_device_validation_batch_size`: default is `32`. Note that if you use a data preprocessing function with the `per_batch` strategy that changes the number of examples, this will be the batch size that is fed into the preprocessing function, and the actual number of examples run may be more or fewer.
- `max_test_samples`: default is `None` (uses all examples). The same value applies to all test datasets.
- `per_device_test_batch_size`: default is `32`. The same value applies to all test datasets. Note that if you use a data preprocessing function with the `per_batch` strategy that changes the number of examples, this will be the batch size that is fed into the preprocessing function, and the actual number of examples run may be more or fewer.
- `ignore_pad_for_token_loss`: default is `True`. Currently ignored.
- `data_preprocessing_fn`: a dictionary mapping keys `train`, `validation`, `test` to the preprocessing function to use for the corresponding dataset(s). Default for each is `data_preprocessing.identity`.
- `data_preprocessing_fn_strategy`: a dictionary mapping keys `train`, `validation`, `test` to one of `per_batch` or `once`. The data preprocessing strategy will be run once over the whole corresponding dataset if `once` is provided. Otherwise, the preprocessing is recomputed each batch (useful, for instance, if you want to randomly mask a potentially different group of tokens every epoch, instead of masking them all once prior to fine-tuning).
- `data_preprocessing_fn_kwargs`: a dictionary mapping `train`, `validation`, `test` to a dict of kwargs to pass to the corresponding dataset's data preprocessing function.
- `epochs`: default is `250`
- `min_epochs`: even if `patience` is used, at least this many epochs will be completed before stopping. Default is `0`
- `patience`: the number of epochs to continue training without observing improved performance on the validation dataset. default is `None` (no early stopping)
- `delta`: the minimum amount of improvement on loss that resets the patience counter. Default is `0` (any improvement resets the patience counter)
- `train_optimizer`: a train optimizer class. Must accept `params=model.parameters()`, as well as additional kwargs.
- `train_optimizer_kwargs`: a dict of kwargs to pass to the train optimizer at initialization. Under the hood, the keys with these names are added directly to the `DataTrainingArguments` instance as well, and are pulled from there at class initialization, to facilitate hyperparameter selection using optuna (it is more straightforward to set the suggest values directly on the dataclass than in a dictionary in the dataclass). Note that this means if you are running a hyperparameter optimization study, the `train_optimizer_kwargs.lr` may print out as the default value, but the `DataTrainingArguments.lr` will give you the value suggested on that particular trial, and the value that is actually used.
- `loss_classes`: a dictionary mapping `train`, `validation` to a list of classes whose `forward` or `__call__()` method computes a loss value. The class must accept `model` and `tokenizer` arguments, and may accept additional kwargs. The `forward` or `__call__()` method must accept `outputs` and `labels` from a single batch, and returns a tensor containing a loss value. Currently classes are `OutputsDefaultLoss` (returns the default loss computed by `model(**inputs)` unchanged, and `KLBaselineLoss` (see below).
- `loss_reduction_fns`: a dictionary mapping `train`, `validation` to a function that accepts an unpacked tuple of tensors as inputs, and returns a single tensor as output, run to combine the loss values into a single value prior to a single training step. Default is `torch.sum`.
- `loss_classes_kwargs`: a dictionary mapping each type of dataset to a dict mapping the name of a loss class to the kwargs to pass to the corresponding class constructors. To facilitate optimization with optuna, these are underlying added to the `DataTrainingArguments` instance as `DatasetType_ClassName_Key`, and pulled from there during class initialization, to facilitate hyperparameter selection with optuna (it is more straightforward to set the suggest values directly on the dataclass than in a dictionary in the dataclass). Note that this means if you are running a hyperparameter optimization study, the `loss_classes_kwargs.train.KLBaselineLoss.scaleby` may print out as the default value, but the `DataTrainingArguments.train_KLBaselineLoss_scaleby` will give you the value suggested on that particular trial, and the value that is actually used.

- `output_dir`: Used to store the output directory name. This will be determined automatically.
- `test_output_file_prefix`: Set the prefix of the output file containing per-token surprisals for each sentence of each test dataset. Default is the model name, with "/" replaced by "-".
- `seed`: an integer to set the random seeds to for reproduceability. Default is given by `random.randint(0, 2**32-1)`.
- `save_tmp_test_files`: saves temporary `.json.gz` files that can be used to resume evaluation on a test dataset later during evaluation.

##### KLBaselineLoss

The `KLBaselineLoss` class computes a loss term based on the Kullbeck-Leibler divergence between the predictions of the model being fine-tuned and the baseline model (the model with the same name or path). This helps address catastrophic forgetting (see [Hawkins et al. 2020](https://aclanthology.org/2020.conll-1.33/) and [Wilson et al. 2023](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00608/118116/How-Abstract-Is-Linguistic-Generalization-in-Large)). Its constructor accepts the following kwargs (which can be passed via `--loss_classes_kwargs.{dataset_type}.KLBaselineLoss.{kwarg_name} {kwarg_value}`).

- `dataset`: the path(s) to the dataset(s) used to compute the KL divergence loss term. No default.
- `batch_size`: default is `32`. Note that if you use a data preprocessing function with the `per_batch` strategy that changes the number of examples, this will be the batch size that is fed into the preprocessing function, and the actual number of examples run may be more or fewer.
- `scaleby`: a multiplier for the KL divergence loss term, or a list of these multipliers for each `kl_dataset`. Default is `2.5`
- `n_examples_per_batch`: how many examples from `dataset` to use per weight update/batch to compute the KL divergence loss term. Can be a single `int`, or a list of `int`s for each `kl_dataset`. Default is `20`.
- `model_kwargs`: Dict of kwargs to pass to the baseline model constructor.
- `tokenizer_kwargs`: Dict of kwargs to pass to the baseline tokenizer constructor.
- `size_average`: passed to `torch.nn.modules.loss.KLDivLoss`.
- `reduce`: passed to `torch.nn.modules.loss.KLDivLoss`.
- `reduction`: default is `'none'`. It is recommended you don't modify this.
- `split_name`: the split name to use for each dataset, or a list of split names, one for each dataset. Default is `train`.
- `max_samples_per_dataset`: `int` or `list[int]`. default is `None` (all samples may be used).
- `max_sample_length`: the maximum allowable example length in number of tokens. Applies to all datasets.
- `preprocessing_num_workers`: how many workers to use when preprocessing the `dataset`s.
- `overwrite_cache`: whether to overwrite the cache when preprocessing the datasets.
- `data_preprocessing_fn`: the function to use to preprocess all datasets. See the section on data preprocessing above.
- `data_preprocessing_fn_kwargs`: see above.
- `data_preprocessing_fn_strategy`: see above.
- `return_all`: whether to return all the KL divergences for each example, or just the mean across all examples in the batch. Default is `False`.

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
	Any keyword arguments provided in `suggest_kwargs` for a particular parameter are passed to the `suggest_` function of the Optuna `trial` object. The default behavior is to suggest a value of the same type as the default value for the hyperparameter specified in `DataTrainingArguments`. However, if a `type` is provided for a hyperparameter's dictionary, and the argument provided has a default value that is either `float` or `int`, the `type` key can map to one of `"float"` or `"int"` to override the default behavior. (Non-numeric defaults can only be suggested as categorical values.) For instance, if the default value of the hyperparameter `lr` is set to `1` (for argument's sake), the default behavior would be to suggest an integer value when optimizing `lr`. However, if `--params.lr.type float` is set, a float in the range specified by `params.lr.values` will be suggested instead.

Currently, to optimize the hyperparameters `KLBaselineLoss.scaleby` and/or `KLBaselineLoss.n_examples_per_batch` differently across multiple datasets, you must pass a list for each with as many values as there are `KLBaselineLoss.dataset`s. The values in this list don't matter during optimization, as they will be overwritten with suggested values. If a list is not passed for these, a single value will be used for all datasets, and that will be the value optimized for.

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

- To pass a list of dicts (which shouldn't generally be needed, but may be for some obscure kwarg argument), you should pass it as a string with `json:` prepended. It will be parsed using `json.loads`.

## Outputs

For fine-tuning, the default output directory is `outputs/${train_file}/${model_name_or_path}/${year-month-day_hour_minute_second.nanoseconds}`.

If you are running only testing, the default output directory is `outputs/test_only/${model_name_or_path}`. This is because, in the general case, running a model on the same test dataset will return the same results, so there isn't typically a need to assign the model a unique id using a timestamp.

In either case, any `/` in `model_name_or_path` will be replaced with `-`.

When running an optimization study, outputs will be `config.txt`, a log file displaying the config options tried over the course of the study by the process whose outputs are in the associated directory, as well as `optimization_results.csv.gz`, a CSV containing the results of the study. This has the loss value associated with each combination of hyperparameters tried during the study. There will also be a file named `optimization_plots.pdf`, containing plots of the optimization study (described in the documentation for `optuna.visualization`).

When not running an optimization study, outputs will be `config.txt`, a log file displaying the config options used to fine-tune the model. If a training file and validation file are provided (i.e., if a model is being fine-tuned), the `model` subdirectory stores the model state that performed best on the validation dataset, and `tokenizer` stores the tokenizer. `metrics.csv.gz` is a CSV with per-token surprisals for every sentence for the test and validation datasets for every batch in every epoch. `loss_curves.pdf` plots the overall loss for train and validation datasets, as well as each training and validation loss when multiple losses are used, for all batches and epochs (for loss classes whose names end in `Loss`).

If a test file or files is provided, a CSV containing the per-token surprisal for each sentence in each test dataset will be output, with the default name being the model name (with "/" replaced with "-") + `-test_results.csv.gz` as the default name. The prefix can be changed by setting `--test_output_file_prefix`. Test datasets are ignored when `--do_optimize` is set, since hyperparameter selection shouldn't be evaluated against test sets.