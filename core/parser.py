import sys
import dataclasses

from typing import (
	Any, Union,
	get_origin, get_args
)
from collections.abc import MutableMapping

def parse_args() -> dict[str,Any]:
	'''
	Parses command line arguments and returns them as a dictionary of key-value pairs.
	
	Argument names must be preceded by "--".
	
	If an argument value can be cast to int or float, it will be. To pass a number as a string,
	put quotes around it. Depending on your shell's behavior, you may need to escape the quotes
	with an additional set of quotes for this to work.
	
	If an argument name has a `.` in it, it will be parsed out as a nested dictionary, since
	`.` is not a valid part of a variable name in Python. For instance, 
	"--function_kwargs.variable_name test" will be parsed out as {..., "function_kwargs": {"variable_name": "test"}, ...}.
	
	If an argument value has a `.` in it, it will be treated as a module and imported. The object
	will be associated with the argument name, rather than the string naming the module.
	To pass an argument with a literal string '.' in it, put quotes around it. To pass an argument with
	quotes in the string, use two sets of quotes around it.
	
	If you want to import a module without a `.` in it to use as an argument value, put "import:" before the
	module name.
	
	If you want to use a predefined Python function as an argument, put `callable:` in front of it.
	
	If an argument value is "True" or "False", it will be cast to boolean.
	
	If an argument value is "None", it will be returned as `None`.
	
	If an argument name is not followed by a value (i.e., a string without "--" before it), the value of that
	argument will be boolean `True`.
	
	This allows you to conventiently and flexibly instantiate and pass dictionaries, booleans, None, Python modules, and
	Python objects as command-line arguments, instead of using a '.json' or a library that requires an external config file.
	'''
	def resolve_value(s: Union[str, list[str]]) -> Union[Any, list[Any]]:
		'''
		Resolves an argument's data type as follows.
		
		If a list is passed, resolves each argument's value as described below and 
		returns as a list.
		
		If an argument is "True" or "False", it is cast to boolean.
		
		If an argument is "None", `None` is returned.
		
		If an argument is "{}" or "[]", an empty dictionary or list is returned.
		
		If it isn't, it is cast to int.
		
		If it cannot, it is cast to float.
		
		If it cannot, and it has 'import:' at the beginning, the module with the name
		following 'import:' is imported.
		
		If it has dots in its name and it is not delimited by quotation marks, the 
		module with that name is imported.
		
		if it cannot, and it has 'callable:' at the beginning, it is replaced with
		a function of that name available in `globals()`.
		
		If it doesn't, and it has no dots in its name, it is returned (unchanged) as a 
		string.
		'''
		def import_class_from_string(path: str) -> Any:
			# from Pat @ https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
			from importlib import import_module
			module_path, _, class_name = path.rpartition('.')
			
			# this handles cases with dots
			if module_path != '':
				mod = import_module(module_path)
				klass = getattr(mod, class_name)
				return klass
			
			# this handles cases without dots
			mod = import_module(class_name)
			return mod
		
		def is_quoted(s: str) -> bool:
			'''Returns True if the passed string starts and ends with the same quotation marks.'''
			if s.startswith("'") and s.endswith("'"):
				return True
			
			if s.startswith('"') and s.endswith('"'):
				return True
			
			return False
		
		if isinstance(s, list):
			return [resolve_value(arg) for arg in s]
			
		# if s is already not a string, return it unchanged
		if not isinstance(s, str):
			return s
		
		if s in ['True', 'False']:
			return True if s == 'True' else False
		
		if s == 'None':
			return None
		
		if s == '{}':
			return {}
		
		if s == '[]':
			return []
		
		try:
			return int(s)
		except ValueError:
			pass
		
		try:
			return float(s)
		except ValueError:
			pass
		
		if s.startswith('import:'):
			s = s.lstrip('import:')
			return import_class_from_string(s)
		
		if '.' in s and not is_quoted(s):
			return import_class_from_string(s)
		
		if s.startswith('callable:'):
			s = s.lstrip('callable:')
			return globals()[s]
		
		# remove leading and trailing quotes
		if is_quoted(s):
			s = s[1:-1]
		
		return s
	
	def make_dict_from_string_list(l: list[str], value: Any = None) -> dict:
		'''
		Initializes a dictionary container from a list of strings. Each
		string is a key in a deeper nesting of the dictionary. The final value
		is set to None.
		'''
		# copy to avoid modifying passed argument in place
		l = l.copy()
		container = {}
		while l:
			container = {}
			container[l.pop()] = value
			value = container
		
		return container
	
	args = sys.argv[1:] # skip script name
	args_nested = []
	
	# associate names with strings to be values
	for arg in args:
		if arg.startswith('--'):
			args_nested.append([arg])
		else:
			args_nested[-1].append(arg)
	
	args_dict = {}
	for arg_list in args_nested:
		# strip off the argument identifier from the beginning
		name = arg_list[0].lstrip('--')
		arg_values = arg_list[1:]
		
		if len(arg_values) == 1:
			arg_values = arg_values[0]
		
		# if there's no value, assume it's True.
		if not arg_values:
			arg_values = True
		
		# if there's only one value, unpack it
		# for dictionaries, for the name, the desired name of the dictionary,
		# is passed followed by a dot to specify the key name within that dictionary.
		# this can be nested so we can have nested dictionaries. We resolve this here.
		# this logic does mean that dots cannot be used in strings that serve as dictionary
		# keys, which isn't strictly prohibited in Python, but it's a minor thing.
		if '.' in name:
			keys = name.split('.')
			if keys[0] not in args_dict:
				args_dict[keys[0]] = make_dict_from_string_list(l=keys[1:], value=resolve_value(arg_values))
				continue
			
			d = args_dict[keys[0]]
			keys = keys[1:]
			while keys:
				# we need to check whether this is empty first,
				# before changing what 'd' points to, since otherwise
				# it will always end up pointing to 'None'.
				# this leaves us with a reference to the deepest
				# dictionary with the same nesting as what we're after.
				check_dict = d.get(keys[0], None)
				if not check_dict:
					break
				d = check_dict
				_ = keys.pop(0)
			
			d.update(make_dict_from_string_list(l=keys, value=resolve_value(arg_values)))
			continue
		
		args_dict[name] = resolve_value(s=arg_values)
	
	return args_dict

def parse_args_into_dataclasses(*args) -> tuple:
	'''
	Uses parse_args to parse arguments, and then associates each argument with
	its named counterpart in one of the passed dataclasses. Dataclasses must
	not have duplicate field names.
	
	Handles overwriting nested dictionary values without overwriting the entire
	dictionary. To delete default fields in a dictionary, set the value to '__delete_field__'
	'''
	def delete_marked_fields(d: dict) -> None:
		'''
		Deletes fields from a nested dict with the value '__DeleteField__'.
		Used to remove default arguments from dictionaries.
		'''
		for k, v in list(d.items()):
			if v == '__delete_field__':
				del d[k]
			elif isinstance(v, dict):
				delete_marked_fields(d=v)
	
	def is_optional_dict(t) -> bool:
		origin = get_origin(t)
		args = get_args(t)
		if origin is not Union:
			return False
		
		if len(args) != 2:
			return False
		
		if type(None) in args and dict in args:
			return True
		
		return False
	
	def get_default_dict(field) -> dict:
		if field.type is not dict and not is_optional_dict(field.type):
			raise ValueError(
				f"It isn't possible to get a default value for the dictionary of {field.name!r}, since "
				f"it is {field.type}!"
			)
		
		if not hasattr(field, 'default_factory') or isinstance(field.default_factory, dataclasses._MISSING_TYPE):
			raise ValueError(
				f'No default factory was found for {field.name!r}! If you do not wish there to be any default values for the dictionary, '
				f'you should set the default_factory argument of the field to `lambda: {{}}`.'
			)
		
		return field.default_factory()
	
	def recursive_merge(d1: dict, d2: dict) -> dict:
		'''
		From https://stackoverflow.com/questions/7204805/deep-merge-dictionaries-of-dictionaries-in-python
		Update two dicts of dicts recursively, 
		if either mapping has leaves that are non-dicts, 
		the second's leaf overwrites the first's.
		'''
		for k, v in d1.items():
			if k in d2:
				# this next check is the only difference!
				if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
					d2[k] = recursive_merge(v, d2[k])
				# we could further check types and merge as appropriate here.
		d3 = d1.copy()
		d3.update(d2)
		return d3
	
	# check for duplicate names before doing anything else
	names = [f.name for dataklass in args for f in dataclasses.fields(dataklass)]
	if len(names) != len(set(names)):
		raise ValueError(
			'Dataclasses may not have duplicate names, but the following names were '
			'found in multiple dataclasses: '
			f'{set([name for name in [name for name in names if names.count(name) > 1]])}'
		)
	
	parsed_args = parse_args()
	
	dataklasses = []
	for dataklass in args:
		dataklass_kwargs = {}
		dataklass_kwargs['__dataclass__'] = dataklass
		for field in dataclasses.fields(dataklass):
			if field.name in parsed_args:
				dataklass_kwargs[field.name] = parsed_args[field.name]
				del parsed_args[field.name]
		
		dataklasses.append(dataklass_kwargs)
	
	dataklass_return = []
	for dataklass in dataklasses:
		klass = dataklass['__dataclass__']
		del dataklass['__dataclass__']
		
		# since we've removed the class name, what's left are the arguments
		# we rename this variable for clarity.
		dataklass_args = dataklass
		
		if not dataklass_args:
			dataklass_return.append(klass())
			continue
		
		for k, v in dataklass_args.items():
			fields = dataclasses.fields(klass)
			for field in fields:
				if (field.type is dict or is_optional_dict(field.type)) and field.name == k:
					dataklass_args[k] = recursive_merge(get_default_dict(field), v)
		
		delete_marked_fields(dataklass_args)
		dataklass_return.append(klass(**{k: v for k, v in dataklass_args.items()}))
	
	if parsed_args:
		dataklass_return.append(parsed_args)
	
	dataklass_return = tuple(dataklass_return)
	return dataklass_return