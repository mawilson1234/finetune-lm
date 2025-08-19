verbs = {
	'give':  {'base': 'give',  'past': 'gave',   'ppart': 'given',  'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'hand':  {'base': 'hand',  'past': 'handed', 'ppart': 'handed', 'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'teach': {'base': 'teach', 'past': 'taught', 'ppart': 'taught', 'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'tell':  {'base': 'tell',  'past': 'told',   'ppart': 'told',   'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'send':  {'base': 'send',  'past': 'sent',   'ppart': 'sent',   'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'mail':  {'base': 'mail',  'past': 'mailed', 'ppart': 'mailed', 'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'throw': {'base': 'throw', 'past': 'threw',  'ppart': 'thrown', 'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	'toss':  {'base': 'toss',  'past': 'tossed', 'ppart': 'tossed', 'p1': 'to', 'p2': '', 'alternation': 'dative', 'part': 'out'},
	
	'spray':  {'base': 'spray',  'past': 'sprayed',  'ppart': 'sprayed',  'p1': 'onto', 'p2': 'with', 'alternation': 'sl', 'part': 'out'},
	'shower': {'base': 'shower', 'past': 'showered', 'ppart': 'showered', 'p1': 'onto', 'p2': 'with', 'alternation': 'sl', 'part': 'out'},
	'rub':    {'base': 'rub',    'past': 'rubbed',   'ppart': 'rubbed',   'p1': 'onto', 'p2': 'with', 'alternation': 'sl', 'part': 'out'},
	'dab':    {'base': 'dab',    'past': 'dabbed',   'ppart': 'dabbed',   'p1': 'onto', 'p2': 'with', 'alternation': 'sl', 'part': 'out'},
	'load':   {'base': 'load',   'past': 'loaded',   'ppart': 'loaded',   'p1': 'onto', 'p2': 'with', 'alternation': 'sl', 'part': 'out'},
	'pack':   {'base': 'pack',   'past': 'packed',   'ppart': 'packed',   'p1': 'into', 'p2': 'with', 'alternation': 'sl', 'part': 'up'},
	'stock':  {'base': 'stock',  'past': 'stocked',  'ppart': 'stocked',  'p1': 'on',   'p2': 'with', 'alternation': 'sl', 'part': 'out'},
	'stuff':  {'base': 'stuff',  'past': 'stuffed',  'ppart': 'stuffed',  'p1': 'into', 'p2': 'with', 'alternation': 'sl', 'part': 'up'}
}

ref_verb_alts = {
	'dative': ['give', 'send', 'mail'], 
	'sl':     ['spray', 'load']
}

masked_token_targets = {
	'dative': {'THAX' : [
							'apple', 'book',   'chair',  'table',  'phone',    'shoe',   'water', 'drink',  'cup',   'flower', 
							'plate', 'bottle', 'desk',   'watch',  'schedule', 'guitar', 'cloth', 'game',   'straw', 'ball',   
							'jar',   'mask',   'hat',    'lesson', 'couch',    'button', 'box',   'story',  'wire',  'paper',  'calendar'
						],
			   'RICKET' : [
			   				'person',  'man',     'woman',    'student', 'teacher', 'king', 'queen',  'prince',  'princess', 'writer', 
			   				'author',  'builder', 'driver',   'human',   'dog',     'bird', 'dancer', 'player',  'angel',    'actor', 
			   				'actress', 'singer',  'director', 'bee',     'friend',  'wolf', 'lion',   'scholar', 'pirate',   'spirit', 'fox'
			   			]
			   },
			   
	'sl': 	  {'THAX' : [
							'paint', 'cream', 'water', 'fuel', 'hay', 'rice', 'oil', 'solvent', 'powder', 'trash', 'beer', 
							'sauce', 'gas', 'icing', 'blood', 'dye', 'wax', 'grain', 'corn', 'bread', 'jelly', 'bacon', 
							'mud', 'grease', 'gravel', 'wood', 'chocolate', 'milk', 'dust', 'ice', 'gold'
						],
			   'GORX' : [
			   				'truck', 'wall', 'field', 'door', 'wagon', 'floor', 'sidewalk', 'porch', 'roof', 'car', 'fence', 
			   				'ceiling', 'tub', 'factory', 'counter', 'oven', 'surface', 'window', 'room', 'piece', 'thing', 
			   				'bottle', 'screen', 'stove', 'house', 'building', 'van', 'cart', 'trunk', 'deck', 'garden'
			   			]
				}
}

masked_token_target_labels = {
	'dative': {'THAX' : 'inanimate', 'RICKET' : 'animate'},
	'sl' :    {'THAX' : 'mass', 'GORX' : 'count'}
}

types = {
	'dative': {'type1': 'PD',           'type2': 'DO', 'spectype': '2-object', 'spectype2': '2'}, 
	'sl':     {'type1': 'theme-object', 'type2': 'goal-object', 'spectype': 'P-object', 'spectype2': 'P'}
}

sentence_types = [
	'[type1] active',
	'[type1] passive',
	'[type2] active',
	'[type2] passive',
	'polar Q [type1] active',
	'polar Q [type1] passive',
	'polar Q [type2] active',
	'polar Q [type2] passive',
	'raising [type1] active',
	'raising [type1] passive',
	'raising [type2] active',
	'raising [type2] passive',
	'mat-wh-Q subject raising [type1] active',
	'mat-wh-Q object raising [type1] active',
	'mat-wh-Q P-object raising [type1] active',
	'mat-wh-Q subject raising [type1] passive',
	'mat-wh-Q P-object raising [type1] passive',
	'mat-wh-Q subject raising [type2] active',
	'mat-wh-Q object raising [type2] active (amb)',
	'mat-wh-Q [spectype] raising [type2] active (amb)',
	'mat-wh-Q subject raising [type2] passive',
	'mat-wh-Q [spectype] raising [type2] passive',
	'emb-wh-Q subject raising [type1] active',
	'emb-wh-Q object raising [type1] active',
	'emb-wh-Q P-object raising [type1] active',
	'emb-wh-Q subject raising [type1] passive',
	'emb-wh-Q P-object raising [type1] passive',
	'emb-wh-Q subject raising [type2] active',
	'emb-wh-Q object raising [type2] active (amb)',
	'emb-wh-Q [spectype] raising [type2] active (amb)',
	'emb-wh-Q subject raising [type2] passive',
	'emb-wh-Q [spectype] raising [type2] passive',
	'neg [type1] active',
	'neg [type1] passive',
	'neg [type2] active',
	'neg [type2] passive',
	'cleft subject [type1] active',
	'cleft object [type1] active',
	'cleft P-object [type1] active',
	'cleft subject [type1] passive',
	'cleft P-object [type1] passive',
	'cleft subject [type2] active',
	'cleft object [type2] active (amb)',
	'cleft [spectype] [type2] active (amb)',
	'cleft subject [type2] passive',
	'cleft [spectype] [type2] passive',
	'mat-wh-Q subject [type1] active',
	'mat-wh-Q object [type1] active',
	'mat-wh-Q P-object [type1] active',
	'mat-wh-Q subject [type1] passive',
	'mat-wh-Q P-object [type1] passive',
	'mat-wh-Q subject [type2] active',
	'mat-wh-Q object [type2] active (amb)',
	'mat-wh-Q [spectype] [type2] active (amb)',
	'mat-wh-Q subject [type2] passive',
	'mat-wh-Q [spectype] [type2] passive',
	'emb-wh-Q subject [type1] active',
	'emb-wh-Q object [type1] active',
	'emb-wh-Q P-object [type1] active',
	'emb-wh-Q subject [type1] passive',
	'emb-wh-Q P-object [type1] passive',
	'emb-wh-Q subject [type2] active',
	'emb-wh-Q object [type2] active (amb)',
	'emb-wh-Q [spectype] [type2] active (amb)',
	'emb-wh-Q subject [type2] passive',
	'emb-wh-Q [spectype] [type2] passive',
	'V Part Obj [type1] active',
	'V Obj Part [type1] active',
	'V Part [type1] passive',
	'SRC [type1] active',
	'ORC [type1] active',
	'PORC [type1] active',
	'SRC [type1] passive',
	'PORC [type1] passive',
	'SRC [type2] active',
	'ORC [type2] active (amb)',
	'[spectype2]ORC [type2] active (amb)',
	'SRC [type2] passive',
	'[spectype2]ORC [type2] passive']

subj_nouns = {
	'dative': ['teacher', 'student', 'person', 'doctor', 'man', 'woman', 'author', 'actor'],
	'sl':     ['teacher', 'worker', 'student', 'dentist', 'teacher', 'student', 'man', 'woman', 'man', 'woman', 'person', 'doctor']
}

args1 = {
	'dative': ['the THAX', 'a THAX'], 
	'sl':     ['the THAX', 'some THAX', 'THAX']
}

args2 = {
	'dative': ['the RICKET', 'a RICKET'], 
	'sl':     ['the GORX', 'a GORX']
}

arg_types = {
	'THAX' :   'theme',
	'RICKET' : 'recipient',
	'GORX' :   'goal'
}

ref_sentence_types = [
	'[verbbase] [type2] active',
	'[verbbase] [type1] active']

ref_templates = [
	'the [noun] [verbpast] [arg2] [p2] [arg1].',
	'the [noun] [verbpast] [arg1] [p1] [arg2].']

templates = [
	'the [noun] [verbpast] [arg1] [p1] [arg2].',
	'[arg1] was [verbpart] [p1] [arg2].',
	'the [noun] [verbpast] [arg2] [p2] [arg1].',
	'[arg2] was [verbpart] [p2] [arg1].',

	'did the [noun] [verbbase] [arg1] [p1] [arg2]?',
	'was [arg1] [verbpart] [p1] [arg2]?',
	'did the [noun] [verbbase] [arg2] [p2] [arg1]?',
	'was [arg2] [verbpart] [p2] [arg1]?',

	'the [noun] seems to have [verbpart] [arg1] [p1] [arg2].',
	'[arg1] seems to have been [verbpart] [p1] [arg2].',
	'the [noun] seems to have [verbpart] [arg2] [p2] [arg1].',
	'[arg2] seems to have been [verbpart] [p2] [arg1].',

	'which [noun] seems to have [verbpart] [arg1] [p1] [arg2]?',
	'which [arg1] does the [noun] seem to have [verbpart] [p1] [arg2]?',
	'which [arg2] does the [noun] seem to have [verbpart] [arg1] [p1]?',
	'which [arg1] seems to have been [verbpart] [p1] [arg2]?',
	'which [arg2] does [arg1] seem to have been [verbpart] [p1]?',

	'which [noun] seems to have [verbpart] [arg2] [p2] [arg1]?',
	'which [arg2] does the [noun] seem to have [verbpart] [p2] [arg1]?',
	'which [arg1] does the [noun] seem to have [verbpart] [arg2] [p2]?',
	'which [arg2] seems to have been [verbpart] [p2] [arg1]?',
	'which [arg1] does [arg2] seem to have been [verbpart] [p2]?',

	'I wonder which [noun] seems to have [verbpart] [arg1] [p1] [arg2].',
	'I wonder which [arg1] the [noun] seems to have [verbpart] [p1] [arg2].',
	'I wonder which [arg2] the [noun] seems to have [verbpart] [arg1] [p1].',
	'I wonder which [arg1] seems to have been [verbpart] [p1] [arg2].',
	'I wonder which [arg2] [arg1] seems to have been [verbpart] [p1].',

	'I wonder which [noun] seems to have [verbpart] [arg2] [p2] [arg1].',
	'I wonder which [arg2] the [noun] seems to have [verbpart] [p2] [arg1].',
	'I wonder which [arg1] the [noun] seems to have [verbpart] [arg2] [p2].',
	'I wonder which [arg2] seems to have been [verbpart] [p2] [arg1].',
	'I wonder which [arg1] [arg2] seems to have been [verbpart] [p2].',

	"the [noun] didn't [verbbase] [arg1] [p1] [arg2].",
	"[arg1] wasn't [verbpart] [p1] [arg2].",
	"the [noun] didn't [verbbase] [arg2] [p2] [arg1].",
	"[arg2] wasn't [verbpart] [p2] [arg1].",

	"it was the [noun] that [verbpast] [arg1] [p1] [arg2].",
	"it was [arg1] that the [noun] [verbpast] [p1] [arg2].",
	"it was [arg2] that the [noun] [verbpast] [arg1] [p1].",
	"it was [arg1] that was [verbpart] [p1] [arg2].",
	"it was [arg2] that [arg1] was [verbpart] [p1].",

	"it was the [noun] that [verbpast] [arg2] [p2] [arg1].",
	"it was [arg2] that the [noun] [verbpast] [p2] [arg1].",
	"it was [arg1] that the [noun] [verbpast] [arg2] [p2].",
	"it was [arg2] that was [verbpart] [p2] [arg1].",
	"it was [arg1] that [arg2] was [verbpart] [p2].",

	"which [noun] [verbpast] [arg1] [p1] [arg2]?",
	"which [arg1] did the [noun] [verbbase] [p1] [arg2]?",
	"which [arg2] did the [noun] [verbbase] [arg1] [p1]?",
	"which [arg1] was [verbpart] [p1] [arg2]?",
	"which [arg2] was [arg1] [verbpart] [p1]?",

	"which [noun] [verbpast] [arg2] [p2] [arg1]?",
	"which [arg2] did the [noun] [verbbase] [p2] [arg1]?",
	"which [arg1] did the [noun] [verbbase] [arg2] [p2]?",
	"which [arg2] was [verbpart] [p2] [arg1]?",
	"which [arg1] was [arg2] [verbpart] [p2]?",

	"I wonder which [noun] [verbpast] [arg1] [p1] [arg2].",
	"I wonder which [arg1] the [noun] [verbpast] [p1] [arg2].",
	"I wonder which [arg2] the [noun] [verbpast] [arg1] [p1].",
	"I wonder which [arg1] was [verbpart] [p1] [arg2].",
	"I wonder which [arg2] [arg1] was [verbpart] [p1].",

	"I wonder which [noun] [verbpast] [arg2] [p2] [arg1].",
	"I wonder which [arg2] the [noun] [verbpast] [p2] [arg1].",
	"I wonder which [arg1] the [noun] [verbpast] [arg2] [p2].",
	"I wonder which [arg2] was [verbpart] [p2] [arg1].",
	"I wonder which [arg1] [arg2] was [verbpart] [p2].",

	'the [noun] [verbpast] [part] [arg1] [p1] [arg2].',
	'the [noun] [verbpast] [arg1] [part] [p1] [arg2].',
	'[arg1] was [verbpart] [part] [p1] [arg2].',

	"the [noun] that [verbpast] [arg1] [p1] [arg2] was everyone's favorite.",
	"[arg1] that the [noun] [verbpast] [p1] [arg2] was everyone's favorite.",
	"[arg2] that the [noun] [verbpast] [arg1] [p1] was everyone's favorite.",
	"[arg1] that was [verbpart] [p1] [arg2] was everyone's favorite.",
	"[arg2] that [arg1] was [verbpart] [p1] was everyone's favorite.",

	"the [noun] that [verbpast] [arg2] [p2] [arg1] was everyone's favorite.",
	"[arg2] that the [noun] [verbpast] [p2] [arg1] was everyone's favorite.",
	"[arg1] that the [noun] [verbpast] [arg2] [p2] was everyone's favorite.",
	"[arg2] that was [verbpart] [p2] [arg1] was everyone's favorite.",
	"[arg1] that [arg2] was [verbpart] [p2] was everyone's favorite."]

def make_data(vs: str = None):
	if not vs or vs is None:
		vs = [v for alt in ref_verb_alts for v in ref_verb_alts[alt]]
	else:
		vs = vs.split(',')
	
	for alt in ref_verb_alts:
		arg_combos = [(arg1, arg2) for arg2 in args2[alt] for arg1 in args1[alt]]
		arg_combos = [arg_combo for arg_combo in arg_combos for _ in range(2)]
		
		nouns_args = tuple(zip(subj_nouns[alt], arg_combos))
		
		ref_verbs = ref_verb_alts[alt]
		for ref_verb in ref_verbs:
			if ref_verb in vs:
				for verb in verbs:
					stypes = sentence_types.copy()
					if alt == verbs[verb]['alternation']:
						all_sentences = []
						for noun, (arg1, arg2) in nouns_args:
							sentences = []
							for template in templates:
								s = template.replace('[noun]', noun)
								s = s.replace('[verbpast]', verbs[verb]['past'])
								s = s.replace('[verbbase]', verbs[verb]['base'])
								s = s.replace('[verbpart]', verbs[verb]['ppart'])
								s = s.replace('[arg1]', arg1)
								s = s.replace('[arg2]', arg2)
								s = s.replace('[part]', verbs[verb]['part'])
								s = s.replace('[p1]', verbs[verb]['p1'])
								s = s.replace('[p2]', verbs[verb]['p2'])
								s = s.replace('  ', ' ')
								s = s.replace(' ?', '?')
								s = s.replace(' .', '.')
								s = s.replace('which the ', 'which ').replace('which a ', 'which ')
								if s.startswith('THAX'):
									s = 'Today, ' + s
								
								s = s[:1].upper() + s[1:]
								sentences += [s]
							
							if not ref_verb == verb:
								for ref_template in ref_templates:
									s = ref_template.replace('[noun]', noun)
									s = s.replace('[verbpast]', verbs[ref_verb]['past'])
									s = s.replace('[verbbase]', verbs[ref_verb]['base'])
									s = s.replace('[verbpart]', verbs[ref_verb]['ppart'])
									s = s.replace('[arg1]', arg1)
									s = s.replace('[arg2]', arg2)
									s = s.replace('[part]', verbs[verb]['part'])
									s = s.replace('[p1]', verbs[verb]['p1'])
									s = s.replace('[p2]', verbs[verb]['p2'])
									s = s.replace('  ', ' ')
									s = s.replace(' ?', '?')
									s = s.replace(' .', '.')
									s = s.replace('which the', 'which').replace('which a', 'which')
									if s.startswith('THAX'):
										s = 'Today, ' + s
									
									s = s[:1].upper() + s[1:]
									sentences = [s] + sentences
									
							sentences = ' , '.join(sentences)
							all_sentences += [sentences]
						
						all_sentences = '\n'.join(all_sentences)
						
						if not ref_verb == verb:
							for ref_stype in ref_sentence_types:
								stypes = [ref_stype.replace('[verbbase]', verbs[ref_verb]['base'])] + stypes
						else:
							stypes = [verbs[ref_verb]['base'] + ' ' + stype if '[verbbase] ' + stype in ref_sentence_types else stype for stype in stypes]
						
						stypes = [stype.replace('[type1]', types[verbs[verb]['alternation']]['type1']) \
									   .replace('[type2]', types[verbs[verb]['alternation']]['type2']) \
									   .replace('[spectype]', types[verbs[verb]['alternation']]['spectype']) \
									   .replace('[spectype2]', types[verbs[verb]['alternation']]['spectype2'])
								for stype in stypes]
						
						if alt == 'sl':
							stypes = [stype.replace(' (amb)', '') for stype in stypes]
						
						with open(f'data/syn_{verb}_{ref_verb}_ext.data', 'w') as f:
							f.write(all_sentences)
							
						with open(f'conf/data/syn_{verb}_{ref_verb}_ext.yaml', 'w') as f:
							f.write(f'# Synthetic constructions with {verb}\n\n')
							f.write(f'name: syn_{verb}_{ref_verb}_ext.data\n')
							f.write(f'description: Synthetic \'{verb}\' tuples\n')
							f.write('exp_type: newarg\n\n')
							f.write('sentence_types:\n  - ')
							f.write('\n  - '.join(stypes) + '\n\n')
							f.write('eval_groups:\n')
							f.write('  ' + arg_types[args1[alt][0].replace('the ', '')] + ' : ' + args1[alt][0].replace('the ', '') + '\n')
							f.write('  ' + arg_types[args2[alt][0].replace('the ', '')] + ' : ' + args2[alt][0].replace('the ', '') + '\n\n')
							f.write('to_mask:\n')
							f.write('  - ' + args1[alt][0].replace('the ', '') + '\n')
							f.write('  - ' + args2[alt][0].replace('the ', '') + '\n\n')
							f.write('masked_token_targets:\n')
							f.write('  ' + args1[alt][0].replace('the ', '') + ' : [' + ', '.join(masked_token_targets[alt][args1[alt][0].replace('the ', '')]) + ']\n')
							f.write('  ' + args2[alt][0].replace('the ', '') + ' : [' + ', '.join(masked_token_targets[alt][args2[alt][0].replace('the ', '')]) + ']\n\n')
							f.write('masked_token_target_labels:\n')
							f.write('  ' + args1[alt][0].replace('the ', '') + ' : ' + masked_token_target_labels[alt][args1[alt][0].replace('the ', '')] + '\n')
							f.write('  ' + args2[alt][0].replace('the ', '') + ' : ' + masked_token_target_labels[alt][args2[alt][0].replace('the ', '')])

if __name__ == '__main__':
	import sys
	vs = sys.argv[-1] if not sys.argv[-1] == 'datamaker.py' else ''
	make_data(vs)