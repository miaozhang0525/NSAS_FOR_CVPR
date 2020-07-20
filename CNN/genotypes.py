from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

STEP=4

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
XNAS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1),
        ('dil_conv_5x5', 4),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_3x3', 3),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=range(2, 6)
)

Random_NSAS=Genotype(
	normal=[
	('sep_conv_5x5', 1), ('dil_conv_3x3', 0), 
	('dil_conv_3x3', 1), ('sep_conv_3x3', 0), 
	('sep_conv_5x5',1), ('avg_pool_3x3', 0), 
	('skip_connect', 0), ('none', 1)],
	normal_concat=[2, 3, 4, 5], 
	reduce=[
	('dil_conv_3x3', 0), ('none', 1), 
	('dil_conv_3x3', 0), ('avg_pool_3x3', 1), 
	('dil_conv_5x5', 2), ('dil_conv_5x5', 3), 
	('none', 0), ('sep_conv_3x3', 3)], 
	reduce_concat=[2, 3, 4, 5])
	
	
Random_NSAS_C=Genotype(
	normal=[
	('sep_conv_3x3', 1), ('dil_conv_3x3', 0), 
	('dil_conv_3x3', 2), ('sep_conv_3x3', 0), 
	('sep_conv_3x3',3), ('skip_connect', 0), 
	('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], 
	normal_concat=[2, 3, 4, 5], 
	reduce=[
	('dil_conv_3x3', 0), ('max_pool_3x3', 1),
	('dil_conv_3x3', 0), ('avg_pool_3x3', 2), 
	('dil_conv_3x3', 0), ('sep_conv_3x3', 3), 
	('avg_pool_3x3', 0), ('dil_conv_5x5', 4)],
	reduce_concat=[2, 3, 4, 5])

