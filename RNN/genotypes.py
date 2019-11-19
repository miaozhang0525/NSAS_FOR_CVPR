from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

ENAS = Genotype(
    recurrent = [
        ('tanh', 0),
        ('tanh', 1),
        ('relu', 1),
        ('tanh', 3),
        ('tanh', 3),
        ('relu', 3),
        ('relu', 4),
        ('relu', 7),
        ('relu', 8),
        ('relu', 8),
        ('relu', 8),
    ],
    concat = [2, 5, 6, 9, 10, 11]
)

Random_NSAS=Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 0), ('tanh', 2), ('tanh', 2), ('sigmoid', 3), ('relu', 4), ('tanh', 3), ('relu', 5)], concat=range(1, 9))
