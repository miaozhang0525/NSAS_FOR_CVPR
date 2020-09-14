from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


STEPS = 4

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

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

ENNAS=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 3)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('none', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('none', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])


ENNAS_PR=Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 3), ('none', 3), ('none', 3), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('none', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))




EENAS_1=Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 3), ('max_pool_3x3', 3), ('sep_conv_3x3', 3), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))


ENNAS_vf1=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('none', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('none', 0), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])



ENNAS_vf=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('none', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('none', 0), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])


research_1=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3',0), ( 'avg_pool_3x3',0), ('skip_connect',0), ( 'sep_conv_3x3',0), ('sep_conv_5x5',1), ('avg_pool_3x3',0 ), ( 'avg_pool_3x3',0), ('sep_conv_5x5',1)], reduce_concat=range(2, 6))
research_2=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5',0, ), ( 'sep_conv_5x5',0), ('sep_conv_3x3',0), ('none',1), ( 'avg_pool_3x3',0), ( 'sep_conv_5x5',0), ('none',2), ('sep_conv_3x3',1)], reduce_concat=range(2, 6))


research_3=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('none', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('none', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

research_4=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('none', 0), ('avg_pool_3x3', 0), ('none', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))



research_5=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('none', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))



research_6=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))





research_7=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))



research_8=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('none', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('none', 2)], reduce_concat=range(2, 6))



research_9=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('none', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))



research_0=Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

ENNAS_D=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

ENNAS_D2=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

ENNAS_D3=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


ENNAS_D4=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

#ENNAS_D2=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2),  ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])


EENAS_2=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])



Random_NSAS=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5',1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('none', 1)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('none', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('none', 0), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])



Random_NSAS_v1=Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5',3), ('avg_pool_3x3', 0), ('skip_connect', 0), ('dil_conv_5x5', 4)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])

PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


PCDARTS = PC_DARTS_cifar

Random_NSAS_v6=Genotype(
				normal=[
				('sep_conv_3x3', 1), ('dil_conv_3x3', 0), 
				('dil_conv_5x5', 2), ('sep_conv_3x3', 0), 
				('sep_conv_3x3',3), ('skip_connect', 0), 
				('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], 
				normal_concat=[2, 3, 4, 5], 
				reduce=[
				('dil_conv_3x3', 0), ('max_pool_3x3', 1), 
				('dil_conv_3x3', 0), ('avg_pool_3x3', 2), 
				('dil_conv_3x3', 0), ('sep_conv_3x3', 3), 
				('avg_pool_3x3', 1), ('dil_conv_5x5', 4)], 
				reduce_concat=[2, 3, 4, 5])


Random_NSAS_v5=Genotype(
				normal=[
				('sep_conv_3x3', 1), ('dil_conv_3x3', 0), 
				('dil_conv_3x3', 2), ('sep_conv_3x3', 0), 
				('sep_conv_3x3', 3), ('skip_connect', 0), 
				('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], 
				normal_concat=[2, 3, 4, 5], 
				reduce=[
				('dil_conv_3x3', 0), ('max_pool_3x3', 1), 
				('dil_conv_3x3', 0), ('avg_pool_3x3', 2), 
				('dil_conv_3x3', 0), ('sep_conv_3x3', 3), 
				('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], 
				reduce_concat=[2, 3, 4, 5])

Random_NSAS_v7=Genotype(
				normal=[
				('sep_conv_3x3', 1), ('dil_conv_5x5', 0), 
				('dil_conv_5x5', 2), ('sep_conv_3x3', 0), 
				('sep_conv_3x3',3), ('skip_connect', 0), 
				('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], 
				normal_concat=[2, 3, 4, 5], 
				reduce=[
				('dil_conv_3x3', 0), ('max_pool_3x3', 1), 
				('dil_conv_5x5', 0), ('avg_pool_3x3', 2), 
				('dil_conv_3x3', 0), ('sep_conv_3x3', 3), 
				('avg_pool_3x3', 1), ('dil_conv_5x5', 4)], 
				reduce_concat=[2, 3, 4, 5])

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

Random_NSAS_v8=Genotype(
				normal=[
				('sep_conv_3x3', 1), ('dil_conv_5x5', 0), 
				('dil_conv_5x5', 2), ('sep_conv_3x3', 0), 
				('sep_conv_3x3',3), ('skip_connect', 0), 
				('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], 
				normal_concat=[2, 3, 4, 5], 
				reduce=[
				('dil_conv_5x5', 0), ('max_pool_3x3', 1), 
				('dil_conv_5x5', 0), ('avg_pool_3x3', 2), 
				('dil_conv_5x5', 0), ('max_pool_3x3', 3), 
				('avg_pool_3x3', 1), ('dil_conv_5x5', 4)], 
				reduce_concat=[2, 3, 4, 5])
				
Random_NSAS_v16 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('dil_conv_5x5', 0),
        ('sep_conv_3x3', 2),
        ('dil_conv_5x5', 0),
        ('sep_conv_3x3', 3),
        ('dil_conv_5x5', 1),
        ('sep_conv_3x3', 4),
        ('skip_connect', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)



Random_NSAS_v17 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('dil_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('skip_connect', 1),
        ('sep_conv_3x3', 4),
        ('skip_connect', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)



Random_NSAS_v18 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('dil_conv_5x5', 0),
        ('sep_conv_3x3', 3),
        ('skip_connect', 1),
        ('sep_conv_3x3', 4),
        ('skip_connect', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)





Random_NSAS_v19 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('dil_conv_5x5', 1),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 4),
        ('skip_connect', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)


Random_NSAS_v20 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 4),
        ('skip_connect', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=range(2, 6)
)





Random_NSAS_v21 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 4),
        ('skip_connect', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=range(2, 6)
)
