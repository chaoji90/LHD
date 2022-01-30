from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_op = namedtuple('Genotype', 'normal normal_output reduce reduce_output')
#Genotype_op = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat normal_output normal_output_concat reduce_output reduce_output_concat')

PRIMITIVES = [
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

#DARTS = DARTS_V2

EX_DARTS_V1=Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

EPOCH_500=Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

EPOCH_332=Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 3), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

EPOCH_234=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

EPOCH_175=Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))

EPOCH_58=Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

EPOCH_134 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

EPOCH_42 = Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

EPOCH_20 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
EPOCH_22 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
EPOCH_23 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
EPOCH_24 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

EPOCH_46= Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
EPOCH_17=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
EPOCH_51=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))



EPOCH_56=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
EPOCH_11=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
EPOCH_28=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
EPOCH_44=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
EPOCH_64=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

EPOCH_11_bak=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
EPOCH_31=Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
EPOCH_21=Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))


DARTS=EPOCH_28

SPARSE_6=Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('skip_connect', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
SPARSE_7=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_3x3', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
SPARSE_8=Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_3x3', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
SPARSE_9=Genotype(normal=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
SPARSE_10=Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
SPARSE_11=Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
SPARSE_12=Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

