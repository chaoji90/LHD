import torch
import torch.nn as nn
from operations_five_dr import *
from torch.autograd import Variable
#from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction:
      #self.preprocess0 = FactorizedReduce(C_prev_prev, C)
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    else:
      #self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
      self.preprocess0 =Identity()
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      #zip按照位置打包
      op_names, indices, steps_list = zip(*genotype.reduce)
      #concat = genotype.reduce_concat
      self.output_path=genotype.reduce_output
    else:
      op_names, indices, steps_list = zip(*genotype.normal)
      #concat = genotype.normal_concat
      self.output_path=genotype.normal_output
    self._compile(C, op_names, indices, reduction, steps_list)

  def _compile(self, C, op_names, indices, reduction, steps_list):
    assert len(op_names) == len(indices)
    self._steps = steps_list[-1]+1
    #self._concat = concat
    self.multiplier = len(self.output_path['concat'])
    #operation counts for each step
    self._steps_counts=[steps_list.count(i) for i in range(self._steps)]
    #print(self._steps_counts)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    #feature maps index in concat_states for feedforward
    self._indices = indices
    #print(self._steps_counts)
    #print(self._indices)
    #exit(0)

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    concat_states = [s0, s1]
    concat_output=[]
    sums = 0
    #start index in _indices for each step
    start=0
    for i in range(self._steps):
        if self._steps_counts[i]==0: 
            concat_states.append(0)
            continue
        state_temp=0
        for j in range(self._steps_counts[i]):
            h=concat_states[self._indices[start+j]]
            op=self._ops[start+j]
            h=op(h)
            #if self.training and drop_prob > 0.:
                #if not isinstance(op, Identity):
                    #h=drop_path(h, drop_prob)
            state_temp+=h
        concat_states+=[state_temp]
        #next step start index in _indices
        start+=self._steps_counts[i]
        if i in self.output_path['concat']: concat_output+=[state_temp]
        if i in self.output_path['sum']: sums+=state_temp
    return sums, torch.cat(concat_output, dim=1)
    '''
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)
    '''


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem_1 = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    self.stem_0 = nn.Sequential(
      nn.Conv2d(3, C, 3, padding=1, bias=False),
      nn.BatchNorm2d(C)
    )
    
    #C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    C_prev_prev, C_prev, C_curr = C, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      reduction_prev = reduction
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      self.cells += [cell]
      C_prev_prev, C_prev = C_curr, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev+C_curr

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev+C_curr, num_classes)

  def forward(self, inputd):
    logits_aux = None
    s0 = self.stem_0(inputd)
    s1 = self.stem_1(inputd)
    for i, cell in enumerate(self.cells):
      #s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      s0, s1 = cell(s0, s1)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          s_a=torch.cat([s0,s1], dim=1)
          logits_aux = self.auxiliary_head(s_a)
    s1=torch.cat([s0,s1], dim=1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, inputd):
    logits_aux = None
    s0 = self.stem0(inputd)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

