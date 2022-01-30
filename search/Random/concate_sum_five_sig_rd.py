import torch
import torch.nn as nn
import torch.nn.functional as F
from operations_five_dr import *
#from operations_five import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype_op
import numpy as np
from scipy.stats import entropy

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      #if 'pool' in primitive:
        #op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    #使用softmax参数对op的输出进行加权求和
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  #def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
  #cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    #由于每个cell有两个input边，因此reduction后一个节点除了考虑直接输入外，还需要下采样一下reduction cell前一个cell的输出
    if reduction_prev:
      #self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    else:
      #self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
      self.preprocess0 =Identity()
	#把输入都处理为C通道
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    #每个Cell的内部节点数量为4
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    #step默认为4,i=0,j=0-1,2;i=1,j=0-2,3;i=2,j=0-3,4;i=3,j=0-4,5
	#2+3+4+5=14
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights, outpath_weights):
    #print(f"{s0.shape};{s1.shape}")
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    #print(f"{s0.shape};{s1.shape}")

    #sum:0, concat:1 within operation parameters
    concat_states = [s0, s1]
    sums = 0
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(concat_states))
      #sum:0, concat:1 within output path
      concat=outpath_weights[1,i]*s
      offset += len(concat_states)
      concat_states.append(concat)
      sums+=outpath_weights[0,i]*s

    #默认multiplier为4,把cell中4个step的输出全部cat到一起，作为整个cell的输出
    return sums,torch.cat(concat_states[-self._multiplier:], dim=1)


class Network(nn.Module):

  #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, output_node=2):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._output_node=output_node

    C_curr = stem_multiplier*C	#16*3=48
    self.stem_1 = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    self.stem_0 = nn.Sequential(
      nn.Conv2d(3, C, 3, padding=1, bias=False),
      nn.BatchNorm2d(C)
    )
 
    #C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #48,48,16
    C_prev_prev, C_prev, C_curr = C, C_curr, C #48,48,16
    self.cells = nn.ModuleList()
    reduction_prev = False
    #layers默认为8
    for i in range(layers):
      #[2,5]
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      reduction_prev = reduction
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      self.cells += [cell]
      #C_prev_prev, C_prev = C_prev, multiplier*C_curr #48,16*4=64
      #print(f"C_prev_prev={C_prev_prev};C_prev={C_prev};C_curr={C_curr}")
      C_prev_prev, C_prev = C_curr, multiplier*C_curr #48,16*4=64

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev+C_curr, num_classes)

    self._initialize_alphas()

  def inputSoftmax(self,temp,reduction):
    n = 2
    start = 0
    weights_list=[]
    for i in range(self._steps):
      end = start + n
      if reduction:
          weights_list.append(F.softmax(self.alphas_reduce[start:end].view(-1)/temp,dim=0).view(n,-1))
      else:
          weights_list.append(F.softmax(self.alphas_normal[start:end].view(-1)/temp,dim=0).view(n,-1))
      start = end
      n += 1
    return torch.cat(weights_list)

  def alphas_reduce_entropy(self,temp=1):
    n = 2
    start = 0
    weights_list=[]
    IES=0
    for i in range(self._steps):
      end = start + n
      IES+=entropy(F.softmax(self.alphas_normal[start:end].view(-1)/temp,dim=0).data.cpu().numpy(),axis=0)
      start = end
      n += 1
    return IES
    

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, inputd,temp=1):
    s0 = self.stem_0(inputd)
    s1 = self.stem_1(inputd)
    for i, cell in enumerate(self.cells):
      if cell.reduction: outpath_weights=torch.stack([F.softmax(self.alphas_reduce_output[0]/temp,dim=-1),F.sigmoid(self.alphas_reduce_output[1])])
      else: outpath_weights=torch.stack([F.softmax(self.alphas_normal_output[0]/temp,dim=-1),F.sigmoid(self.alphas_normal_output[1])])
      weights = self.inputSoftmax(temp,cell.reduction)
      #s0, s1 = s1, cell(s0, s1, weights,outpath_weights)
      s0, s1 = cell(s0, s1, weights,outpath_weights)
    s1=torch.cat([s0,s1], dim=1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, inputd, target,temp=1):
    #forward(inputd)
    logits = self(inputd,temp)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
	#step==4,k==14,k=2+3+4+5,对应4个node
    num_ops = len(PRIMITIVES)

    #标准正态初始化arch参数
    self.alphas_normal = Variable(0.1*torch.randn(k, num_ops), requires_grad=True)
    self.alphas_reduce = Variable(0.1*torch.randn(k, num_ops), requires_grad=True)
    self.alphas_normal_output = Variable(0.1*torch.randn(self._output_node,self._steps), requires_grad=True)
    self.alphas_reduce_output = Variable(0.1*torch.randn(self._output_node,self._steps), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.alphas_normal_output,
      self.alphas_reduce_output,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self,temp=1,threshold=0.4):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        #W_o = np.around(weights[start:end].copy(),decimals=4)
        W = weights[start:end].ravel()
        W_sorted_index=np.argsort(-W)
        W_sorted=np.sort(W)[::-1]
        W_cumsum=W_sorted.cumsum()
        quantile_index=2
        #quantile_index=np.argmax(W_cumsum>threshold)+1
        for a in range(quantile_index):
            op_index=W_sorted_index[a]%len(PRIMITIVES)
            pre_node_index=W_sorted_index[a]//len(PRIMITIVES)
            gene.append((PRIMITIVES[op_index],pre_node_index,i))
        start = end
        n += 1
      return gene

    with torch.no_grad():
        gene_normal = _parse(self.inputSoftmax(temp,reduction=False).data.cpu().numpy())
        gene_reduce = _parse(self.inputSoftmax(temp,reduction=True).data.cpu().numpy())
        gene_normal_output = torch.stack([F.softmax(self.alphas_normal_output[0]/temp,dim=-1),F.sigmoid(self.alphas_normal_output[1])])
        gene_reduce_output = torch.stack([F.softmax(self.alphas_reduce_output[0]/temp,dim=-1),F.sigmoid(self.alphas_reduce_output[1])])

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    #output_concat=range(self._steps)
    genotype = Genotype_op(
      normal=gene_normal,
      normal_output=gene_normal_output,
      reduce=gene_reduce,
      reduce_output=gene_reduce_output,
    )
    return genotype

