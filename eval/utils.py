import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset
import numpy as np
import preproc

class_nums={'cifar10':10,'svhn':10,'cifar100':100,'mit67':67,'sport8':8,'flowers102':102,'caltech256':257,'places365':365}
LARGE_DATASETS = ["imagenet", "mit67", "sport8", "flowers102",'caltech256','places365']

#def class_nums(dataset):
    #return class_nums[args.dataset]

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def get_data(dataset, data_root, args, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()
    
    data_path = data_root
    n_classes = class_nums[dataset]
    if dataset == 'places365': dset_cls = dset.Places365
    elif dataset == 'caltech256': dset_cls = dset.Caltech256
    elif dataset == 'cifar10': dset_cls = dset.CIFAR10
    elif dataset == 'cifar100': dset_cls = dset.CIFAR100
    elif dataset == 'svhn': dset_cls = dset.SVHN
    elif dataset == 'mnist': dset_cls = dset.MNIST
    elif dataset == 'fashionmnist': dset_cls = dset.FashionMNIST
    #New Datasets
    elif dataset == 'mit67':
        dset_cls = dset.ImageFolder
        data_path = '%s/MIT67/train' % data_root  # 'data/MIT67/train'
        val_path = '%s/MIT67/test' % data_root  # 'data/MIT67/val'
    elif dataset == 'sport8':
        dset_cls = dset.ImageFolder
        data_path = '%s/Sport8/train' % data_root  # 'data/Sport8/train'
        val_path = '%s/Sport8/test' % data_root # 'data/Sport8/val'
    elif dataset == 'flowers102':
        dset_cls = dset.ImageFolder
        data_path = '%s/flowers102/train' % data_root  # 'data/flowers102/train'
        val_path = '%s/flowers102/test' % data_root  # 'data/flowers102/val' 
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, args)
    if dataset in LARGE_DATASETS:
        if dataset == 'places365': trn_data = dset_cls(root=data_path, download=True, transform=trn_transform)
        else:
            trn_data = dset_cls(root=data_path, transform=trn_transform)
        shape = trn_data[0][0].unsqueeze(0).shape
        print(shape)
        assert shape[2] == shape[3], "not expected shape = {}".format(shape)
        input_size = shape[2]
    else:
        if dataset == 'svhn': trn_data = dset_cls(root=data_path, split='train', download=True, transform=trn_transform)
        else: trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
        # assuming shape is NHW or NHWC
        try:
            shape = trn_data.data.shape
        except AttributeError:
            shape = trn_data.train_data.shape
        if dataset == 'svhn':
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            input_size = shape[2]
        else:
            assert shape[1] == shape[2], "not expected shape = {}".format(shape)
            input_size = shape[1]
        
    input_channels = 3 if len(shape) == 4 else 1
    # print("Number of input channels: ", input_channels)

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        if dataset in LARGE_DATASETS:
            ret.append(dset_cls(root=val_path, transform=val_transform))
        else:
            if dataset == 'places365': val_data = dset_cls(root=data_path, split='val', download=True, transform=trn_transform)
            elif dataset == 'svhn': val_data = dset_cls(root=data_path, split='test', download=True, transform=trn_transform)
            else: val_data = dset_cls(root=data_path, train=False, download=True, transform=trn_transform)
        ret.append(val_data)

    return ret

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

def get_dataset(args):
  if args.dataset=='cifar10':
      train_transform, valid_transform = _data_transforms_cifar10(args)
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'cifar100':
      train_transform, valid_transform = _data_transforms_cifar100(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'svhn':
      train_transform, valid_transform = _data_transforms_svhn(args)
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
      valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
  return train_data,valid_data

def _data_transforms_svhn(args):
  SVHN_MEAN = [0.4377, 0.4438, 0.4728]
  SVHN_STD = [0.1980, 0.2010, 0.1970]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length,
                                             args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  return train_transform, valid_transform


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
  CIFAR_STD = [0.2673, 0.2564, 0.2762]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length,
                                             args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  return train_transform, valid_transform


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def process_step_vector(x, method, mask, tau=None):
  if method == 'softmax':
    output = F.softmax(x, dim=-1)
  elif method == 'dirichlet':
    output = torch.distributions.dirichlet.Dirichlet(F.elu(x) + 1).rsample()
  elif method == 'gumbel':
    output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)
  
  if mask is None:
    return output
  else:
    output_pruned = torch.zeros_like(output)
    output_pruned[mask] = output[mask]
    output_pruned /= output_pruned.sum()
    assert (output_pruned[~mask] == 0.0).all()
    return output_pruned
    

def process_step_matrix(x, method, mask, tau=None):
  weights = []
  if mask is None:
    for line in x:
      weights.append(process_step_vector(line, method, None, tau))
  else:
    for i, line in enumerate(x):
      weights.append(process_step_vector(line, method, mask[i], tau))
  return torch.stack(weights)


def prune(x, num_keep, mask, reset=False):
  if not mask is None:
    x.data[~mask] -= 1000000
  src, index = x.topk(k=num_keep, dim=-1)
  if not reset:
    x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=src))
  else:
    x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=1e-3*torch.randn_like(src)))
  mask = torch.zeros_like(x, dtype=torch.bool).scatter(
      dim=1, index=index, src=torch.ones_like(src,dtype=torch.bool))
  return mask

