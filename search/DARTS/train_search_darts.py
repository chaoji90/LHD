import os
import sys
import time
import glob
import numpy as np
import torch
sys.path.insert(0, '../')
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
import pickle as pk

from torch.autograd import Variable
from concate_sum_five_sig import Network
from scipy.stats import entropy

torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=176, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
#parser.add_argument('--learning_rate_min', type=float, default=0.05, help='min learning rate')
parser.add_argument('--learning_rate_min', type=float, default=3e-4, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.0003, help='learning rate for arch encoding')
#parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--alphas_suffix',type=str,default='-darts')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  args.seed = random.randint(1, 100000)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  input_size, input_channels, n_classes, train_data = utils.get_data(args.dataset, args.data, args, validation=False)
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, n_classes, args.layers, criterion,5,5)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  w_optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  p_optimizer = torch.optim.Adam(model.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader( train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), pin_memory=True)

  valid_queue = torch.utils.data.DataLoader( train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), pin_memory=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  temp=1
  C=0

  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, p_optimizer, criterion, w_optimizer, lr, scheduler)
    logging.info('train_acc %f', train_acc)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    with torch.no_grad():
        alphas_normal=model.inputSoftmax(temp, reduction=False).cpu().data
        logging.info(alphas_normal)
        alphas_reduce=model.inputSoftmax(temp, reduction=True).cpu().data
        logging.info(alphas_reduce)
        normal_output_weights=torch.stack([F.softmax(model.alphas_normal_output[0]/temp,dim=-1),F.sigmoid(model.alphas_normal_output[1])]).cpu().data
        logging.info(normal_output_weights)
        reduce_output_weights=torch.stack([F.softmax(model.alphas_reduce_output[0]/temp,dim=-1),F.sigmoid(model.alphas_reduce_output[1])]).cpu().data
        logging.info(reduce_output_weights)
        entropy_normal=model.alphas_reduce_entropy(temp)
        entropy_normal_output=entropy(normal_output_weights[0],axis=-1)
        entropy_reduce_output=entropy(reduce_output_weights[0],axis=-1)
        logging.info('C=%f, softmax_temp=%f, entropy_normal=%f, entropy_normal_output=%f, entropy_reduce_output=%f',C,temp,entropy_normal,entropy_normal_output,entropy_reduce_output)
        if epoch==args.epochs-1:
            pk.dump([alphas_normal.numpy(),alphas_reduce.numpy(),normal_output_weights.numpy(),reduce_output_weights.numpy()],open(time.strftime("%Y%m%d-%H%M%S")+args.alphas_suffix+f'-{args.dataset}'+'.pl','wb'))
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

    scheduler.step()
    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, p_optimizer, criterion, w_optimizer, lr, scheduler):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (inputd, target) in enumerate(train_queue):
    model.train()
    n = inputd.size(0)

    inputd = inputd.cuda()
    target = target.cuda(non_blocking=True)

    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    p_optimizer.zero_grad()
    logits = model(input_search)
    loss = criterion(logits, target_search)
    loss.backward()
    p_optimizer.step()

    w_optimizer.zero_grad()
    logits = model(inputd)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    w_optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (inputd, target) in enumerate(valid_queue):
    inputd = inputd.cuda()
    target = target.cuda(non_blocking=True)

    logits = model(inputd)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = inputd.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

