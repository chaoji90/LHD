#python -u train.py --work-dir ./experiments/densenet40_k12_svhn --resume ./experiments/densenet40_k12_svhn/checkpoints/last_checkpoint.pth
import os
import datetime
import warnings
from collections import OrderedDict
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
#from torch.utils.tensorboard import SummaryWriter

import models
from data import SVHNDataset
from utils import AverageMeter, Evaluator
from configs import Config
import utils_data as utils
from torchstat import stat
import gen_trans
import pickle as pk
from multi_path_concat_sum import NetworkCIFAR as Network

warnings.filterwarnings('ignore')


class Trainer:

    def __init__(self, cfgs, model):
        """
        Args:
            cfgs: (class) 类对象，其属性为训练的各种超参数设置
            model: (torch.nn.Module) 待训练的模型实例
        """
        self.cfgs = cfgs
        self.model = model
        self.start_epoch = 1
        self.best_err = 1.1

        self.device = torch.device(cfgs.gpu if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 取消某些层0的权重衰减
        named_params = list(self.model.named_parameters())
        grouped_params = [
            {'params': [param
                        for name, param in named_params if not any(no_decay in name for no_decay in cfgs.no_decays)],
             'weight_decay': cfgs.weight_decay},
            {'params': [param
                        for name, param in named_params if any(no_decay in name for no_decay in cfgs.no_decays)],
             'weight_decay': 0.0}
        ]

        self.criterion = cfgs.CriterionClass().to(self.device)
        self.optimizer = cfgs.OptimizerClass(grouped_params, **cfgs.optimizer_params)
        self.scheduler = cfgs.SchedulerClass(self.optimizer, **cfgs.scheduler_params)

        # optionally resume from a checkpoint
        '''
        if cfgs.resume:
            if os.path.isfile(cfgs.resume):
                print("=> loading checkpoint '{}'".format(cfgs.resume))
                checkpoint = torch.load(cfgs.resume, map_location=self.device)
                self.load(cfgs.resume)
                print("=> loaded checkpoint '{}' (epoch {})".format(cfgs.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(cfgs.resume))
        '''

        # checkpoint 存储的目录
        if not os.path.exists(cfgs.save_dir):
            os.makedirs(cfgs.save_dir)
        # 训练损失和（或）指标记录和tensorboard 日志文件所在的的目录
        if not os.path.exists(cfgs.log_dir):
            os.makedirs(cfgs.log_dir)
        #self.writer = SummaryWriter(log_dir=cfgs.log_dir)

        self.log('Trainer prepared in device: {}'.format(self.device))

    def fit(self, train_loader, valid_loader):
        """
        训练及验证模型
        Args:
            train_loader: (torch.utils.data.DataLoader) 训练数据加载器
            valid_loader: (torch.utils.data.DataLoader) 验证数据加载器
        """
        for epoch in range(self.start_epoch, self.cfgs.epochs + 1):
            if self.cfgs.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.datetime.now().isoformat()
                self.log('{}\tEpoch: {}\tLR: {}'.format(timestamp, epoch, lr))

            # 训练一个 epoch
            err = self.train(epoch, train_loader)

            self.save(epoch, f'{self.cfgs.save_dir}/last_checkpoint.pth')
            self.log(f'[RESULT]: Train Epoch: {epoch}\t Error Rate: {err:6.4f}')
            #self.writer.add_scalars('error', {'train': err}, epoch)

            # 验证
            err = self.validate(epoch, valid_loader)

            if err < self.best_err:
                self.best_err = err
                self.save(epoch, f'{self.cfgs.save_dir}/best_checkpoint_{str(epoch).zfill(3)}epoch.pth')
            self.log(f'[RESULT]: Valid Epoch: {epoch}\t Error Rate: {err:6.4f}')
            #self.writer.add_scalars('error', {'valid': err}, epoch)

            self.scheduler.step()

    def train(self, epoch, train_loader):
        """
        使用训练集训练一个epoch
        Args:
            epoch: (int) 第几代训练
            train_loader: (torch.utils.data.DataLoader) 训练数据加载器
        Returns:
            err: (float) error rate
        """
        losses = AverageMeter()
        evaluator = Evaluator(self.cfgs.num_classes)

        self.model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description('Train Epoch {}'.format(epoch))

            for step, (input_, target) in enumerate(train_loader):
                # move data to device
                input_ = torch.tensor(input_, device=self.device, dtype=torch.float32)
                target = torch.tensor(target, device=self.device, dtype=torch.long)

                # forward and compute loss
                output = self.model(input_)[0]
                loss = self.criterion(output, target)

                # backward and update params
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record loss and show it in the pbar
                losses.update(loss.item(), input_.size(0))
                postfix = OrderedDict({'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}'})
                pbar.set_postfix(ordered_dict=postfix)
                pbar.update()

                # visualization with TensorBoard
                total_iter = (epoch - 1) * len(train_loader) + step + 1
                #self.writer.add_scalar('training_loss', losses.val, total_iter)

                # update confusion matrix
                true = target.cpu().numpy()
                pred = output.max(dim=1)[1].cpu().numpy()
                evaluator.update_matrix(true, pred)

        return evaluator.error()

    def validate(self, epoch, valid_loader):
        """
        使用验证集测试模型的效果
        Args:
            epoch: (int) 第几代训练
            valid_loader: (torch.utils.data.DataLoader) 验证数据加载器
        Returns:
            err: (float) error rate
        """
        losses = AverageMeter()
        evaluator = Evaluator(self.cfgs.num_classes)

        self.model.eval()
        with tqdm(valid_loader) as pbar:
            pbar.set_description('Valid Epoch {}'.format(epoch))

            for i, (input_, target) in enumerate(valid_loader):
                # move data to GPU
                input_ = torch.tensor(input_, device=self.device, dtype=torch.float32)
                target = torch.tensor(target, device=self.device, dtype=torch.long)

                with torch.no_grad():
                    # compute output and loss
                    output = self.model(input_)[0]
                    loss = self.criterion(output, target)

                # record loss and show it in the pbar
                losses.update(loss.item(), input_.size(0))
                postfix = OrderedDict({'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}'})
                pbar.set_postfix(ordered_dict=postfix)
                pbar.update()

                # update confusion matrix
                true = target.cpu().numpy()
                pred = output.max(dim=1)[1].cpu().numpy()
                evaluator.update_matrix(true, pred)

        return evaluator.error()

    def save(self, epoch, path):
        self.model.eval()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_err': self.best_err,
            'epoch': epoch
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_err = checkpoint['best_err']
        self.start_epoch = checkpoint['epoch'] + 1

    def log(self, msg):
        if self.cfgs.verbose:
            print(msg)

        log_path = os.path.join(self.cfgs.log_dir, 'log.txt')
        with open(log_path, 'a+') as logger:
            logger.write(f'{msg}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--operations_cumsum_threshold', type=float, default='2',help='location of the data corpus')
    parser.add_argument('--sum_output_cumsum_threshold', type=float, default='0.9',help='location of the data corpus')
    parser.add_argument('--steps', type=int, default='5',help='location of the data corpus')
    parser.add_argument('--sum_path_wise_threshold', type=float, default='0.2',help='location of the data corpus')
    parser.add_argument('--normal_concat_path_wise_threshold', type=float, default='0',help='location of the data corpus')
    parser.add_argument('--reduce_concat_path_wise_threshold', type=float, default='0',help='location of the data corpus')
    parser.add_argument('--model_capacity', type=float, default='0',help='location of the data corpus')
    parser.add_argument('--arch_file', type=str, help='location of the data corpus')
    args = parser.parse_args()
    # ========== get model ==========
    #model = models.__dict__[Config.arch]()
    #print(utils.count_parameters_in_MB(model))
    #stat(model, (3, 32, 32))
    #exit(0)

    # ========== get data ==========
    #train_set = SVHNDataset(Config.train_root, ToTensor())
    #valid_set = SVHNDataset(Config.valid_root, ToTensor())
    input_size, input_channels, n_classes, train_set, valid_set = utils.get_data('svhn', './datapath', validation=True)
    alphaList=pk.load(open(args.arch_file,'rb'))
    genotype=gen_trans.genotype_unbind(alphaList,args)
    model = Network(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
    #stat(model, (3, 32, 32))
    #exit(0)

    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=Config.workers)
    valid_loader = DataLoader(valid_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.workers)

    # ========== train ==========
    trainer = Trainer(Config, model)
    trainer.fit(train_loader, valid_loader)
