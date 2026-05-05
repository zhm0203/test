import os
import pdb
import sys

import math
import time
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
# from transformers.optimization import get_scheduler
from tqdm.autonotebook import trange
from loguru import logger

import constants
from evaluator import predict, get_eval_metric_fn, EarlyStopping
from trainer_utils import SupervisedTrainCollator, TrainDataset
from trainer_utils import get_parameter_names
from trainer_utils import get_scheduler
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self,
                 model,
                 train_set_list,
                 test_set_list=None,
                 collate_fn=None,
                 output_dir='./ckpt_zero',
                 num_epoch=10,
                 batch_size=64,
                 lr=1e-4,
                 weight_decay=0,
                 patience=5,
                 eval_batch_size=256,
                 warmup_ratio=None,
                 warmup_steps=None,
                 balance_sample=False,
                 load_best_at_last=True,
                 ignore_duplicate_cols=False,
                 eval_metric='acc',
                 eval_less_is_better=False,
                 num_workers=0,
                 **kwargs,
                 ):
        """
        args:
        train_set_list: a list of training sets [(x_1,y_1),(x_2,y_2),...]
        test_set_list: a list of tuples of test set (x, y), same as train_set_list. if set None, do not do evaluation
        and early stopping
        patience: the max number of early stop patience
        num_workers: how many workers used to process dataloader. recommend to be 0 if training data smaller than 10000.
        eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
        """
        self.model = model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)
        print("Model is on device:", next(model.parameters()).device)
        

        if isinstance(train_set_list, tuple):
            train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple):
            test_set_list = [test_set_list]

        self.train_set_list = train_set_list
        self.test_set_list = test_set_list
        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )
        self.trainloader_list = [
            self._build_dataloader(trainset, batch_size, collator=self.collate_fn, num_workers=num_workers) for trainset
            in train_set_list
        ]
        if test_set_list is not None:
            self.testloader_list = [
                self._build_dataloader(testset, eval_batch_size, collator=self.collate_fn, num_workers=num_workers,
                                       shuffle=False) for testset in test_set_list
            ]
        else:
            self.testloader_list = None

        self.test_set_list = test_set_list
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False,
                                            less_is_better=eval_less_is_better)
        self.args = {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_epoch': num_epoch,
            'eval_batch_size': eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
        }
        self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch * len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def train(self):
        print("Model is on device:", next(self.model.parameters()).device)
        args = self.args
        self.create_optimizer()   # 创建优化器
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:  # 预热
            num_train_steps = args['num_training_steps']
            logger.info(f'set warmup training in initial {num_train_steps} steps')
            self.create_scheduler(num_train_steps, self.optimizer)

        train_losses = []
        val_losses = []
        
        start_time = time.time()
        for epoch in trange(args['num_epoch'], desc='Epoch'):   # 进度条
            ite = 0
            train_loss_all = 0
            for dataindex in range(len(self.trainloader_list)):
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()   # 前向传播更新前清除之前的梯度
                    # print("data[0]:",data[0]['x_num'].shape)
                    # print(data[1])
                    logits, loss = self.model(data[0], data[1])   # 0是数据 1是标签

                    loss.backward()   # 反向传播
                    self.optimizer.step()   # 更新参数
                    train_loss_all += loss.item()
                    # train_losses.append(loss.item())
                    ite += 1                # 累计迭代
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()    # 调整学习率
                    # break
            train_loss_all /= ite
            train_losses.append(train_loss_all)
            if self.test_set_list is not None:
                eval_res_list, val_loss = self.evaluate()  
                eval_res = np.mean(eval_res_list)
                print('epoch: {}, test {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    # break
            val_losses.append(val_loss)
            print('val_losses:', val_losses)
            print('epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all,
                                                                                         self.optimizer.param_groups[0][
                                                                                             'lr'],
                                                                                         time.time() - start_time))
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        print('figrue complete')
        plt.savefig('loss_curve.png')  # 可选：保存图片

        if os.path.exists(self.output_dir):
            if self.test_set_list is not None:
                # load checkpoints
                logger.info(f'load best at last from {self.output_dir}')
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

        logger.info('training complete, cost {:.1f} secs.'.format(time.time() - start_time))

    def evaluate(self):
        # evaluate in each epoch
        self.model.eval()     # 评估模式，会关闭 dropout 和 batch normalization 等训练时特有的行为
        eval_res_list = []    # 返回评估结果
        val_loss = 0
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in self.testloader_list[dataindex]:
                if data[1] is not None:
                    label = data[1]
                    # print(label)
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1])
                # print(logits.shape)
                # print(torch.softmax(logits, -1).shape)
                if loss is not None:
                    loss_list.append(loss.item())
                
                if logits is not None:
                    if logits.shape[-1] == 1:  # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                        # print('logits:',logits)
                        
                        # probs = torch.sigmoid(logits)
                        # preds = (probs > 0.5).long()
                        print('logits.shape1:',logits.shape[-1])

                    else:  # multi-class classification
                        pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())

            if len(pred_list) > 0:
                pred_all = np.concatenate(pred_list, 0)
                print('pred_all.shape2:',pred_all.shape)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()
                    print('pred_all.shape3:',pred_all.shape)

            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = np.concatenate(y_test, 0)
                print('y_test.shape:',y_test.shape)
                print('pred_all.shape4:',pred_all.shape)
                eval_res = self.args['eval_metric'](y_test, pred_all)    

            eval_res_list.append(eval_res)
        val_loss = np.mean(loss_list) if len(loss_list) > 0 else None

        return eval_res_list, val_loss

    def train_no_dataloader(self,
                            resume_from_checkpoint=None,
                            ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            print('set warmup training.')
            self.create_scheduler(args['num_training_steps'], self.optimizer)
        
        train_losses = []
        val_losses = []
        for epoch in range(args['num_epoch']):
            ite = 0
            # go through all train sets
            for train_set in self.train_set_list:
                x_train, y_train = train_set
                train_loss_all = 0
                for i in range(0, len(x_train), args['batch_size']):
                    self.model.train()
                    if self.balance_sample:
                        bs_x_train_pos = x_train.loc[y_train == 1].sample(int(args['batch_size'] / 2))
                        bs_y_train_pos = y_train.loc[bs_x_train_pos.index]
                        bs_x_train_neg = x_train.loc[y_train == 0].sample(int(args['batch_size'] / 2))
                        bs_y_train_neg = y_train.loc[bs_x_train_neg.index]
                        bs_x_train = pd.concat([bs_x_train_pos, bs_x_train_neg], axis=0)
                        bs_y_train = pd.concat([bs_y_train_pos, bs_y_train_neg], axis=0)
                    else:
                        bs_x_train = x_train.iloc[i:i + args['batch_size']]
                        bs_y_train = y_train.loc[bs_x_train.index]

                    self.optimizer.zero_grad()
                    logits, loss = self.model(bs_x_train, bs_y_train)
                    loss.backward()

                    self.optimizer.step()
                    train_loss_all += loss.item()
                    train_losses.append(loss.item())
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set is not None:
                # evaluate in each epoch
                self.model.eval()
                x_test, y_test = self.test_set
                val_loss, pred_all = predict(self.model, x_test, self.args['eval_batch_size'])
                eval_res = self.args['eval_metric'](y_test, pred_all)
                print('epoch: {}, test {}: {}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break
            val_losses.append(val_loss)

            print('epoch: {}, train loss: {}, lr: {:.6f}'.format(epoch, train_loss_all,
                                                                 self.optimizer.param_groups[0]['lr']))
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        print('figrue complete')
        plt.savefig('loss_curve.png')  # 可选：保存图片
        # plt.show()

        
        
        if os.path.exists(self.output_dir):
            if self.test_set is not None:
                # load checkpoints
                print('load best at last from', self.output_dir)
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

    def save_model(self, output_dir=None):
        if output_dir is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        logger.info(f'saving model checkpoint to {output_dir}')
        self.model.save(output_dir)
        self.collate_fn.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))
        if self.args is not None:
            train_args = {}
            for k, v in self.args.items():
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float):
                    train_args[k] = v
            with open(os.path.join(output_dir, constants.TRAINING_ARGS_NAME), 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_args))

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer=optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset           # Dataframe
            total_step += np.ceil(len(x_train) / batch_size)

        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(
                num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,    # 数据将加载到 CUDA 内存
            drop_last=False,
        )
        return trainloader
