#!/usr/bin/env python
# -*- coding: utf-8 -*-
from script import parameter
from script.tools import *
from script.model import PresRecST
from script.dataloader import process_dataset
from script.training_and_testing import training_and_testing
import torch
import numpy as np
import sys
import time


def main():
    # params
    params = parameter.Para(
        lr=1e-4, rec=7e-3, drop=0.0, batch_size=20, epoch=20, dev_ratio=0.0, test_ratio=0.2, embedding_dim=64,
        alpha1=1.0, alpha2=1.0, alpha3=1.0
    )
    out_name = 'PresRecST'
    sys.stdout = Logger(f'result/{out_name}_log.txt')

    print('/----- PresRecST start -----/')
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    # seed related
    seed = 2022
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    # data & settings
    print('-- Data Processing --')
    x_train, x_test, train_loader, test_loader = process_dataset(params.dev_ratio, params.test_ratio, params.batch_size)
    print('-- Parameter Setting --')
    print("lr:", params.lr, "rec:", params.rec, "dropout:", params.drop, "batch_size:",
          params.batch_size, "epoch:", params.epoch, "dev_ratio:", params.dev_ratio, "test_ratio:", params.test_ratio)
    model = PresRecST(params.batch_size, params.embedding_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.rec)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)

    # training and testing
    print('-- Training and Testing --')
    training_and_testing(model, x_train, x_test, train_loader, test_loader, params, optimizer, criterion, scheduler)

    print('-- PresRecST Finished! --')


if __name__ == '__main__':
    main()

