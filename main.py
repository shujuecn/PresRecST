# -*- coding: utf-8 -*-

import torch
import numpy as np
import sys
import time
import os

from script.tools import Para, Logger
from script.model import PresRecST4Lung, PresRecST4PD
from script.dataloader import process_dataset
from script.training_and_testing import train_and_test


def main():
    """
    通过 dataset_type 参数来控制使用 "lung" 还是 "tcmpd" 数据集。
    """
    # ============ 1. 根据需求选择数据集类型 ============
    # 您可以通过命令行参数、配置文件或手动来指定
    dataset_type = "lung"  # 可选值："lung" 或 "tcmpd"

    # ============ 2. 统一参数设置 (适用于两种数据集) ============
    params = Para(
        lr=1e-4,
        rec=7e-3,
        drop=0.0,
        batch_size=32,
        epoch=100,
        dev_ratio=0.0,
        test_ratio=0.2,
        embedding_dim=64,
        alpha1=1.0,
        alpha2=1.0,
        alpha3=1.0,
    )

    # ============ 3. 不同数据集的差异化处理 ============

    if dataset_type == "lung":
        # ---- Lung 数据集 ----
        out_name = "PresRecST_Lung"
        add_name = "Lung"
        is_lung4 = True
        is_tcmpd = False  # lung 数据集时，这里为 False

    elif dataset_type == "tcmpd":
        # ---- TCMPD 数据集 ----
        out_name = "PresRecST_TCMPD"
        add_name = "TCMPD"
        is_lung4 = False
        is_tcmpd = True

    else:
        raise ValueError("Unknown dataset_type, must be 'lung' or 'tcmpd'")

    # ============ 4. 日志文件 ============
    os.makedirs("log", exist_ok=True)
    # 日志文件名根据 out_name + add_name
    sys.stdout = Logger(f"log/{out_name}_{add_name}_log.txt")

    print(f"/---- {out_name} start -----/")
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    # ============ 5. 设置随机种子，保证可复现性 ============
    seed = 2022
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    # ============ 6. 数据加载与处理 ============
    print("-- Data Processing --")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if is_lung4:
        # Lung 数据集处理
        x_train, x_test, train_loader, test_loader, data_len = process_dataset(
            params=params, dataset_type="lung", device=device, seed=seed
        )
    else:
        # TCMPD 数据集处理
        x_train, x_test, train_loader, test_loader, data_len = process_dataset(
            params=params, dataset_type="tcmpd", device=device, seed=seed
        )

    # ============ 7. 输出参数信息 ============
    print("-- Parameter Setting --")
    print(
        f"lr: {params.lr} | rec: {params.rec} | dropout: {params.drop} |"
        f" batch_size: {params.batch_size} | epoch: {params.epoch} |"
        f" dev_ratio: {params.dev_ratio} | test_ratio: {params.test_ratio}"
    )

    # ============ 8. 根据不同数据集选择不同的模型 ============
    if is_lung4:
        model = PresRecST4Lung(params.batch_size, params.embedding_dim, data_len)
    else:
        model = PresRecST4PD(params.batch_size, params.embedding_dim, data_len)

    model = model.to(device)

    # ============ 9. 设置损失函数、优化器、调度器 ============
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.rec
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)

    # ============ 10. 训练与测试 ============
    print("-- Training and Testing --")
    test_result = train_and_test(
        model=model,
        x_train=x_train,
        x_test=x_test,
        train_loader=train_loader,
        test_loader=test_loader,
        params=params,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        is_lung4=is_lung4,
        is_tcmpd=is_tcmpd,
    )

    # ============ 11. 保存结果 ============
    os.makedirs("result", exist_ok=True)
    test_result.to_excel(f"result/herb_result_{add_name}.xlsx", index=False)

    # ============ 12. 打印结束标记 ============
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print(f"-- {out_name} Finished! --")


if __name__ == "__main__":
    main()
