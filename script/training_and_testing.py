# -*- coding: utf-8 -*-

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from typing import List, Tuple, Any


def compute_topk_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    topk_p_count: List[float],
    topk_r_count: List[float]
) -> None:
    """
    计算 batch 内每个样本在 k=1~20 时的 Precision / Recall 累加值，待所有 batch 完成后再做平均。

    :param preds:   [batch_size, num_classes] 的预测张量
    :param targets: [batch_size, num_classes] 的二值标签张量
    :param topk_p_count: 用于累加每个 k 下的 precision 值
    :param topk_r_count: 用于累加每个 k 下的 recall 值
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
        true_indices = (targets[i] == 1).nonzero(as_tuple=True)[0].tolist()
        if not true_indices:  # 若没有任何正样本则跳过
            continue
        true_indices_set = set(true_indices)

        # 一次性 topk(20)，再拆分到 k=1..20，减少重复计算
        _, top_indices = torch.topk(preds[i], 20)
        top_indices_list = top_indices.tolist()

        for k in range(1, 21):
            selected = top_indices_list[:k]
            intersection_count = len(true_indices_set.intersection(selected))
            topk_p_count[k - 1] += intersection_count / k
            topk_r_count[k - 1] += intersection_count / len(true_indices)


def evaluate_model(
    model: Module,
    data_loader: DataLoader,
    params: Any,
    criterion: Module,
    is_lung4: bool = False,
    is_tcmpd: bool = False
) -> Tuple[float, List[float], List[float]]:
    """
    在 data_loader 上评估模型，并返回平均损失、累加的 topk precision/recall。
    多任务 (lung4) 时会计算 (pred_sd, pred_treat, pred_herb) 的多任务损失；
    单任务 (tcmpd) 时只需要 herb 的损失和预测。

    :param model:       PyTorch 模型
    :param data_loader: 测试或验证集 DataLoader
    :param params:      参数对象 (包含 alpha1, alpha2, alpha3 等)
    :param criterion:   损失函数
    :param is_lung4:    是否使用 lung4 (多任务) 数据集
    :param is_tcmpd:    是否使用 tcmpd (单任务) 数据集
    :return:            (test_loss, topk_p_count, topk_r_count)
    """
    model.eval()
    test_loss = 0.0
    topk_p_count = [0.0] * 20
    topk_r_count = [0.0] * 20

    alpha1 = params.alpha1
    alpha2 = getattr(params, "alpha2", 0.0)
    alpha3 = getattr(params, "alpha3", 0.0)

    with torch.no_grad():
        for batch in data_loader:
            if is_lung4 and not is_tcmpd:
                # batch = (sid, sdid, tid, hid)
                sid, sdid, tid, hid = batch
                sid, sdid, tid, hid = sid.float(), sdid.float(), tid.float(), hid.float()

                preds_sd, preds_treat, preds_herb = model(sid)
                loss = (
                    alpha1 * criterion(preds_herb, hid)
                    + alpha2 * criterion(preds_sd, sdid)
                    + alpha3 * criterion(preds_treat, tid)
                )
                test_loss += loss.item()
                # Top-k指标只针对 herb
                compute_topk_metrics(preds_herb, hid, topk_p_count, topk_r_count)

            elif is_tcmpd and not is_lung4:
                # batch = (sid, hid)
                sid, hid = batch
                sid, hid = sid.float(), hid.float()

                preds_herb = model(sid)
                loss = alpha1 * criterion(preds_herb, hid)
                test_loss += loss.item()

                compute_topk_metrics(preds_herb, hid, topk_p_count, topk_r_count)

            else:
                raise ValueError("Unknown dataset type in evaluate_model")

    test_loss /= len(data_loader)
    return test_loss, topk_p_count, topk_r_count


def calculate_precision_recall_f1(
    topk_p_count: List[float],
    topk_r_count: List[float],
    data_size: int
) -> pd.DataFrame:
    """
    根据累加好的 precision / recall 统计值，计算最终 P/R/F1，并返回 DataFrame。

    :param topk_p_count: 每个 k 的 precision 累加值
    :param topk_r_count: 每个 k 的 recall 累加值
    :param data_size:    数据集的样本数量，用于做平均
    :return:             包含 K, Precision, Recall, F1-score 的 DataFrame
    """
    p_list, r_list, f1_list = [], [], []
    for k in range(20):
        precision_k = topk_p_count[k] / data_size
        recall_k = topk_r_count[k] / data_size
        p_list.append(precision_k)
        r_list.append(recall_k)
        if precision_k + recall_k == 0:
            f1_list.append(0.0)
        else:
            f1_list.append((2 * precision_k * recall_k) / (precision_k + recall_k))

    df = pd.DataFrame({
        "K": [i + 1 for i in range(20)],
        "Precision": p_list,
        "Recall": r_list,
        "F1-score": f1_list
    })
    return df


def train_and_test(
    model: Module,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    params: Any,
    optimizer: Optimizer,
    criterion: Module,
    scheduler: Any,
    is_lung4: bool = False,
    is_tcmpd: bool = False
) -> pd.DataFrame:
    """
    同一个函数同时支持 lung4 (多任务) 与 tcmpd (单任务)。通过 is_lung4 与 is_tcmpd 控制。

    :param model:        PyTorch 模型
    :param x_train:      训练集张量（仅用于获取长度或尺寸）
    :param x_test:       测试集张量（仅用于获取长度或尺寸）
    :param train_loader: 训练集 DataLoader
    :param test_loader:  测试集 DataLoader
    :param params:       参数对象，需包含:
                            - epoch: 训练总轮数
                            - alpha1, alpha2, alpha3: 损失权重 (若是 lung4)
    :param optimizer:    优化器
    :param criterion:    损失函数
    :param scheduler:    学习率调度器
    :param is_lung4:     是否使用 lung4（多任务）数据集
    :param is_tcmpd:     是否使用 tcmpd（单任务）数据集
    :return:             返回最终测试指标 DataFrame
    """
    train_loss_list = []
    test_loss_list = []

    alpha1 = params.alpha1
    alpha2 = getattr(params, "alpha2", 0.0)
    alpha3 = getattr(params, "alpha3", 0.0)

    # ========= 训练阶段 (Train) =========
    for epoch in range(params.epoch):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            if is_lung4 and not is_tcmpd:
                # lung4: batch = (sid, sdid, tid, hid)
                sid, sdid, tid, hid = batch
                sid, sdid, tid, hid = sid.float(), sdid.float(), tid.float(), hid.float()

                preds_sd, preds_treat, preds_herb = model(sid)
                loss = (
                    alpha1 * criterion(preds_herb, hid)
                    + alpha2 * criterion(preds_sd, sdid)
                    + alpha3 * criterion(preds_treat, tid)
                )

            elif is_tcmpd and not is_lung4:
                # tcmpd: batch = (sid, hid) => 单任务
                sid, hid = batch
                sid, hid = sid.float(), hid.float()

                preds_herb = model(sid)
                loss = alpha1 * criterion(preds_herb, hid)

            else:
                # 若出现同时为 True 或同时为 False 的情况，则视为配置错误
                raise ValueError("Unknown dataset type in train_and_test")

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        print(f"[Epoch {epoch + 1}] train_loss: {avg_train_loss:.4f}")

        # ========= 测试阶段 (Test) =========
        # 可以按需求决定每多少个 epoch 测一次；此处示例为每 10 epoch 测一次
        if (epoch + 1) % 10 == 0:
            test_loss, topk_p_count, topk_r_count = evaluate_model(
                model=model,
                data_loader=test_loader,
                params=params,
                criterion=criterion,
                is_lung4=is_lung4,
                is_tcmpd=is_tcmpd
            )
            test_loss_list.append(test_loss)
            print(f"Epoch {epoch + 1} - test_loss: {test_loss:.4f}")

            # 简要打印 k=5, 10, 20 的指标
            data_size = len(x_test)

            p5 = topk_p_count[4] / data_size
            p10 = topk_p_count[9] / data_size
            p20 = topk_p_count[19] / data_size
            r5 = topk_r_count[4] / data_size
            r10 = topk_r_count[9] / data_size
            r20 = topk_r_count[19] / data_size

            def f1_score(p, r):
                return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

            print(f"Precision@5-10-20: {p5:.4f}, {p10:.4f}, {p20:.4f}")
            print(f"Recall@5-10-20:    {r5:.4f}, {r10:.4f}, {r20:.4f}")
            print(f"F1@5-10-20:        "
                  f"{f1_score(p5, r5):.4f}, "
                  f"{f1_score(p10, r10):.4f}, "
                  f"{f1_score(p20, r20):.4f}")

        # 学习率调度
        scheduler.step()

    # ========= 训练完成，做一次最终测试评估 =========
    print("-- Final Test Evaluation --", "(lung4)" if is_lung4 else "(tcmpd)")
    final_loss, final_p_count, final_r_count = evaluate_model(
        model=model,
        data_loader=test_loader,
        params=params,
        criterion=criterion,
        is_lung4=is_lung4,
        is_tcmpd=is_tcmpd
    )
    print(f"Final test_loss: {final_loss:.4f}")

    # 计算并打印最终的指标
    test_df = calculate_precision_recall_f1(final_p_count, final_r_count, len(x_test))
    print(test_df)

    return test_df
