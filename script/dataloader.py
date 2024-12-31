# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from script.tools import PreDatasetPTM2, PreDatasetLung4
from torch.utils.data import DataLoader


def parse_label_indices(label_str: str, max_index: int) -> List[int]:
    """
    将逗号或其他分隔的字符串解析为整数列表，只保留 [0, max_index) 范围内的有效索引。
    若字符串为空或含非法字符则返回空列表。
    """
    if not isinstance(label_str, str):
        return []

    results = []
    for x in label_str.split(","):
        x = x.strip()
        # 这里也可扩展为判断负数或其他非法字符
        if x.isdigit():
            idx = int(x)
            if 0 <= idx < max_index:
                results.append(idx)
    return results


def build_one_hot_matrix(
    list_of_index_lists: List[List[int]], num_cols: int
) -> np.ndarray:
    """
    根据每行的索引列表，把 [row, col] 位置置为 1，其余为 0，返回 shape=[n_rows, num_cols] 的 numpy 数组。

    :param list_of_index_lists:
        例如 [[1, 2], [3], [0, 5, 7], ...]，表示每行有哪些标签索引
    :param num_cols:
        One-Hot 向量的长度
    :return:
        二维 numpy 数组，dtype=float32
    """
    n_rows = len(list_of_index_lists)
    arr = np.zeros((n_rows, num_cols), dtype=np.float32)

    row_idx = []
    col_idx = []
    for i, indices in enumerate(list_of_index_lists):
        row_idx.extend([i] * len(indices))
        col_idx.extend(indices)

    arr[row_idx, col_idx] = 1.0
    return arr


def process_dataset(
    params, dataset_type: str, device: torch.device, seed: int = 2022
) -> Tuple[
    List[int],
    List[int],
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    List[int],
]:
    """
    通用数据加载函数：根据 dataset_type ('tcmpd' or 'lung')
    读取并处理对应数据集，返回 (x_train, x_test, train_loader, test_loader, data_len)。

    :param params:       参数对象(要求至少包含: batch_size, dev_ratio, test_ratio 等)
    :param dataset_type: 数据集类型，可为 'tcmpd' 或 'lung'
    :param device:       'cpu' 或 'cuda' 设备
    :param seed:         随机种子
    :return:            (x_train, x_test, train_loader, test_loader, data_len)
                        data_len:
                          - tcmpd => [sym_dim, herb_dim]
                          - lung  => [sym_len, syn_len, treat_len, herb_len]
    """
    np.random.seed(seed)

    if dataset_type.lower() == "tcmpd":
        # ========================= 处理 TCMPD 数据集 =========================
        data_file = "data/prescript_1195.csv"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File {data_file} not found.")

        df = pd.read_csv(data_file)
        data_amount = len(df)

        # 定义两列维度
        sym_dim = 390
        herb_dim = 805

        # 1) 解析 symptom 索引
        df["sym_indices"] = df.iloc[:, 0].apply(
            lambda s: parse_label_indices(s, sym_dim)
        )

        # 2) 解析 herb 索引，但先用 0..1194 做过滤
        #    后续需将 idx>=390 的 herb 再平移到 [0..805)
        df["herb_indices_raw"] = df.iloc[:, 1].apply(
            lambda s: parse_label_indices(s, 1195)
        )

        # 将 >=390 的部分映射到 0..804
        df["herb_indices"] = df["herb_indices_raw"].apply(
            lambda idx_list: [idx - 390 for idx in idx_list if 390 <= idx < 1195]
        )

        # 3) 构建 One-Hot
        sym_array = build_one_hot_matrix(df["sym_indices"].tolist(), sym_dim)
        herb_array = build_one_hot_matrix(df["herb_indices"].tolist(), herb_dim)

        # 4) Train/Test 切分
        idx_list = list(range(data_amount))
        dev_ratio = params.dev_ratio
        test_ratio = params.test_ratio
        x_train, x_test = train_test_split(
            idx_list,
            test_size=(dev_ratio + test_ratio),
            shuffle=False,
            random_state=seed,
        )
        print(f"[TCMPD] train_size: {len(x_train)}, test_size: {len(x_test)}")

        # 5) 转为 Tensor
        sym_tensor = torch.tensor(sym_array, device=device)
        herb_tensor = torch.tensor(herb_array, device=device)

        # 6) 构造 Dataset & DataLoader
        train_dataset = PreDatasetPTM2(sym_tensor[x_train], herb_tensor[x_train])
        test_dataset = PreDatasetPTM2(sym_tensor[x_test], herb_tensor[x_test])

        train_loader = DataLoader(
            train_dataset, batch_size=params.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=params.batch_size, shuffle=False
        )

        data_len = [sym_dim, herb_dim]

    elif dataset_type.lower() == "lung":
        # ========================= 处理 LUNG 数据集 =========================
        data_file = "data/TCM_Lung.xlsx"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File {data_file} not found.")

        df = pd.read_excel(data_file)
        data_amount = len(df)
        id_list = list(range(data_amount))

        # 只用 test_ratio (无需 dev_ratio)，保持与原逻辑一致
        test_ratio = params.test_ratio
        x_train, x_test = train_test_split(
            id_list, test_size=test_ratio, shuffle=False, random_state=seed
        )
        print(f"[Lung] train_size: {len(x_train)}, test_size: {len(x_test)}")

        # 读取四个字典表，获取各自的维度
        sym_ids = pd.read_excel(data_file, sheet_name="Symptom Dictionary")
        syn_ids = pd.read_excel(data_file, sheet_name="Syndrome Dictionary")
        treat_ids = pd.read_excel(data_file, sheet_name="Treat Dictionary")
        herb_ids = pd.read_excel(data_file, sheet_name="Herb Dictionary")

        sym_len = len(sym_ids)
        syn_len = len(syn_ids)
        treat_len = len(treat_ids)
        herb_len = len(herb_ids)

        # 这里假设 df 的第 0 列是 symptom 索引字符串,
        # 第 1 列是 syndrome, 第 2 列是 treat, 第 3 列是 herb
        # 对每列做 apply(parse_label_indices) 并限制范围
        df["sym_indices"] = df.iloc[:, 0].apply(
            lambda s: parse_label_indices(s, sym_len)
        )
        df["syn_indices"] = df.iloc[:, 1].apply(
            lambda s: parse_label_indices(s, syn_len)
        )
        df["treat_indices"] = df.iloc[:, 2].apply(
            lambda s: parse_label_indices(s, treat_len)
        )
        df["herb_indices"] = df.iloc[:, 3].apply(
            lambda s: parse_label_indices(s, herb_len)
        )

        # 分别构建 One-Hot
        sym_array = build_one_hot_matrix(df["sym_indices"].tolist(), sym_len)
        syn_array = build_one_hot_matrix(df["syn_indices"].tolist(), syn_len)
        treat_array = build_one_hot_matrix(df["treat_indices"].tolist(), treat_len)
        herb_array = build_one_hot_matrix(df["herb_indices"].tolist(), herb_len)

        # 转 tensor 并构建 Dataset/DataLoader
        sym_tensor = torch.tensor(sym_array, device=device)
        syn_tensor = torch.tensor(syn_array, device=device)
        treat_tensor = torch.tensor(treat_array, device=device)
        herb_tensor = torch.tensor(herb_array, device=device)

        train_dataset = PreDatasetLung4(
            sym_tensor[x_train],
            syn_tensor[x_train],
            treat_tensor[x_train],
            herb_tensor[x_train],
        )
        test_dataset = PreDatasetLung4(
            sym_tensor[x_test],
            syn_tensor[x_test],
            treat_tensor[x_test],
            herb_tensor[x_test],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=params.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=params.batch_size, shuffle=False
        )

        data_len = [sym_len, syn_len, treat_len, herb_len]

    else:
        raise ValueError("Unknown dataset_type. Must be 'tcmpd' or 'lung'.")

    return x_train, x_test, train_loader, test_loader, data_len
