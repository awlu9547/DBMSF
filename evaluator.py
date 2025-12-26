import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import Extra_Dataset
from utils.metrics import plot_roc_curve, save_fpr_tpr_auc
from models.main_model import DBMSF
import numpy as np
import random
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)


def load_config(config_path=r".\your config path"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DBMSF")
    args = parser.parse_args()
    config_keys = ['batch_size', 'lr', 'freeze_ratio', 'cmb',
                   'epochs', 'iters_to_val', 'save_best', 'UNI_path',
                   'class_names']
    for key in config_keys:
        setattr(args, key, load_config()[key])
    return args


def print_configs(_args):
    print('============== Configs ==============')
    for key, value in vars(_args).items():
        print(f'{key:<10}  \t {value}', end='\n')
    print('=====================================')


def test_collate_fn(batch):
    slide, tile, patch_nums, label, feature_matrix, cooadj_matrix = zip(*batch)
    return (
        list(slide),
        torch.stack(tile),  # [batch_size, 3, 224, 224]
        torch.stack(patch_nums),  # [batch_size]
        torch.stack(label),  # [batch_size]
        list(feature_matrix),  # [batch_size, 1536]
        list(cooadj_matrix)
    )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_auc(model, loader, class_names, save_path, save_as_pdf=False):
    tic = time.time()
    num_classes = len(class_names)
    device = next(model.parameters()).device

    slide_list = []
    label_list = []
    prob_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Test', total=len(loader)):
            slide, tile, patch_num, label, feature_list, cooadj_list = batch

            tile = tile.to(device)
            patch_num = patch_num.to(device)
            label = label.to(device)
            feature_list = [fea.to(device) for fea in feature_list]
            cooadj_list = [adj.to(device) for adj in cooadj_list]

            output, _, _ = model(tile, patch_num, feature_list, cooadj_list)  # [B, C]

            probs = torch.softmax(output, dim=1)

            slide_list.extend(slide)
            label_list.append(label.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    label_all = np.concatenate(label_list, axis=0)  # (N,)
    prob_all = np.concatenate(prob_list, axis=0)  # (N, C)
    N = label_all.shape[0]
    assert len(slide_list) == N

    # DataFrame
    df = pd.DataFrame({
        'slide': slide_list,
        'label': label_all
    })
    for i in range(num_classes):
        df[f'prob_{i}'] = prob_all[:, i]

    prob_cols = [f'prob_{i}' for i in range(num_classes)]
    df_slide = df.groupby('slide')[prob_cols].mean()  # MeanPoolingMIL
    df_slide['label_pred'] = df_slide[prob_cols].idxmax(axis=1).str.split('_').str[-1].astype(int)

    # merge true label
    true_df = df[['slide', 'label']].drop_duplicates('slide').set_index('slide')
    df_slide = df_slide.join(true_df, how='left').reset_index()

    label_true = df_slide['label'].astype(int)
    label_pred = df_slide['label_pred'].astype(int)

    # metrics
    acc = accuracy_score(label_true, label_pred)
    f1 = f1_score(label_true, label_pred, average='macro')
    precision = precision_score(label_true, label_pred, average='macro')
    recall = recall_score(label_true, label_pred, average='macro')
    # 混淆矩阵中的数量和为val中所有slides和
    cm = confusion_matrix(label_true, label_pred, labels=np.arange(num_classes))

    # ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        bin_true = (label_true == i).astype(int)

        # fpr 与 tpr 分别表示假正例率和真正例率
        fpr[i], tpr[i], _ = roc_curve(bin_true, df_slide[f'prob_{i}'])
        roc_auc[i] = auc(fpr[i], tpr[i])

    ext = 'pdf' if save_as_pdf else 'png'
    fpr_tpr_auc_path = os.path.join(save_path, f'test_fpr_tpr_auc.json')
    save_fpr_tpr_auc(fpr, tpr, roc_auc, fpr_tpr_auc_path)

    roc_path = os.path.join(save_path, f'slide_test.{ext}')
    slide_auc = plot_roc_curve(
        fpr, tpr, roc_auc,
        num_classes, class_names,
        title='Slide-level ROC',
        save_path=roc_path
    )

    print(f'Validation time: {time.time() - tic:.2f}s')
    return {
        'slide_AUC': slide_auc,
        'slide_ACC': acc,
        'slide_F1': f1,
        'slide_Precision': precision,
        'slide_Recall': recall,
        'confusion_matrix': cm
    }


if __name__ == '__main__':
    args = parse_args()
    data_path = r'.\your data path'
    set_seed(42)

    combination = args.cmb

    print_configs(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    class_names = args.class_names
    num_classes = len(class_names)

    # create test_dataset objects
    test_dataset = Extra_Dataset(data_path)
    if test_dataset is not None:
        print("load test_dataset successfully")

    # create test_loaders
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12,
                             collate_fn=test_collate_fn)
    if test_loader is not None:
        print("load test_loader successfully")

    model = MSUNI(n_classes=num_classes, freeze_ratio=args.freeze_ratio, cmb=args.cmb, ckpt_path=args.UNI_path)
    ckpt_path = r''
    checkpoint = torch.load(ckpt_path, weights_only=False)
    print("load checkpoint successfully")
    print(checkpoint)

    model.load_state_dict(checkpoint['model'])
    model.to(device)

    result_path = r".\your result path"
    store_path = os.path.join(result_path, f'Test_CHOL')
    os.makedirs(store_path, exist_ok=True)

    result = test_auc(model=model, loader=test_loader, class_names=class_names, save_path=store_path)
    slide_AUC = result['slide_AUC']
    slide_ACC = result['slide_ACC']
    slide_F1 = result['slide_F1']
    slide_Precision = result['slide_Precision']
    slide_Recall = result['slide_Recall']
    confusion_matrix = result['confusion_matrix']

    # Update log file
    log_data = {'slide_auc': slide_AUC, 'slide_accuracy': slide_ACC,
                'slide_F1': slide_F1, 'slide_precision': slide_Precision,
                'slide_Recall': slide_Recall, 'confusion_matrix': confusion_matrix}

    log_file_path = os.path.join(store_path, 'log_test_metrics.csv')

    if not os.path.exists(log_file_path):
        pd.DataFrame(columns=['slide_auc', 'slide_accuracy', 'slide_F1', 'slide_precision',
                              'slide_Recall', 'confusion_mat', ]).to_csv(log_file_path, index=False)
    pd.DataFrame([log_data]).to_csv(log_file_path, mode='a', header=False, index=False)
