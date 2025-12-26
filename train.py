import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PathologyDatasetKFold
from utils.metrics import val_auc
from models.main_model import MSUNI
import numpy as np
import random


def load_config(config_path=r".\your config path"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DBMSF")
    parser.add_argument('--fold', type=int, default=4, help='Fold for cross-validation')
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


def train_collate_fn(batch):
    tile, patch_nums, label, feature_matrix, cooadj_matrix = zip(*batch)
    return (
        torch.stack(tile),  # [batch_size, 3, 224, 224]
        torch.stack(patch_nums),
        torch.stack(label),  # [batch_size]
        list(feature_matrix),  # [batch_size, 1536]
        list(cooadj_matrix)
    )


def val_collate_fn(batch):
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


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n


if __name__ == '__main__':
    args = parse_args()

    set_seed(42)

    combination = args.cmb

    print_configs(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    class_names = args.class_names
    num_classes = len(class_names)

    # kf dir check
    assert os.path.exists(
        r'.\your dataset kf path'), 'k-fold dir not exists, run utils/gen_kfold_split.py first'
    print("check kf file successfully")

    # create dataset objects for train and val
    train_dataset = PathologyDatasetKFold(mode='train', combination=combination, fold=args.fold)
    val_dataset = PathologyDatasetKFold(mode='val', combination=combination, fold=args.fold)
    if train_dataset and val_dataset is not None:
        print("load dataset successfully")

    # create data loaders for train and val
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
                              collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12,
                            collate_fn=val_collate_fn)
    if train_loader and val_loader is not None:
        print("load loader successfully")

    model = MSUNI(n_classes=num_classes, freeze_ratio=args.freeze_ratio, cmb=args.cmb, ckpt_path=args.UNI_path)
    model.to(device)

    print_network(model)

    criterion_train = nn.CrossEntropyLoss().to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)

    # scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    result_path = r".\your result path"
    store_path = os.path.join(result_path, f'runs/{combination}_{args.freeze_ratio}/{args.fold}')
    os.makedirs(store_path, exist_ok=True)

    best_slide_auc = 0

    for epoch in range(args.epochs):
        epoch_avg_loss = 0

        print(f'Current epoch : {epoch + 1}, LR : {optimizer.param_groups[0]["lr"]:.8f}')

        for iter_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()

            tile, patch_num, label, feature_list, cooadj_list = batch
            tile = tile.to(device)
            patch_num = patch_num.to(device)
            label = label.to(device)  # [batch_size, num_classes]
            feature_list = [fea.to(device) for fea in feature_list]
            cooadj_list = [adj.to(device) for adj in cooadj_list]

            output = model(tile, patch_num, feature_list, cooadj_list)

            loss = criterion_train(output, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if ((args.iters_to_val != -1 and iter_idx % args.iters_to_val == 0 and iter_idx != 0)
                    or (args.iters_to_val == -1 and iter_idx == len(train_loader) - 1)):
                """
                option 1 :  每隔 iters_to_val 进行一个验证
                option 2 ： 指定 iters_to_val=-1，则默认在epoch的最后进行验证
                """
                model.eval()
                print(f"\nEpoch {epoch + 1} Iter {iter_idx}\n")

                result = val_auc(model=model, loader=val_loader, criterion=criterion_val,
                                 class_names=class_names, save_path=store_path, epoch=epoch,
                                 iter_idx=iter_idx)
                slide_AUC = result['slide_AUC']
                slide_ACC = result['slide_ACC']
                slide_F1 = result['slide_F1']
                slide_Precision = result['slide_Precision']
                slide_Recall = result['slide_Recall']
                confusion_matrix = result['confusion_matrix']

                # Update log file
                log_data = {'epoch': epoch + 1, 'iter_idx': iter_idx, 'slide_auc': slide_AUC,
                            'slide_accuracy': slide_ACC,
                            'slide_F1': slide_F1, 'slide_precision': slide_Precision,
                            'slide_Recall': slide_Recall, 'confusion_matrix': confusion_matrix}

                log_file_path = os.path.join(store_path, 'log_val_metrics.csv')

                if not os.path.exists(log_file_path):
                    pd.DataFrame(
                        columns=['epoch', 'iter_idx', 'slide_auc', 'slide_accuracy', 'slide_F1', 'slide_precision',
                                 'slide_Recall', 'confusion_mat', ]).to_csv(log_file_path, index=False)
                pd.DataFrame([log_data]).to_csv(log_file_path, mode='a', header=False, index=False)

                model_path = f'{args.fold}_best.pth'

                if args.save_best is True and slide_AUC > best_slide_auc:
                    best_slide_auc = slide_AUC
                    torch.save(model, os.path.join(store_path, model_path))
                    print(f'-----Saving best model to {model_path}, current best auc: {best_slide_auc:.4f}-----')

        scheduler.step()
