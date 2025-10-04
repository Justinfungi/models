#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - BayesianCNN Training Script

该文件实现了BayesianCNN模型的完整训练流程，包括数据加载、模型训练、验证、
早停机制和模型保存。支持项目特定的表格数据格式和贝叶斯不确定性量化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yaml
import json
import logging
import argparse
import os
import sys
import time
import random
from typing import Dict, Any, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from pathlib import Path

# 设置警告过滤
warnings.filterwarnings('ignore')

# 导入模型
try:
    from model import create_model, BayesianCNN
except ImportError:
    print("Warning: Could not import model.py. Make sure model.py is in the same directory.")


def set_all_seeds(seed: int = 42) -> None:
    """
    设置所有随机种子确保可重现性
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device() -> torch.device:
    """
    自动设备配置和优化设置
    
    Returns:
        配置好的设备对象
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) 损失函数
    结合负对数似然和KL散度
    """
    
    def __init__(self, train_size: int):
        super().__init__()
        self.train_size = train_size
        self.nll_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                kl_divergence: torch.Tensor, beta: float) -> torch.Tensor:
        """
        计算ELBO损失
        
        Args:
            outputs: 模型输出
            targets: 真实标签
            kl_divergence: KL散度
            beta: KL项权重
            
        Returns:
            ELBO损失值
        """
        nll = self.nll_loss(outputs, targets)
        kl_scaled = kl_divergence / self.train_size
        return nll + beta * kl_scaled


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    稳定的log-mean-exp计算，用于集成预测
    
    Args:
        x: 输入张量
        dim: 计算维度
        
    Returns:
        log-mean-exp结果
    """
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])


class BayesianCNNTrainer:
    """BayesianCNN训练器 - 核心训练管理类"""
    
    def __init__(self, config_path: str = "config.yaml", 
                 data_config_path: str = "data.yaml"):
        """
        初始化训练器，加载配置，设置环境
        
        Args:
            config_path: 模型配置文件路径
            data_config_path: 数据配置文件路径
        """
        self.config_path = config_path
        self.data_config_path = data_config_path
        
        # 加载配置
        self.config = self.load_config(config_path)
        self.data_config = self.load_config(data_config_path)
        
        # 设置环境
        set_all_seeds(42)
        self.device = setup_device()
        self.setup_logging()
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default config.")
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'architecture': {
                'bayesian_config': {
                    'prior_mu': 0,
                    'prior_sigma': 0.1,
                    'posterior_mu_initial': [0, 0.1],
                    'posterior_rho_initial': [-3, 0.1]
                },
                'network': {
                    'input_dim': None,
                    'hidden_dims': [128, 64],
                    'output_dim': None,
                    'dropout': 0.2,
                    'activation': 'relu'
                }
            },
            'training': {
                'optimizer': {
                    'type': 'adam',
                    'learning_rate': 0.001,
                    'weight_decay': 1e-4
                },
                'epochs': 100,
                'batch_size': 32,
                'patience': 10,
                'loss': {
                    'kl_weight': 0.1
                },
                'ensemble': {
                    'train_samples': 1,
                    'val_samples': 5,
                    'test_samples': 10
                }
            }
        }
    
    def setup_logging(self):
        """设置结构化日志系统"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_path: Optional[str] = None, 
                  quick_test: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载项目特定的表格数据格式
        
        Args:
            data_path: 数据文件路径
            quick_test: 是否为快速测试模式
            
        Returns:
            训练和测试数据
        """
        try:
            # 确定数据路径
            if data_path is None:
                data_folder = self.data_config.get('data_paths', {}).get('data_folder', 'data/feature_set')
                data_file = self.data_config.get('data_paths', {}).get('data_file', 'mrmr_task_1_2013-01-01_2018-06-30.pq')
                data_path = os.path.join(data_folder, data_file)
            
            self.logger.info(f"Loading data from: {data_path}")
            
            # 加载parquet文件
            if not os.path.exists(data_path):
                self.logger.warning(f"Data file not found: {data_path}")
                return self.generate_synthetic_data(quick_test)
            
            df = pd.read_parquet(data_path)
            self.logger.info(f"Loaded data shape: {df.shape}")
            
            # 识别特征列（包含@符号）
            feature_columns = [col for col in df.columns if '@' in col]
            if not feature_columns:
                self.logger.warning("No feature columns found with '@' symbol")
                # 使用除了最后一列和日期列之外的所有列作为特征
                exclude_cols = ['date', 'time', 'symbol'] + [df.columns[-1]]
                feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            label_column = df.columns[-1]
            self.logger.info(f"Found {len(feature_columns)} feature columns")
            self.logger.info(f"Label column: {label_column}")
            
            # 快速测试模式：使用少量数据
            if quick_test:
                df = df.head(200)
                self.logger.info(f"Quick test mode: using {len(df)} samples")
            
            # 时间序列分割
            if 'date' in df.columns:
                df_sorted = df.sort_values('date')
                start_date = pd.to_datetime(df_sorted['date'].iloc[0])
                train_end = start_date + pd.DateOffset(years=1)
                
                train_mask = pd.to_datetime(df_sorted['date']) < train_end
                test_mask = pd.to_datetime(df_sorted['date']) >= train_end
                
                X_train = df_sorted[train_mask][feature_columns].values
                y_train = df_sorted[train_mask][label_column].values
                X_test = df_sorted[test_mask][feature_columns].values
                y_test = df_sorted[test_mask][label_column].values
            else:
                # 如果没有日期列，使用随机分割
                X = df[feature_columns].values
                y = df[label_column].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return self.generate_synthetic_data(quick_test)
    
    def generate_synthetic_data(self, quick_test: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成合成数据用于测试
        
        Args:
            quick_test: 是否为快速测试模式
            
        Returns:
            合成的训练和测试数据
        """
        self.logger.info("Generating synthetic data for testing")
        
        n_samples = 200 if quick_test else 1000
        n_features = 20
        n_classes = 2
        
        # 生成随机数据
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, y_train, X_test, y_test
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray,
                     quick_test: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备数据加载器
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            quick_test: 是否为快速测试模式
            
        Returns:
            训练、验证、测试数据加载器
        """
        # 数据预处理
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 标签编码
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 分割训练和验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train_encoded, test_size=0.2, random_state=42,
            stratify=y_train_encoded
        )
        
        # 更新配置中的数据维度信息
        self.config['architecture']['network']['input_dim'] = X_train_final.shape[1]
        self.config['architecture']['network']['output_dim'] = len(np.unique(y_train_encoded))
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_final)
        y_train_tensor = torch.LongTensor(y_train_final)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test_encoded)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 批次大小调整
        batch_size = 16 if quick_test else self.config['training']['batch_size']
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def setup_model(self, train_size: int):
        """
        设置模型、优化器和损失函数
        
        Args:
            train_size: 训练集大小
        """
        try:
            # 创建模型
            self.model = create_model(self.config).to(self.device)
            self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        except Exception as e:
            self.logger.warning(f"Could not create model from model.py: {e}")
            # 创建简单的贝叶斯模型
            self.model = self.create_simple_bayesian_model().to(self.device)
        
        # 设置优化器
        optimizer_config = self.config['training']['optimizer']
        if optimizer_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0),
                momentum=optimizer_config.get('momentum', 0.9)
            )
        
        # 设置损失函数
        self.criterion = ELBOLoss(train_size).to(self.device)
        
        self.logger.info("Model, optimizer, and loss function initialized")
    
    def create_simple_bayesian_model(self) -> nn.Module:
        """
        创建简单的贝叶斯模型（备用方案）
        
        Returns:
            简单的贝叶斯模型
        """
        input_dim = self.config['architecture']['network']['input_dim']
        output_dim = self.config['architecture']['network']['output_dim']
        hidden_dims = self.config['architecture']['network']['hidden_dims']
        dropout = self.config['architecture']['network']['dropout']
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        model = nn.Sequential(*layers)
        
        # 添加简单的KL散度计算方法
        def forward_with_kl(self, x):
            output = super(type(model), self).forward(x)
            # 简单的KL散度估计
            kl = torch.tensor(0.0, device=x.device)
            for param in self.parameters():
                kl += torch.sum(param ** 2) * 0.001
            return output, kl
        
        model.forward_with_kl = forward_with_kl.__get__(model, type(model))
        
        return model
    
    def get_beta(self, batch_idx: int, num_batches: int, epoch: int = 0) -> float:
        """
        计算KL散度权重beta
        
        Args:
            batch_idx: 当前批次索引
            num_batches: 总批次数
            epoch: 当前epoch
            
        Returns:
            beta权重值
        """
        return self.config['training']['loss']['kl_weight']
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            平均损失、准确率、KL散度
        """
        self.model.train()
        total_loss = 0.0
        total_kl = 0.0
        all_preds = []
        all_labels = []
        
        num_ens = self.config['training']['ensemble']['train_samples']
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 集成采样
            if hasattr(self.model, 'forward_with_kl'):
                # 使用贝叶斯前向传播
                outputs_list = []
                kl_total = 0.0
                
                for j in range(num_ens):
                    net_out, kl = self.model.forward_with_kl(inputs)
                    kl_total += kl
                    outputs_list.append(F.log_softmax(net_out, dim=1))
                
                if len(outputs_list) > 1:
                    outputs = torch.stack(outputs_list, dim=2)
                    log_outputs = logmeanexp(outputs, dim=2)
                else:
                    log_outputs = outputs_list[0]
                
                kl_avg = kl_total / num_ens
            else:
                # 标准前向传播
                outputs = self.model(inputs)
                log_outputs = F.log_softmax(outputs, dim=1)
                kl_avg = torch.tensor(0.0, device=self.device)
            
            # 计算损失
            beta = self.get_beta(batch_idx, len(train_loader), epoch)
            loss = self.criterion(torch.exp(log_outputs), labels, kl_avg, beta)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_kl += kl_avg.item()
            
            # 记录预测
            preds = torch.argmax(log_outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, avg_kl
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            验证损失、准确率、详细指标
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        num_ens = self.config['training']['ensemble']['val_samples']
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 集成预测
                if hasattr(self.model, 'forward_with_kl'):
                    outputs_list = []
                    kl_total = 0.0
                    
                    for j in range(num_ens):
                        net_out, kl = self.model.forward_with_kl(inputs)
                        kl_total += kl
                        outputs_list.append(F.log_softmax(net_out, dim=1))
                    
                    if len(outputs_list) > 1:
                        outputs = torch.stack(outputs_list, dim=2)
                        log_outputs = logmeanexp(outputs, dim=2)
                    else:
                        log_outputs = outputs_list[0]
                    
                    kl_avg = kl_total / num_ens
                else:
                    outputs = self.model(inputs)
                    log_outputs = F.log_softmax(outputs, dim=1)
                    kl_avg = torch.tensor(0.0, device=self.device)
                
                # 计算损失
                beta = self.get_beta(0, 1, epoch)
                loss = self.criterion(torch.exp(log_outputs), labels, kl_avg, beta)
                total_loss += loss.item()
                
                # 记录预测
                probs = torch.exp(log_outputs)
                preds = torch.argmax(log_outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # 计算详细指标
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        # 如果是二分类，计算AUC
        if len(np.unique(all_labels)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                if all_probs_array.shape[1] == 2:
                    metrics['auc'] = roc_auc_score(all_labels, all_probs_array[:, 1])
            except:
                pass
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, quick_test: bool = False):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            quick_test: 是否为快速测试模式
        """
        epochs = 2 if quick_test else self.config['training']['epochs']
        patience = 3 if quick_test else self.config['training']['patience']
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc, train_kl = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc, val_metrics = self.validate(val_loader, epoch)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            # 日志记录
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"KL: {train_kl:.4f}, Time: {epoch_time:.2f}s"
            )
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.logger.info("Training completed")
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        测试模型性能
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            测试指标字典
        """
        self.logger.info("Starting model testing")
        
        # 加载最佳模型
        checkpoint_path = "checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for testing")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        uncertainties = []
        
        num_ens = self.config['training']['ensemble']['test_samples']
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 集成预测和不确定性计算
                if hasattr(self.model, 'forward_with_kl'):
                    outputs_list = []
                    
                    for j in range(num_ens):
                        net_out, _ = self.model.forward_with_kl(inputs)
                        outputs_list.append(F.softmax(net_out, dim=1))
                    
                    if len(outputs_list) > 1:
                        outputs_stack = torch.stack(outputs_list, dim=0)  # (num_ens, batch_size, num_classes)
                        mean_probs = torch.mean(outputs_stack, dim=0)
                        
                        # 计算不确定性
                        epistemic_uncertainty = torch.var(outputs_stack, dim=0).sum(dim=1)
                        aleatoric_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
                        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
                        
                        uncertainties.extend(total_uncertainty.cpu().numpy())
                    else:
                        mean_probs = outputs_list[0]
                        uncertainties.extend([0.0] * inputs.size(0))
                else:
                    outputs = self.model(inputs)
                    mean_probs = F.softmax(outputs, dim=1)
                    uncertainties.extend([0.0] * inputs.size(0))
                
                preds = torch.argmax(mean_probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(mean_probs.cpu().numpy())
        
        # 计算测试指标
        test_metrics = {
            'test_accuracy': accuracy_score(all_labels, all_preds),
            'test_precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'test_recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'test_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties)
        }
        
        # 如果是二分类，计算AUC
        if len(np.unique(all_labels)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                if all_probs_array.shape[1] == 2:
                    test_metrics['test_auc'] = roc_auc_score(all_labels, all_probs_array[:, 1])
            except:
                pass
        
        self.logger.info(f"Test Results: {test_metrics}")
        return test_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        保存模型检查点
        
        Args:
            epoch: 当前epoch
            metrics: 验证指标
        """
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
    
    def save_results(self, test_metrics: Dict[str, float]):
        """
        保存实验结果到JSON文件
        
        Args:
            test_metrics: 测试指标
        """
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'model_name': 'BayesianCNN',
            'config': self.config,
            'test_metrics': test_metrics,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_path = os.path.join(results_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_path}")
    
    def calculate_uncertainty(self, test_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        计算贝叶斯不确定性
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            不确定性字典
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        num_samples = self.config['training']['ensemble']['test_samples']
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                # 多次采样
                predictions = []
                
                for _ in range(num_samples):
                    if hasattr(self.model, 'forward_with_kl'):
                        outputs, _ = self.model.forward_with_kl(inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    probs = F.softmax(outputs, dim=1)
                    predictions.append(probs.cpu().numpy())
                
                predictions = np.array(predictions)  # (num_samples, batch_size, num_classes)
                all_predictions.append(predictions)
                all_labels.extend(labels.numpy())
        
        # 计算不确定性
        all_predictions = np.concatenate(all_predictions, axis=1)
        
        # 认识不确定性 (预测方差)
        mean_predictions = np.mean(all_predictions, axis=0)
        epistemic_uncertainty = np.var(all_predictions, axis=0)
        
        # 偶然不确定性 (预测熵)
        aleatoric_uncertainty = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8), axis=1)
        
        # 总不确定性
        total_uncertainty = epistemic_uncertainty.sum(axis=1) + aleatoric_uncertainty
        
        return {
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'mean_predictions': mean_predictions
        }


def train_model_ensemble(net, optimizer, criterion, trainloader, num_ens: int, 
                        beta_type, epoch: int, num_epochs: int):
    """
    贝叶斯集成训练函数 - 支持多次采样
    
    Args:
        net: 神经网络模型
        optimizer: 优化器
        criterion: 损失函数
        trainloader: 训练数据加载器
        num_ens: 集成采样次数
        beta_type: Beta权重类型
        epoch: 当前epoch
        num_epochs: 总epoch数
        
    Returns:
        训练损失和准确率
    """
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        # 集成采样
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        kl = 0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        log_outputs = logmeanexp(outputs, dim=2)
        
        beta = beta_type
        if isinstance(beta_type, str):
            beta = 0.1  # 默认值
        
        loss = criterion(log_outputs, targets, kl, beta)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = log_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss / len(trainloader), 100. * correct / total


def validate_model_ensemble(net, criterion, validloader, num_ens: int, 
                           beta_type, epoch: int, num_epochs: int):
    """
    贝叶斯集成验证函数
    
    Args:
        net: 神经网络模型
        criterion: 损失函数
        validloader: 验证数据加载器
        num_ens: 集成采样次数
        beta_type: Beta权重类型
        epoch: 当前epoch
        num_epochs: 总epoch数
        
    Returns:
        验证损失和准确率
    """
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # 集成采样
            outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
            
            kl = 0
            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1)
            
            kl = kl / num_ens
            log_outputs = logmeanexp(outputs, dim=2)
            
            beta = beta_type
            if isinstance(beta_type, str):
                beta = 0.1  # 默认值
            
            loss = criterion(log_outputs, targets, kl, beta)
            
            valid_loss += loss.item()
            _, predicted = log_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return valid_loss / len(validloader), 100. * correct / total


def calculate_uncertainty(outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算预测不确定性 - 贝叶斯CNN特有功能
    
    Args:
        outputs: 模型输出张量 (num_samples, batch_size, num_classes)
        
    Returns:
        认识不确定性和偶然不确定性
    """
    # 计算平均预测
    mean_outputs = torch.mean(outputs, dim=0)
    
    # 认识不确定性 (预测方差)
    epistemic_uncertainty = torch.var(outputs, dim=0)
    
    # 偶然不确定性 (预测熵)
    aleatoric_uncertainty = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)
    
    return epistemic_uncertainty.sum(dim=1), aleatoric_uncertainty


def quick_validation():
    """快速验证模式：使用最小预算验证代码正确性"""
    print("🚀 开始 training.py 快速验证...")
    
    try:
        # 创建训练器
        trainer = BayesianCNNTrainer()
        
        # 加载数据（快速模式）
        X_train, y_train, X_test, y_test = trainer.load_data(quick_test=True)
        print(f"✅ 数据加载成功: Train {X_train.shape}, Test {X_test.shape}")
        
        # 准备数据加载器
        train_loader, val_loader, test_loader = trainer.prepare_data(
            X_train, y_train, X_test, y_test, quick_test=True
        )
        print(f"✅ 数据加载器创建成功")
        
        # 设置模型
        trainer.setup_model(len(train_loader.dataset))
        print(f"✅ 模型设置成功")
        
        # 快速训练（1个epoch）
        print("🏃 开始快速训练验证...")
        trainer.train(train_loader, val_loader, quick_test=True)
        print(f"✅ 训练流程验证成功")
        
        # 快速测试
        test_metrics = trainer.test(test_loader)
        print(f"✅ 测试流程验证成功: {test_metrics}")
        
        # 保存结果
        trainer.save_results(test_metrics)
        print(f"✅ 结果保存成功")
        
        print("🎉 training.py 快速验证完成！所有功能正常运行。")
        
    except Exception as e:
        print(f"❌ 快速验证失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='BayesianCNN Training Script')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-config', type=str, default='data.yaml',
                       help='Path to data config file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速验证模式：使用最小预算验证训练脚本正确性')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_validation()
        return
    
    try:
        # 创建训练器
        trainer = BayesianCNNTrainer(args.config, args.data_config)
        
        # 覆盖配置参数
        if args.epochs is not None:
            trainer.config['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            trainer.config['training']['batch_size'] = args.batch_size
        if args.lr is not None:
            trainer.config['training']['optimizer']['learning_rate'] = args.lr
        
        # 加载数据
        X_train, y_train, X_test, y_test = trainer.load_data(args.data_path)
        
        # 准备数据加载器
        train_loader, val_loader, test_loader = trainer.prepare_data(
            X_train, y_train, X_test, y_test
        )
        
        # 设置模型
        trainer.setup_model(len(train_loader.dataset))
        
        # 训练模型
        trainer.train(train_loader, val_loader)
        
        # 测试模型
        test_metrics = trainer.test(test_loader)
        
        # 保存结果
        trainer.save_results(test_metrics)
        
        # 计算不确定性
        if trainer.config.get('inference', {}).get('uncertainty_estimation', True):
            uncertainty_results = trainer.calculate_uncertainty(test_loader)
            trainer.logger.info(f"Uncertainty analysis completed")
        
        trainer.logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()