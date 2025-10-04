#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - Mamba模型训练脚本
基于选择性状态空间模型的高效序列处理架构，适配二元分类任务的完整训练流程
"""

import os
import sys
import argparse
import logging
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# 导入模型
try:
    from model import create_model, MambaConfig
except ImportError:
    print("警告: 无法导入model.py，将使用简化模型")
    create_model = None
    MambaConfig = None

warnings.filterwarnings('ignore')

class MambaTrainer:
    """Mamba模型训练器 - 核心训练管理类"""
    
    def __init__(self, config_path: str = "config.yaml", data_config_path: str = "data.yaml", quick_test: bool = False):
        """
        初始化训练器，加载配置，设置环境
        
        Args:
            config_path: 模型配置文件路径
            data_config_path: 数据配置文件路径
            quick_test: 是否为快速测试模式
        """
        self.quick_test = quick_test
        
        # 硬编码环境设置 - 先设置日志系统
        self.setup_device()
        self.setup_logging()
        self.set_seed()
        
        # 然后加载配置
        self.config = self.load_config(config_path)
        self.data_config = self.load_config(data_config_path)
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        # 创建必要目录
        self.create_directories()
        
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
            self.logger.info(f"成功加载配置文件: {config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"配置文件未找到: {config_path}")
            # 返回默认配置
            if 'config.yaml' in config_path:
                return self.get_default_config()
            else:
                return self.get_default_data_config()
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
            
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认模型配置"""
        return {
            'architecture': {
                'd_model': 256,
                'd_state': 16,
                'd_conv': 4,
                'expand': 2,
                'n_layer': 8,
                'dropout': 0.1
            },
            'training': {
                'optimizer': 'adamw',
                'learning_rate': 0.0003,
                'weight_decay': 1e-5,
                'epochs': 100,
                'batch_size': 32,
                'patience': 10,
                'gradient_clip_value': 1.0,
                'scheduler': 'cosine_annealing_warm_restarts',
                'scheduler_params': {
                    'T_0': 10,
                    'T_mult': 2,
                    'eta_min': 1e-5
                }
            },
            'inference': {
                'batch_size': 64,
                'max_length': 512
            }
        }
        
    def get_default_data_config(self) -> Dict[str, Any]:
        """获取默认数据配置"""
        return {
            'data_format': {
                'input_type': 'tabular',
                'file_format': 'parquet',
                'feature_identifier': '@',
                'label_position': 'last'
            },
            'task': {
                'type': 'binary_classification',
                'num_features': None,
                'num_classes': None
            },
            'data_paths': {
                'data_folder': '/home/feng.hao.jie/deployment/model_explorer/b_model_reproduction_agent/data/feature_set',
                'data_phase': 1,
                'data_file': 'mrmr_task_1_2013-01-01_2018-06-30.pq'
            },
            'preprocessing': {
                'feature_selection': {
                    'method': 'automatic'
                },
                'scaling': {
                    'method': 'standard'
                },
                'split': {
                    'method': 'time_series',
                    'train_duration': '1_year',
                    'test_duration': 'remaining'
                }
            }
        }
    
    def setup_logging(self):
        """设置结构化日志系统（硬编码配置）"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("训练日志系统初始化完成")
        
        # 记录设备信息
        if torch.cuda.is_available():
            self.logger.info(f"使用GPU设备: {torch.cuda.get_device_name()}")
        else:
            self.logger.info("使用CPU设备")
        
        # 记录种子信息
        self.logger.info(f"设置随机种子: 42")
        
    def setup_device(self):
        """自动设备检测和配置（硬编码处理）"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 延迟日志记录到logger初始化后
        else:
            # 延迟日志记录到logger初始化后
            pass
            
    def set_seed(self, seed: int = 42):
        """设置随机种子（硬编码处理）"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 延迟日志记录到logger初始化后
        
    def create_directories(self):
        """创建必要的目录"""
        directories = ['checkpoints', 'results', 'logs']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
            
    def load_data(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载项目特定格式的Parquet数据
        
        Args:
            data_path: 数据文件路径，如果为None则使用配置文件中的路径
            
        Returns:
            特征数组和标签数组的元组
        """
        try:
            if data_path is None:
                data_folder = self.data_config['data_paths']['data_folder']
                data_file = self.data_config['data_paths']['data_file']
                data_path = os.path.join(data_folder, data_file)
            
            self.logger.info(f"加载数据文件: {data_path}")
            
            # 检查文件是否存在
            if not os.path.exists(data_path):
                # 尝试查找匹配的文件
                data_folder = os.path.dirname(data_path)
                pattern = os.path.basename(data_path)
                
                if '*' in pattern:
                    import glob
                    matching_files = glob.glob(data_path)
                    if matching_files:
                        data_path = matching_files[0]
                        self.logger.info(f"找到匹配文件: {data_path}")
                    else:
                        raise FileNotFoundError(f"未找到匹配的数据文件: {data_path}")
                else:
                    raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
            # 加载Parquet文件
            df = pd.read_parquet(data_path)
            self.logger.info(f"数据形状: {df.shape}")
            
            # 自动识别特征列（包含@符号的列）
            feature_identifier = self.data_config['data_format']['feature_identifier']
            feature_columns = [col for col in df.columns if feature_identifier in col]
            
            if not feature_columns:
                self.logger.warning("未找到包含@符号的特征列，使用除最后一列外的所有列作为特征")
                feature_columns = df.columns[:-1].tolist()
            
            # 获取标签列（最后一列）
            label_column = df.columns[-1]
            
            self.logger.info(f"特征列数量: {len(feature_columns)}")
            self.logger.info(f"标签列: {label_column}")
            
            # 提取特征和标签
            X = df[feature_columns].values.astype(np.float32)
            y = df[label_column].values.astype(np.int64)
            
            # 处理缺失值
            if np.isnan(X).any():
                self.logger.warning("检测到缺失值，使用均值填充")
                X = np.nan_to_num(X, nan=np.nanmean(X))
            
            # 更新配置中的特征维度和类别数量
            self.data_config['task']['num_features'] = X.shape[1]
            self.data_config['task']['num_classes'] = len(np.unique(y))
            
            self.logger.info(f"特征维度: {X.shape[1]}")
            self.logger.info(f"类别数量: {len(np.unique(y))}")
            self.logger.info(f"样本数量: {X.shape[0]}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
            
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备时间序列数据加载器，实现固定时间分割
        
        Args:
            X: 特征数组
            y: 标签数组
            
        Returns:
            训练、验证、测试数据加载器的元组
        """
        try:
            # 快速测试模式：使用极少数据
            if self.quick_test:
                n_samples = min(100, X.shape[0])
                indices = np.random.choice(X.shape[0], n_samples, replace=False)
                X = X[indices]
                y = y[indices]
                self.logger.info(f"快速测试模式：使用 {n_samples} 个样本")
            
            # 标准化处理
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 保存scaler
            scaler_path = 'checkpoints/scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Scaler已保存到: {scaler_path}")
            
            # 时间序列分割（简化版：按比例分割）
            n_samples = X_scaled.shape[0]
            
            if self.quick_test:
                # 快速测试模式：简单分割
                train_size = int(0.6 * n_samples)
                val_size = int(0.2 * n_samples)
                
                train_indices = np.arange(train_size)
                val_indices = np.arange(train_size, train_size + val_size)
                test_indices = np.arange(train_size + val_size, n_samples)
            else:
                # 正常模式：时间序列分割
                train_size = int(0.6 * n_samples)  # 60%用于训练
                val_size = int(0.2 * n_samples)    # 20%用于验证
                
                train_indices = np.arange(train_size)
                val_indices = np.arange(train_size, train_size + val_size)
                test_indices = np.arange(train_size + val_size, n_samples)
            
            # 创建数据集
            X_train, y_train = X_scaled[train_indices], y[train_indices]
            X_val, y_val = X_scaled[val_indices], y[val_indices]
            X_test, y_test = X_scaled[test_indices], y[test_indices]
            
            self.logger.info(f"训练集大小: {X_train.shape[0]}")
            self.logger.info(f"验证集大小: {X_val.shape[0]}")
            self.logger.info(f"测试集大小: {X_test.shape[0]}")
            
            # 转换为PyTorch张量
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test)
            )
            
            # 创建数据加载器
            batch_size = self.config['training']['batch_size']
            if self.quick_test:
                batch_size = min(16, batch_size)  # 快速测试使用更小的批次
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['inference']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['inference']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"数据准备失败: {e}")
            raise
            
    def setup_model(self):
        """设置Mamba模型、优化器和损失函数"""
        try:
            # 获取模型配置
            model_config = self.config['architecture'].copy()
            
            # 设置输入维度
            input_dim = self.data_config['task']['num_features']
            num_classes = self.data_config['task']['num_classes']
            
            # 创建模型
            if create_model is not None and MambaConfig is not None:
                # 使用真实的Mamba模型
                self.model = create_model(
                    config_path=None,  # 使用默认配置
                    num_features=input_dim,
                    num_classes=num_classes
                )
            else:
                # 使用简化的替代模型
                self.model = SimpleMambaModel(
                    input_dim=input_dim,
                    d_model=model_config['d_model'],
                    n_layer=model_config['n_layer'],
                    num_classes=num_classes,
                    dropout=model_config['dropout']
                )
            
            self.model = self.model.to(self.device)
            
            # 设置优化器
            optimizer_config = self.config['training']['optimizer']
            if isinstance(optimizer_config, dict):
                optimizer_name = optimizer_config.get('type', 'adamw').lower()
                lr = float(optimizer_config.get('lr', 0.0003))
                weight_decay = float(optimizer_config.get('weight_decay', 1e-5))
            else:
                optimizer_name = optimizer_config.lower()
                lr = float(self.config['training'].get('learning_rate', 0.0003))
                weight_decay = float(self.config['training'].get('weight_decay', 1e-5))
            
            if optimizer_name == 'adamw':
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer_name == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            else:
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=0.9
                )
            
            # 设置学习率调度器
            scheduler_name = self.config['training'].get('scheduler', 'cosine_annealing_warm_restarts')
            if scheduler_name == 'cosine_annealing_warm_restarts':
                scheduler_params = self.config['training'].get('scheduler_params', {})
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=scheduler_params.get('T_0', 10),
                    T_mult=scheduler_params.get('T_mult', 2),
                    eta_min=scheduler_params.get('eta_min', 1e-5)
                )
            else:
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=30,
                    gamma=0.1
                )
            
            # 设置损失函数
            self.criterion = nn.CrossEntropyLoss()
            
            # 计算模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"模型总参数数量: {total_params:,}")
            self.logger.info(f"可训练参数数量: {trainable_params:,}")
            
        except Exception as e:
            self.logger.error(f"模型设置失败: {e}")
            raise
            
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        单个epoch的训练过程
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均损失、AUC、准确率的元组
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            gradient_clipping = self.config['training'].get('gradient_clipping', {})
            if gradient_clipping.get('enabled', True):
                clip_value = gradient_clipping.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测结果
            probabilities = torch.softmax(output, dim=1)
            predictions = probabilities[:, 1].detach().cpu().numpy()  # 正类概率
            labels = target.detach().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            if batch_idx % 10 == 0:
                self.logger.debug(f'训练批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.6f}')
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            auc = 0.0
            
        predicted_classes = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predicted_classes)
        
        return avg_loss, auc, accuracy
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, Dict[str, float]]:
        """
        模型验证和指标计算
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均损失、AUC、准确率和详细指标字典的元组
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 收集预测结果
                probabilities = torch.softmax(output, dim=1)
                predictions = probabilities[:, 1].detach().cpu().numpy()
                labels = target.detach().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            auc = 0.0
            
        predicted_classes = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predicted_classes)
        
        # 详细指标
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision_score(all_labels, predicted_classes, average='binary', zero_division=0),
            'recall': recall_score(all_labels, predicted_classes, average='binary', zero_division=0),
            'f1': f1_score(all_labels, predicted_classes, average='binary', zero_division=0)
        }
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, predicted_classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            })
        
        return avg_loss, auc, accuracy, metrics
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        完整训练流程，包含早停和检查点保存
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        epochs = self.config['training'].get('max_epochs', self.config['training'].get('epochs', 100))
        patience = self.config['training'].get('early_stopping', {}).get('patience', 
                   self.config['training'].get('patience', 10))
        
        if self.quick_test:
            epochs = min(2, epochs)  # 快速测试只运行2个epoch
            patience = 1
        
        self.logger.info(f"开始训练，总epoch数: {epochs}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_auc, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_auc, val_acc, val_metrics = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"训练损失: {train_loss:.4f}, 训练AUC: {train_auc:.4f}, 训练准确率: {train_acc:.4f} - "
                f"验证损失: {val_loss:.4f}, 验证AUC: {val_auc:.4f}, 验证准确率: {val_acc:.4f} - "
                f"时间: {epoch_time:.2f}s"
            )
            
            # 早停检查
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                
                # 保存最佳模型
                self.save_checkpoint(epoch, val_metrics)
                self.logger.info(f"保存最佳模型，验证AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break
                
        self.logger.info("训练完成")
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        保存模型检查点
        
        Args:
            epoch: 当前epoch
            metrics: 验证指标
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_auc': self.best_val_auc,
            'metrics': metrics,
            'config': self.config,
            'data_config': self.data_config
        }
        
        checkpoint_path = 'checkpoints/best_model.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存到: {checkpoint_path}")
        
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        最终测试评估
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            测试指标字典
        """
        # 加载最佳模型
        try:
            checkpoint_path = 'checkpoints/best_model.pth'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("加载最佳模型进行测试")
        except FileNotFoundError:
            self.logger.warning("未找到保存的模型，使用当前模型进行测试")
        
        # 测试评估
        test_loss, test_auc, test_acc, test_metrics = self.validate(test_loader)
        
        self.logger.info(f"测试结果 - 损失: {test_loss:.4f}, AUC: {test_auc:.4f}, 准确率: {test_acc:.4f}")
        self.logger.info(f"详细指标: {test_metrics}")
        
        return test_metrics
        
    def save_results(self, test_metrics: Dict[str, float]):
        """
        保存实验结果到JSON文件
        
        Args:
            test_metrics: 测试指标
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_config': self.data_config,
            'test_metrics': test_metrics,
            'best_val_auc': self.best_val_auc,
            'quick_test': self.quick_test
        }
        
        results_path = 'results/training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验结果已保存到: {results_path}")
        
    def quick_validation(self):
        """快速验证模式：使用最小预算验证代码正确性"""
        print("🚀 开始 training.py 快速验证...")
        
        try:
            # 1. 生成模拟数据
            print("📊 生成模拟数据...")
            n_samples = 100
            n_features = 50
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            y = np.random.randint(0, 2, n_samples).astype(np.int64)
            
            # 更新配置
            self.data_config['task']['num_features'] = n_features
            self.data_config['task']['num_classes'] = 2
            
            print(f"✅ 模拟数据生成完成: {X.shape}, {y.shape}")
            
            # 2. 准备数据
            print("🔄 准备数据加载器...")
            train_loader, val_loader, test_loader = self.prepare_data(X, y)
            print("✅ 数据加载器准备完成")
            
            # 3. 设置模型
            print("🏗️ 设置模型...")
            self.setup_model()
            print("✅ 模型设置完成")
            
            # 4. 快速训练验证
            print("🚀 开始快速训练验证...")
            self.train(train_loader, val_loader)
            print("✅ 训练验证完成")
            
            # 5. 测试验证
            print("🧪 开始测试验证...")
            test_metrics = self.test(test_loader)
            print("✅ 测试验证完成")
            
            # 6. 保存结果
            print("💾 保存结果...")
            self.save_results(test_metrics)
            print("✅ 结果保存完成")
            
            print("🎉 training.py 快速验证成功完成！")
            return True
            
        except Exception as e:
            print(f"❌ 快速验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False


class SimpleMambaModel(nn.Module):
    """简化的Mamba模型替代实现"""
    
    def __init__(self, input_dim: int, d_model: int, n_layer: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 简化的Transformer-like层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len) or (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        # 投影到模型维度
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # 通过层
        for layer in self.layers:
            x = layer(x)
        
        # 归一化
        x = self.norm(x)
        
        # 池化（取平均）
        x = x.mean(dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output


def main():
    """主函数 - 完整训练流程执行"""
    parser = argparse.ArgumentParser(description='Mamba模型训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data-config', type=str, default='data.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--data-path', type=str, default=None,
                       help='数据文件路径（可选）')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速验证模式：使用最小预算验证训练脚本正确性')
    
    args = parser.parse_args()
    
    try:
        input(args.quick_test)
        # 1. 创建训练器实例
        trainer = MambaTrainer(
            config_path=args.config,
            data_config_path=args.data_config,
            quick_test=args.quick_test
        )
        
        # 快速验证模式
        if args.quick_test:
            success = trainer.quick_validation()
            if success:
                print("✅ 快速验证成功！代码可以正常运行。")
                return 0
            else:
                print("❌ 快速验证失败！请检查代码。")
                return 1
        
        # 正常训练模式
        print("🚀 开始正常训练模式...")
        
        # 2. 加载和准备数据
        print("📊 加载数据...")
        X, y = trainer.load_data(args.data_path)
        
        print("🔄 准备数据...")
        train_loader, val_loader, test_loader = trainer.prepare_data(X, y)
        
        # 3. 设置模型
        print("🏗️ 设置模型...")
        trainer.setup_model()
        
        # 4. 执行训练
        print("🚀 开始训练...")
        trainer.train(train_loader, val_loader)
        
        # 5. 进行测试
        print("🧪 开始测试...")
        test_metrics = trainer.test(test_loader)
        
        # 6. 保存结果
        print("💾 保存结果...")
        trainer.save_results(test_metrics)
        
        print("🎉 训练完成！")
        print(f"最终测试结果: {test_metrics}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)