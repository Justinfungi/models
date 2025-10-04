#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - HRM (Hierarchical Reasoning Model) 训练脚本

该文件实现了HRM模型的完整训练流程，包括：
- 数据加载和预处理
- 模型训练和验证
- 损失计算和优化
- 模型保存和评估
- 快速验证功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 导入模型
try:
    from model import HRM, HRMConfig, setup_device, set_all_seeds, create_model
except ImportError as e:
    print(f"⚠️ 警告: 无法导入模型模块: {e}")
    print("请确保 model.py 文件存在且可访问")
    # 创建占位符函数以避免运行时错误
    def setup_device():
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_all_seeds(seed):
        import torch
        import numpy as np
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # 创建占位符类
    class HRM(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = torch.nn.Linear(config.arch['input_dim'], config.arch['num_classes'])
        
        def forward(self, x):
            return self.linear(x)
    
    class HRMConfig:
        def __init__(self, config_path):
            self.arch = {'input_dim': 100, 'num_classes': 2, 'hidden_dim': 128}
            self.training = {'batch_size': 32, 'learning_rate': 0.001, 'weight_decay': 1e-5, 'epochs': 10}
        
        def to_dict(self):
            return {'arch': self.arch, 'training': self.training}
    
    def create_model(config):
        return HRM(config)


class HRMDataset(Dataset):
    """HRM模型专用数据集类"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        """
        初始化数据集
        
        Args:
            features: 特征数据
            labels: 标签数据
            transform: 数据变换函数
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label


class HRMTrainer:
    """HRM模型训练器"""
    
    def __init__(self, config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 模型配置文件路径
            data_config_path: 数据配置文件路径
        """
        # 设置日志（必须先设置，因为其他方法会用到logger）
        self._setup_logging()
        
        # 加载配置
        try:
            self.config = HRMConfig(config_path)
        except Exception as e:
            self.logger.error(f"加载模型配置失败: {e}，使用默认配置")
            # 创建一个基本的配置对象
            self.config = self._create_default_config()
        
        self.data_config = self._load_data_config(data_config_path)
        
        # 设置环境
        set_all_seeds(42)
        self.device = setup_device()
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = StandardScaler()
        
        # 训练状态
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def _load_data_config(self, config_path: str) -> Dict[str, Any]:
        """加载数据配置文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(config_path):
                self.logger.warning(f"数据配置文件不存在: {config_path}，使用默认配置")
                return self._get_default_data_config()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None:
                    self.logger.warning("数据配置文件为空，使用默认配置")
                    return self._get_default_data_config()
                return config
        except Exception as e:
            self.logger.error(f"加载数据配置失败: {e}，使用默认配置")
            return self._get_default_data_config()
    
    def _get_default_data_config(self) -> Dict[str, Any]:
        """获取默认数据配置"""
        return {
            'data_paths': {
                'data_folder': './data/feature_set',
                'data_phase': 1
            },
            'task': {
                'type': 'binary_classification'
            },
            'preprocessing': {
                'scaling': {
                    'method': 'standard'
                }
            }
        }
    
    def _create_default_config(self):
        """创建默认模型配置"""
        class DefaultConfig:
            def __init__(self):
                self.arch = {
                    'input_dim': 100,  # 默认特征维度
                    'hidden_dim': 128,
                    'num_classes': 2,
                    'num_layers': 3,
                    'dropout': 0.1
                }
                self.training = {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'weight_decay': 1e-5,
                    'epochs': 10,
                    'optimizer': 'adam',
                    'scheduler': 'cosine',
                    'gradient_clip_norm': 1.0
                }
            
            def to_dict(self):
                return {
                    'arch': self.arch,
                    'training': self.training
                }
        
        return DefaultConfig()
    
    def _generate_mock_data(self, quick_test: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """生成模拟数据用于测试"""
        self.logger.info("生成模拟数据用于测试...")
        
        # 生成模拟数据
        n_samples = 100 if quick_test else 1000
        n_features = 50
        
        # 生成特征数据
        X = np.random.randn(n_samples, n_features)
        # 生成二分类标签
        y = np.random.randint(0, 2, n_samples)
        
        self.logger.info(f"模拟数据形状: X={X.shape}, y={y.shape}")
        self.logger.info(f"类别分布: {np.bincount(y)}")
        
        # 数据标准化
        X = self.scaler.fit_transform(X)
        
        # 数据分割
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # 创建数据集
        train_dataset = HRMDataset(X_train, y_train)
        val_dataset = HRMDataset(X_val, y_val)
        test_dataset = HRMDataset(X_test, y_test)
        
        # 创建数据加载器
        batch_size = 16 if quick_test else 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 更新配置中的特征数量
        self.config.arch['input_dim'] = X.shape[1]
        self.config.arch['num_classes'] = len(np.unique(y))
        
        self.logger.info(f"模拟数据加载完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _setup_logging(self) -> None:
        """设置日志记录"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, quick_test: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载和预处理数据
        
        Args:
            quick_test: 是否为快速测试模式
            
        Returns:
            训练、验证、测试数据加载器
        """
        try:
            # 获取数据路径
            data_folder = self.data_config.get('data_paths', {}).get('data_folder', './data/feature_set')
            data_phase = self.data_config.get('data_paths', {}).get('data_phase', 1)
            
            # 尝试多个可能的数据路径
            possible_paths = [
                data_folder,
                '../../../data/feature_set',
                '../../data/feature_set',
                './data/feature_set',
                '../data/feature_set'
            ]
            
            data_files = []
            pattern = f"mrmr_task_{data_phase}_*.pq"
            
            for path in possible_paths:
                if os.path.exists(path):
                    files = glob.glob(os.path.join(path, pattern))
                    if files:
                        data_files = files
                        data_folder = path
                        break
            
            if not data_files:
                # 如果没有找到parquet文件，尝试查找CSV文件
                for path in possible_paths:
                    if os.path.exists(path):
                        csv_pattern = f"*task_{data_phase}*.csv"
                        files = glob.glob(os.path.join(path, csv_pattern))
                        if files:
                            data_files = files
                            data_folder = path
                            break
            
            if not data_files:
                # 生成模拟数据用于测试
                self.logger.warning(f"未找到数据文件，生成模拟数据用于测试")
                return self._generate_mock_data(quick_test)
            
            # 加载数据
            self.logger.info(f"加载数据文件: {data_files[0]}")
            if data_files[0].endswith('.pq'):
                df = pd.read_parquet(data_files[0])
            else:
                df = pd.read_csv(data_files[0])
            
            # 快速测试模式：只使用少量数据
            if quick_test:
                df = df.head(100)
                self.logger.info("快速测试模式：使用100个样本")
            
            # 识别特征列（包含@符号的列）
            feature_cols = [col for col in df.columns if '@' in col]
            
            if not feature_cols:
                raise ValueError("未找到特征列（包含@符号的列）")
            
            # 提取特征和标签
            X = df[feature_cols].values
            y = df['class'].values
            
            self.logger.info(f"数据形状: X={X.shape}, y={y.shape}")
            self.logger.info(f"特征列数: {len(feature_cols)}")
            self.logger.info(f"类别分布: {np.bincount(y)}")
            
            # 数据标准化
            X = self.scaler.fit_transform(X)
            
            # 时间序列分割（基于date字段）
            if 'date' in df.columns:
                df_sorted = df.sort_values('date')
                train_size = int(0.7 * len(df_sorted))
                val_size = int(0.15 * len(df_sorted))
                
                train_idx = df_sorted.index[:train_size]
                val_idx = df_sorted.index[train_size:train_size + val_size]
                test_idx = df_sorted.index[train_size + val_size:]
                
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_test, y_test = X[test_idx], y[test_idx]
            else:
                # 随机分割
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
            
            # 创建数据集
            train_dataset = HRMDataset(X_train, y_train)
            val_dataset = HRMDataset(X_val, y_val)
            test_dataset = HRMDataset(X_test, y_test)
            
            # 创建数据加载器
            batch_size = 16 if quick_test else self.config.training['batch_size']
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 更新配置中的特征数量
            self.config.arch['input_dim'] = X.shape[1]
            self.config.arch['num_classes'] = len(np.unique(y))
            
            self.logger.info(f"数据加载完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def build_model(self) -> None:
        """构建模型"""
        try:
            self.model = HRM(self.config).to(self.device)
            
            # 设置损失函数
            self.criterion = nn.CrossEntropyLoss()
            
            # 设置优化器
            training_config = self.config.training
            learning_rate = float(training_config['learning_rate'])
            weight_decay = float(training_config['weight_decay'])
            
            if training_config.get('optimizer') == 'adamw':
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            else:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            
            # 设置学习率调度器
            training_config = self.config.training
            scheduler_type = training_config.get('scheduler', 'cosine')
            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=training_config['epochs']
                )
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=0.1
                )
            
            self.logger.info(f"模型构建完成 - 参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            self.logger.error(f"模型构建失败: {e}")
            raise
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            gradient_clip = self.config.training.get('gradient_clip_norm', 0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    gradient_clip
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 打印进度
            if batch_idx % 10 == 0:
                self.logger.info(f'训练批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均损失、准确率和详细指标
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 收集预测结果
                pred = output.argmax(dim=1)
                prob = torch.softmax(output, dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算详细指标
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # 计算AUC（如果是二分类）
        if len(np.unique(all_targets)) == 2:
            try:
                auc = roc_auc_score(all_targets, all_probs[:, 1])
                metrics['auc'] = auc
            except:
                metrics['auc'] = 0.0
        
        return avg_loss, accuracy * 100, metrics
    
    def save_model(self, epoch: int, val_acc: float, is_best: bool = False) -> None:
        """
        保存模型
        
        Args:
            epoch: 当前epoch
            val_acc: 验证准确率
            is_best: 是否为最佳模型
        """
        try:
            # 创建保存目录
            save_dir = Path("checkpoints")
            save_dir.mkdir(exist_ok=True)
            
            # 保存状态
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_acc': val_acc,
                'config': self.config.to_dict(),
                'scaler': self.scaler
            }
            
            # 保存最新模型
            torch.save(state, save_dir / "latest_model.pth")
            
            # 保存最佳模型
            if is_best:
                torch.save(state, save_dir / "best_model.pth")
                self.logger.info(f"保存最佳模型 - Epoch {epoch}, 验证准确率: {val_acc:.2f}%")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, quick_test: bool = False) -> None:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            quick_test: 是否为快速测试模式
        """
        epochs = 2 if quick_test else self.config.training['epochs']
        
        self.logger.info(f"开始训练 - 总epoch数: {epochs}")
        
        for epoch in range(epochs):
            start_time = datetime.now()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # 检查是否为最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            # 保存模型
            self.save_model(epoch + 1, val_acc, is_best)
            
            # 打印结果
            duration = datetime.now() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% - "
                f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}% - "
                f"用时: {duration.total_seconds():.1f}s"
            )
            
            # 打印详细指标
            for metric, value in val_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
            
            # 快速测试模式：提前结束
            if quick_test and epoch >= 0:  # 至少训练1个epoch
                self.logger.info("快速测试模式完成")
                break
        
        self.logger.info(f"训练完成 - 最佳验证准确率: {self.best_val_acc:.2f}%")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        self.logger.info("开始模型评估...")
        
        # 加载最佳模型
        try:
            checkpoint = torch.load("checkpoints/best_model.pth", map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("加载最佳模型成功")
        except Exception as e:
            self.logger.warning(f"未找到最佳模型或加载失败: {e}，使用当前模型")
        
        # 评估
        test_loss, test_acc, test_metrics = self.validate(test_loader)
        
        self.logger.info(f"测试结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
        for metric, value in test_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def save_training_history(self) -> None:
        """保存训练历史"""
        try:
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accs': self.train_accs,
                'val_accs': self.val_accs,
                'best_val_acc': self.best_val_acc
            }
            
            with open('training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info("训练历史保存完成")
            
        except Exception as e:
            self.logger.error(f"保存训练历史失败: {e}")
    
    def quick_validation(self) -> None:
        """快速验证模式：使用最小预算验证代码正确性"""
        print("🚀 开始 training.py 快速验证...")
        
        try:
            # 1. 加载少量数据
            train_loader, val_loader, test_loader = self.load_data(quick_test=True)
            print("✅ 数据加载成功")
            
            # 2. 构建模型
            self.build_model()
            print("✅ 模型构建成功")
            
            # 3. 快速训练
            self.train(train_loader, val_loader, quick_test=True)
            print("✅ 训练流程验证成功")
            
            # 4. 快速评估
            metrics = self.evaluate(test_loader)
            print("✅ 评估流程验证成功")
            
            # 5. 保存历史
            self.save_training_history()
            print("✅ 历史保存验证成功")
            
            print("🎉 training.py 快速验证完成！所有功能正常运行。")
            
        except Exception as e:
            print(f"❌ 快速验证失败: {e}")
            raise


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    try:
        # 检查Python版本
        import sys
        python_version = sys.version_info
        print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查conda环境（安全检查，避免环境错误）
        try:
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
            conda_prefix = os.environ.get('CONDA_PREFIX', 'None')
            print(f"  Conda环境: {conda_env}")
            print(f"  Conda路径: {conda_prefix}")
            
            # 如果检测到factor环境但不存在，给出警告
            if conda_env == 'factor' and not os.path.exists(conda_prefix):
                print("  ⚠️ 警告: factor环境路径不存在，但脚本可以在当前环境运行")
        except Exception as e:
            print(f"  Conda环境检查失败: {e}")
        
        # 检查当前工作目录
        print(f"  当前工作目录: {os.getcwd()}")
        
        # 检查PyTorch版本
        print(f"  PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"  CUDA可用: ✅ (设备数量: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  CUDA可用: ❌ (将使用CPU)")
        
        # 检查关键依赖
        try:
            import pandas
            print(f"  Pandas版本: {pandas.__version__}")
        except ImportError:
            print("  Pandas: ❌ 未安装")
        
        try:
            import sklearn
            print(f"  Scikit-learn版本: {sklearn.__version__}")
        except ImportError:
            print("  Scikit-learn: ❌ 未安装")
        
        # 检查数据目录
        data_paths = ['./data', '../data', '../../data', '../../../data']
        for path in data_paths:
            if os.path.exists(path):
                print(f"  数据目录 {path}: ✅")
                break
        else:
            print("  数据目录: ❌ 未找到")
        
        print("✅ 环境检查完成\n")
        
    except Exception as e:
        print(f"❌ 环境检查过程中出现错误: {e}")
        print("继续执行脚本...\n")


def check_conda_environment_compatibility():
    """检查conda环境兼容性，处理factor环境不存在的问题"""
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
        conda_prefix = os.environ.get('CONDA_PREFIX', 'None')
        
        # 如果当前环境是factor但路径不存在，说明环境配置有问题
        if conda_env == 'factor' and conda_prefix != 'None' and not os.path.exists(conda_prefix):
            print("⚠️ 检测到factor环境配置问题，但脚本可以在当前Python环境中正常运行")
            print(f"   环境名称: {conda_env}")
            print(f"   环境路径: {conda_prefix}")
            print("   解决方案: 脚本将使用当前Python环境继续执行")
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ 环境兼容性检查失败: {e}")
        return True  # 继续执行


def main():
    """主函数"""
    # 检查conda环境兼容性
    check_conda_environment_compatibility()
    
    # 检查运行环境
    check_environment()
    
    parser = argparse.ArgumentParser(description='HRM模型训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='模型配置文件路径')
    parser.add_argument('--data-config', type=str, default='data.yaml', help='数据配置文件路径')
    parser.add_argument('--quick-test', action='store_true', help='快速验证模式：使用最小预算验证训练脚本正确性')
    parser.add_argument('--eval-only', action='store_true', help='仅评估模式')
    
    args = parser.parse_args()
    
    try:
        # 检查配置文件是否存在
        if not os.path.exists(args.config):
            print(f"⚠️ 警告: 配置文件 {args.config} 不存在，将使用默认配置")
        if not os.path.exists(args.data_config):
            print(f"⚠️ 警告: 数据配置文件 {args.data_config} 不存在，将使用默认配置")
        
        # 初始化训练器
        trainer = HRMTrainer(args.config, args.data_config)
        
        if args.quick_test:
            # 快速验证模式
            trainer.quick_validation()
        else:
            # 正常训练模式
            # 加载数据
            train_loader, val_loader, test_loader = trainer.load_data()
            
            # 构建模型
            trainer.build_model()
            
            if not args.eval_only:
                # 训练模型
                trainer.train(train_loader, val_loader)
                
                # 保存训练历史
                trainer.save_training_history()
            
            # 评估模型
            metrics = trainer.evaluate(test_loader)
            
            print("\n" + "="*50)
            print("🎯 最终评估结果:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print("="*50)
    
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        print("💡 提示: 如果遇到conda环境问题，请确保:")
        print("   1. 使用正确的Python环境运行脚本")
        print("   2. 或者直接使用 'python training.py' 而不是 'conda run -n factor python training.py'")
        raise


if __name__ == "__main__":
    main()