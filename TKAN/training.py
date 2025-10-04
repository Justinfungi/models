#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - TKAN模型训练脚本
实现完整的TKAN模型训练流程，支持二元分类任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import os
import json
import random
from typing import Dict, Any, Optional, Tuple, Union, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from model import TKAN, set_all_seeds, setup_device


class TKANTrainer:
    """TKAN模型训练器类"""
    
    def __init__(self, config_path: str, data_config_path: str, quick_test: bool = False):
        """
        初始化训练器
        
        Args:
            config_path: 模型配置文件路径
            data_config_path: 数据配置文件路径
            quick_test: 是否为快速测试模式
        """
        self.config_path = config_path
        self.data_config_path = data_config_path
        self.quick_test = quick_test
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.data_config = self._load_config(data_config_path)
        
        # 设置设备和随机种子
        self.device = setup_device()
        set_all_seeds(42)
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = StandardScaler()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        # 设置日志
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise FileNotFoundError(f"无法加载配置文件 {config_path}: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志记录"""
        logger = logging.getLogger('TKANTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_data(self) -> None:
        """加载和预处理数据"""
        try:
            self.logger.info("开始加载数据...")
            
            # 构建数据文件路径
            data_folder = self.data_config['data_paths']['data_folder']
            data_phase = self.data_config['data_paths']['data_phase']
            data_file_pattern = self.data_config['data_paths']['data_file']
            
            # 查找匹配的数据文件
            import glob
            file_pattern = os.path.join(data_folder, data_file_pattern)
            data_files = glob.glob(file_pattern)
            
            if not data_files:
                raise FileNotFoundError(f"未找到匹配的数据文件: {file_pattern}")
            
            # 加载第一个匹配的文件
            data_file = data_files[0]
            self.logger.info(f"加载数据文件: {data_file}")
            
            # 读取Parquet文件
            df = pd.read_parquet(data_file)
            self.logger.info(f"数据形状: {df.shape}")
            
            # 快速测试模式：只使用少量数据
            if self.quick_test:
                df = df.head(100)
                self.logger.info(f"快速测试模式：使用 {len(df)} 个样本")
            
            self.raw_data = df
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def prepare_data(self) -> None:
        """准备训练数据"""
        try:
            self.logger.info("开始数据预处理...")
            
            df = self.raw_data.copy()
            
            # 识别特征列（包含@符号的列）
            feature_identifier = self.data_config['data_format']['feature_identifier']
            feature_cols = [col for col in df.columns if feature_identifier in col]
            
            if not feature_cols:
                raise ValueError(f"未找到包含 '{feature_identifier}' 的特征列")
            
            self.logger.info(f"识别到 {len(feature_cols)} 个特征列")
            
            # 获取特征和标签
            X = df[feature_cols].values.astype(np.float32)
            y = df['class'].values.astype(np.float32)
            
            # 处理缺失值
            X = np.nan_to_num(X, nan=0.0)
            
            # 时间序列分割
            if 'date' in df.columns and not self.quick_test:
                dates = pd.to_datetime(df['date'])
                # 按时间排序
                sort_idx = dates.argsort()
                X = X[sort_idx]
                y = y[sort_idx]
                dates = dates.iloc[sort_idx]
                
                # 前1年作为训练集，剩余作为测试集
                train_end_date = dates.min() + pd.DateOffset(years=1)
                train_mask = dates <= train_end_date
                
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[~train_mask]
                y_test = y[~train_mask]
                
                # 确保测试集不为空
                if len(X_test) == 0:
                    # 如果按年份分割导致测试集为空，使用比例分割
                    train_size = int(len(X) * 0.7)
                    val_size = int(len(X) * 0.2)
                    test_size = len(X) - train_size - val_size
                    
                    X_train = X[:train_size]
                    y_train = y[:train_size]
                    X_val = X[train_size:train_size+val_size]
                    y_val = y[train_size:train_size+val_size]
                    X_test = X[train_size+val_size:]
                    y_test = y[train_size+val_size:]
                else:
                    # 从训练集中分出验证集（20%）
                    val_size = int(len(X_train) * 0.2)
                    X_val = X_train[-val_size:]
                    y_val = y_train[-val_size:]
                    X_train = X_train[:-val_size]
                    y_train = y_train[:-val_size]
                
            else:
                # 如果没有日期列，使用简单的时间序列分割
                train_size = int(len(X) * 0.6)
                val_size = int(len(X) * 0.2)
                test_size = len(X) - train_size - val_size
                
                # 确保每个集合至少有1个样本
                if test_size < 1:
                    train_size = int(len(X) * 0.7)
                    val_size = int(len(X) * 0.2)
                    test_size = len(X) - train_size - val_size
                
                if test_size < 1:
                    # 如果数据太少，调整比例
                    train_size = max(1, int(len(X) * 0.8))
                    val_size = max(1, int(len(X) * 0.1))
                    test_size = max(1, len(X) - train_size - val_size)
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:train_size+val_size]
                y_val = y[train_size:train_size+val_size]
                X_test = X[train_size+val_size:train_size+val_size+test_size]
                y_test = y[train_size+val_size:train_size+val_size+test_size]
            
            self.logger.info(f"训练集大小: {len(X_train)}")
            self.logger.info(f"验证集大小: {len(X_val)}")
            self.logger.info(f"测试集大小: {len(X_test)}")
            
            # 特征标准化
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            
            # 转换为PyTorch张量
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)
            
            # 为TKAN添加时间维度（假设每个样本是一个时间步）
            X_train = X_train.unsqueeze(1)  # (batch, 1, features)
            X_val = X_val.unsqueeze(1)
            X_test = X_test.unsqueeze(1)
            
            # 创建数据加载器
            batch_size = self.config['training']['batch_size']
            if self.quick_test:
                batch_size = min(batch_size, 16)
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # 保存数据信息
            self.num_features = X_train.shape[-1]
            self.num_samples = len(X_train)
            
            self.logger.info("数据预处理完成")
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            raise
    
    def setup_model(self) -> None:
        """设置模型、优化器和调度器"""
        try:
            self.logger.info("设置模型...")
            
            # 创建模型
            self.model = TKAN(self.config)
            self.model.to(self.device)
            
            # 通过一个样本来初始化模型（如果有数据加载器）
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                sample_batch = next(iter(self.train_loader))
                sample_input, _ = sample_batch
                sample_input = sample_input.to(self.device)
                _ = self.model(sample_input)  # 触发延迟初始化
            
            # 设置优化器
            optimizer_name = self.config['training']['optimizer'].lower()
            lr = float(self.config['training']['learning_rate'])
            weight_decay = float(self.config['training']['weight_decay'])
            
            if optimizer_name == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
            elif optimizer_name == 'adamw':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"不支持的优化器: {optimizer_name}")
            
            # 设置学习率调度器
            scheduler_config = self.config['training'].get('scheduler', {})
            if isinstance(scheduler_config, str):
                scheduler_type = scheduler_config
                scheduler_config = {}
            else:
                scheduler_type = scheduler_config.get('type', 'step')
            
            if scheduler_type == 'step':
                step_size = scheduler_config.get('step_size', 30)
                gamma = scheduler_config.get('gamma', 0.1)
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == 'cosine':
                T_max = scheduler_config.get('T_max', 50)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
            elif scheduler_type == 'reduce':
                patience = scheduler_config.get('patience', 10)
                factor = scheduler_config.get('factor', 0.5)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=factor)
            else:
                self.scheduler = None
            
            self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
            
        except Exception as e:
            self.logger.error(f"模型设置失败: {e}")
            raise
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 计算损失
            loss = nn.BCELoss()(output.squeeze(), target)
            loss.backward()
            
            # 梯度裁剪
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 快速测试模式：只训练几个批次
            if self.quick_test and batch_idx >= 2:
                break
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = nn.BCELoss()(output.squeeze(), target)
                total_loss += loss.item()
                
                # 收集预测结果
                probs = output.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
                
                # 快速测试模式：只验证几个批次
                if self.quick_test and batch_idx >= 2:
                    break
        
        # 计算指标
        metrics = self.calculate_metrics(all_targets, all_preds, all_probs)
        metrics['loss'] = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """测试模型"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 收集预测结果
                probs = output.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
                
                # 快速测试模式：只测试几个批次
                if self.quick_test and batch_idx >= 2:
                    break
        
        # 计算指标
        metrics = self.calculate_metrics(all_targets, all_preds, all_probs)
        
        return metrics
    
    def calculate_metrics(self, y_true: List, y_pred: List, y_prob: List) -> Dict[str, float]:
        """计算评估指标"""
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
            
            # 计算AUC指标（如果有足够的样本）
            if len(np.unique(y_true)) > 1:
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                    metrics['auc_pr'] = average_precision_score(y_true, y_prob)
                except:
                    metrics['auc_roc'] = 0.0
                    metrics['auc_pr'] = 0.0
            else:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"指标计算失败: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc_roc': 0.0,
                'auc_pr': 0.0
            }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """保存检查点"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': metrics,
                'config': self.config,
                'scaler': self.scaler
            }
            
            # 创建检查点目录
            checkpoint_dir = 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存最新检查点
            checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # 如果是最佳模型，额外保存
            if metrics['loss'] < self.best_val_loss:
                best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                self.best_val_loss = metrics['loss']
                self.logger.info(f"保存最佳模型，验证损失: {metrics['loss']:.4f}")
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        try:
            # 使用weights_only=False来兼容sklearn对象
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.scaler = checkpoint['scaler']
            
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise
    
    def export_model(self, export_path: str) -> None:
        """导出模型"""
        try:
            # 创建导出目录（如果路径包含目录）
            export_dir = os.path.dirname(export_path)
            if export_dir:  # 只有当目录路径不为空时才创建
                os.makedirs(export_dir, exist_ok=True)
            
            # 保存模型
            export_data = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'scaler': self.scaler,
                'num_features': self.num_features
            }
            
            torch.save(export_data, export_path)
            self.logger.info(f"模型已导出到: {export_path}")
            
        except Exception as e:
            self.logger.error(f"模型导出失败: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """执行完整的训练流程"""
        try:
            self.logger.info("开始训练...")
            
            epochs = self.config['training']['epochs']
            patience = self.config['training']['patience']
            
            # 快速测试模式：只训练少量epoch
            if self.quick_test:
                epochs = min(epochs, 2)
                patience = min(patience, 2)
                self.logger.info(f"快速测试模式：训练 {epochs} 个epoch")
            
            for epoch in range(epochs):
                # 训练
                train_loss = self.train_epoch()
                
                # 验证
                val_metrics = self.validate()
                val_loss = val_metrics['loss']
                
                # 更新学习率
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # 记录训练历史
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
                self.training_history['val_f1'].append(val_metrics['f1'])
                
                # 打印进度
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
                
                # 保存检查点
                self.save_checkpoint(epoch, val_metrics)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= patience:
                    self.logger.info(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                    break
            
            # 测试最佳模型
            best_model_path = 'checkpoints/best_model.pth'
            if os.path.exists(best_model_path):
                self.load_checkpoint(best_model_path)
            
            test_metrics = self.test()
            
            # 保存训练历史
            history_path = 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # 导出最终模型
            self.export_model('final_model.pth')
            
            self.logger.info("训练完成！")
            self.logger.info(f"最终测试结果: {test_metrics}")
            
            return {
                'training_history': self.training_history,
                'test_metrics': test_metrics,
                'best_val_loss': self.best_val_loss
            }
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise


def quick_validation():
    """快速验证模式：使用最小预算验证代码正确性"""
    print("🚀 开始 training.py 快速验证...")
    
    try:
        # 检查配置文件是否存在
        config_path = "config.yaml"
        data_config_path = "data.yaml"
        
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
        if not os.path.exists(data_config_path):
            print(f"❌ 数据配置文件不存在: {data_config_path}")
            return False
        
        print("✅ 配置文件检查通过")
        
        # 创建训练器实例
        trainer = TKANTrainer(config_path, data_config_path, quick_test=True)
        print("✅ 训练器初始化成功")
        
        # 尝试加载数据
        try:
            trainer.load_data()
            print("✅ 数据加载成功")
        except Exception as e:
            print(f"⚠️ 数据加载失败（可能是数据文件不存在）: {e}")
            # 创建模拟数据进行验证
            print("🔄 使用模拟数据进行验证...")
            
            # 创建模拟数据
            np.random.seed(42)
            n_samples = 100
            n_features = 50
            
            # 模拟特征数据
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            y = np.random.randint(0, 2, n_samples).astype(np.float32)
            
            # 创建模拟DataFrame
            feature_cols = [f"feature_{i}@test" for i in range(n_features)]
            data_dict = {col: X[:, i] for i, col in enumerate(feature_cols)}
            data_dict['class'] = y
            data_dict['date'] = pd.date_range('2020-01-01', periods=n_samples, freq='D')
            
            trainer.raw_data = pd.DataFrame(data_dict)
            print("✅ 模拟数据创建成功")
        
        # 数据预处理
        trainer.prepare_data()
        print("✅ 数据预处理成功")
        
        # 设置模型
        trainer.setup_model()
        print("✅ 模型设置成功")
        
        # 快速训练验证
        print("🔄 开始快速训练验证...")
        results = trainer.train()
        print("✅ 训练验证成功")
        
        print("🎉 training.py 快速验证完成！所有功能正常运行")
        return True
        
    except Exception as e:
        print(f"❌ 快速验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TKAN模型训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data-config', type=str, default='data.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速验证模式：使用最小预算验证训练脚本正确性')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    if args.quick_test:
        success = quick_validation()
        if success:
            print("✅ 快速验证成功，代码可以正常运行")
        else:
            print("❌ 快速验证失败，请检查代码")
        return
    
    try:
        # 创建训练器
        trainer = TKANTrainer(args.config, args.data_config)
        
        # 加载数据
        trainer.load_data()
        trainer.prepare_data()
        
        # 设置模型
        trainer.setup_model()
        
        # 从检查点恢复（如果指定）
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        results = trainer.train()
        
        print("训练完成！")
        print(f"最终测试指标: {results['test_metrics']}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()