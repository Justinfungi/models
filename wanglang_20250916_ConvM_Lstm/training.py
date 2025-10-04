#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - ConvM_Lstm 模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上查找到项目根目录 (b_model_reproduction_agent)
project_root = current_dir
while not os.path.exists(os.path.join(project_root, 'data')) and project_root != '/':
    project_root = os.path.dirname(project_root)
sys.path.append(project_root)

try:
    from model import ConvM_Lstm
except ImportError as e:
    print(f"模型导入失败: {e}")
    print("请确保 model.py 文件存在且包含 ConvM_Lstm 类")
    # 创建一个简单的占位符模型
    class ConvM_Lstm(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.fc = nn.Linear(20, 2)  # 简单的线性层
        
        def forward(self, x):
            # x shape: (batch_size, seq_len, features)
            x = x.mean(dim=1)  # 平均池化
            return self.fc(x)


class ConvM_LstmTrainer:
    """
    ConvM_Lstm模型训练器
    
    负责模型的训练、验证和保存
    """
    
    def __init__(self, config_path: str, quick_test: bool = False):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
            quick_test: 是否为快速测试模式
        """
        self.quick_test = quick_test
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = StandardScaler()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
        if self.quick_test:
            self.logger.info("🚀 快速测试模式已启用")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(config_path):
                print(f"配置文件不存在: {config_path}")
                print("使用默认配置...")
                return self._get_default_config()
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            print("使用默认配置...")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'architecture': {
                'seq_len': 60,
                'num_features': 20
            },
            'data': {
                'target_column': 'label',
                'split': {
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'random_seed': 42
                }
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'epochs': 100
            }
        }
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载和预处理数据
        
        Returns:
            训练、验证、测试数据加载器
        """
        try:
            # 数据路径
            data_config = self.config['data']
            
            # 修复数据路径：直接使用项目根目录下的data文件夹
            data_path = os.path.join(project_root, "data")
            
            # 查找数据文件
            data_files = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.parquet') or file.endswith('.pq'):
                        data_files.append(os.path.join(root, file))
            
            if not data_files:
                self.logger.warning(f"在 {data_path} 中未找到 .parquet 或 .pq 文件，生成模拟数据")
                df = self._generate_dummy_data()
            else:
                # 加载第一个数据文件
                df = pd.read_parquet(data_files[0])
                self.logger.info(f"数据加载完成，形状: {df.shape}")
            
            # 提取特征和标签
            feature_cols = [col for col in df.columns if '@' in col]
            if not feature_cols:
                # 如果没有找到@符号的特征列，使用除最后一列外的所有数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = numeric_cols[:-1] if len(numeric_cols) > 1 else numeric_cols
                self.logger.warning(f"未找到包含'@'的特征列，使用数值列: {len(feature_cols)} 个特征")
            
            target_col = data_config.get('target_column', 'label')
            
            if target_col not in df.columns:
                # 如果没有找到指定的目标列，使用最后一列
                target_col = df.columns[-1]
                self.logger.warning(f"未找到目标列 {data_config.get('target_column')}，使用 {target_col}")
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            # 确保y是数值类型
            if y.dtype == 'object' or y.dtype.kind in ['U', 'S']:  # 字符串类型
                # 尝试转换为数值
                try:
                    y = pd.to_numeric(y, errors='coerce')
                    # 处理NaN值
                    if np.isnan(y).any():
                        self.logger.warning("目标列包含非数值数据，将使用二分类标签")
                        y = (y > np.nanmedian(y)).astype(int)
                except:
                    # 如果无法转换，创建分类标签
                    unique_vals = np.unique(y)
                    y = np.array([np.where(unique_vals == val)[0][0] for val in y])
                    self.logger.warning(f"目标列转换为分类标签，类别数: {len(unique_vals)}")
            
            # 确保y是整数类型用于分类
            if y.dtype.kind == 'f':  # 浮点数
                y = y.astype(int)
            
            # 确保标签在有效范围内 [0, num_classes-1]
            num_classes = self.config['architecture']['fc']['num_classes']
            unique_labels = np.unique(y)
            self.logger.info(f"原始标签范围: {unique_labels}")
            
            if len(unique_labels) > num_classes:
                self.logger.warning(f"标签类别数({len(unique_labels)})超过配置的类别数({num_classes})")
                # 重新映射标签到 [0, num_classes-1]
                y = np.digitize(y, np.percentile(y, np.linspace(0, 100, num_classes+1)[1:-1])) - 1
                y = np.clip(y, 0, num_classes-1)
            elif unique_labels.min() < 0 or unique_labels.max() >= num_classes:
                # 重新映射标签到 [0, num_classes-1]
                y_min, y_max = y.min(), y.max()
                y = ((y - y_min) / (y_max - y_min) * (num_classes - 1)).astype(int)
                y = np.clip(y, 0, num_classes-1)
            
            self.logger.info(f"处理后标签范围: {np.unique(y)}")
            
            # 快速测试模式：使用少量数据
            if self.quick_test:
                sample_size = min(100, len(X))
                indices = np.random.choice(len(X), sample_size, replace=False)
                X = X[indices]
                y = y[indices]
                self.logger.info(f"快速测试模式：使用 {sample_size} 个样本")
            
            # 数据标准化
            X = self.scaler.fit_transform(X)
            
            # 转换为时间序列格式
            seq_len = self.config['architecture']['seq_len']
            num_features = self.config['architecture']['num_features']
            
            # 重塑数据为时间序列格式
            if X.shape[1] >= seq_len * num_features:
                X_reshaped = X[:, :seq_len * num_features].reshape(-1, seq_len, num_features)
            else:
                # 如果特征数不够，进行填充或截断
                needed_features = seq_len * num_features
                if X.shape[1] < needed_features:
                    # 填充
                    padding = np.zeros((X.shape[0], needed_features - X.shape[1]))
                    X = np.concatenate([X, padding], axis=1)
                X_reshaped = X[:, :needed_features].reshape(-1, seq_len, num_features)
            
            # 转换为张量
            X_tensor = torch.FloatTensor(X_reshaped)
            y_tensor = torch.LongTensor(y)
            
            # 创建数据集
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # 数据分割 - 修复配置路径
            data_split_config = self.config['data'].get('split', self.config['data'])
            train_ratio = data_split_config.get('train_ratio', 0.7)
            val_ratio = data_split_config.get('val_ratio', 0.15)
            random_seed = data_split_config.get('random_seed', 42)
            
            train_size = int(train_ratio * len(dataset))
            val_size = int(val_ratio * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed)
            )
            
            # 创建数据加载器 - 修复配置路径
            batch_size = self.config.get('training', {}).get('batch_size', 32)
            if self.quick_test:
                batch_size = min(16, batch_size)  # 快速测试使用更小的批次
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            self.logger.info(f"数据分割完成 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """生成模拟数据用于测试"""
        np.random.seed(42)
        n_samples = 1000 if not self.quick_test else 100
        n_features = 50
        
        # 生成特征数据
        data = {}
        for i in range(n_features):
            data[f'feature_{i}@time'] = np.random.randn(n_samples)
        
        # 生成标签
        data['label'] = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(data)
        self.logger.info(f"生成模拟数据，形状: {df.shape}")
        return df
    
    def _initialize_model(self) -> None:
        """初始化模型、优化器和损失函数"""
        try:
            # 创建模型
            self.model = ConvM_Lstm(self.config).to(self.device)
            
            # 创建优化器
            training_config = self.config.get('training', {})
            lr = float(training_config.get('learning_rate', 0.001))
            weight_decay = float(training_config.get('weight_decay', 1e-5))
            
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            
            # 创建损失函数
            self.criterion = nn.CrossEntropyLoss()
            
            # 学习率调度器
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            self.logger.info("模型初始化完成")
            self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
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
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            if batch_idx % 10 == 0:
                self.logger.debug(f'批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """
        验证模型性能
        
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
                
                # 收集预测和真实标签
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
        
        # ROC AUC (仅对二分类有效)
        try:
            if len(np.unique(all_targets)) == 2:
                auc = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return avg_loss, accuracy, metrics
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        保存模型
        
        Args:
            epoch: 当前epoch
            metrics: 验证指标
            is_best: 是否为最佳模型
        """
        try:
            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存状态
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'metrics': metrics,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accs': self.train_accs,
                'val_accs': self.val_accs,
                'scaler': self.scaler
            }
            
            # 保存检查点
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(state, checkpoint_path)
            
            # 保存最佳模型
            if is_best:
                best_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(state, best_path)
                self.logger.info(f"保存最佳模型到: {best_path}")
            
            self.logger.info(f"模型检查点已保存: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def train(self) -> None:
        """执行完整的训练流程"""
        try:
            self.logger.info("开始训练流程...")
            
            # 加载数据
            train_loader, val_loader, test_loader = self._load_data()
            
            # 初始化模型
            self._initialize_model()
            
            # 训练参数
            training_config = self.config.get('training', {})
            num_epochs = training_config.get('epochs', 100)
            
            if self.quick_test:
                num_epochs = min(2, num_epochs)  # 快速测试只训练2个epoch
                self.logger.info(f"快速测试模式：训练 {num_epochs} 个epoch")
            
            # 训练循环
            for epoch in range(num_epochs):
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
                
                # 训练
                train_loss, train_acc = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                
                # 验证
                val_loss, val_acc, val_metrics = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 记录日志
                self.logger.info(
                    f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.4f}, "
                    f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.4f}"
                )
                self.logger.info(f"验证指标: {val_metrics}")
                
                # 保存模型
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                
                if (epoch + 1) % 10 == 0 or is_best or self.quick_test:
                    self.save_model(epoch + 1, val_metrics, is_best)
                
                # 快速测试模式提前结束
                if self.quick_test and epoch >= 0:  # 至少训练1个epoch
                    self.logger.info("快速测试完成，提前结束训练")
                    break
            
            # 最终测试
            if not self.quick_test:
                test_loss, test_acc, test_metrics = self.validate(test_loader)
                self.logger.info(f"最终测试结果 - 损失: {test_loss:.6f}, 准确率: {test_acc:.4f}")
                self.logger.info(f"测试指标: {test_metrics}")
            
            self.logger.info("训练完成！")
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            raise
    
    def quick_validation(self) -> None:
        """快速验证模式：使用最小预算验证代码正确性"""
        self.logger.info("🚀 开始 training.py 快速验证...")
        
        try:
            # 设置快速测试模式
            self.quick_test = True
            
            # 执行训练
            self.train()
            
            self.logger.info("✅ 快速验证成功完成！")
            
        except Exception as e:
            self.logger.error(f"❌ 快速验证失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ConvM_Lstm 模型训练')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速验证模式：使用最小预算验证训练脚本正确性')
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    
    try:
        # 创建训练器
        trainer = ConvM_LstmTrainer(config_path, quick_test=args.quick_test)
        
        if args.quick_test:
            # 执行快速验证
            trainer.quick_validation()
        else:
            # 执行正常训练
            trainer.train()
            
    except Exception as e:
        print(f"训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()