#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - 混合频率量价因子模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import yaml
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import MixedFrequencyFactorModel, ModelConfig

class DataConfig:
    """数据配置类"""
    
    def __init__(self, config_path: str = "data.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载数据配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"数据配置文件 {self.config_path} 不存在")
        except yaml.YAMLError as e:
            raise ValueError(f"数据配置文件格式错误: {e}")

class CustomDataLoader:
    """数据加载器"""
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, quick_test: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        加载数据
        
        Args:
            quick_test: 是否为快速测试模式
            
        Returns:
            数据DataFrame和特征列名列表
        """
        config = self.data_config.config
        
        # 获取数据路径
        main_path = config['data_source']['train_path']
        alternative_paths = config['data_source'].get('alternative_paths', [])
        
        # 尝试加载数据
        data_path = None
        for path in [main_path] + alternative_paths:
            if os.path.exists(path):
                data_path = path
                break
                
        if data_path is None:
            raise FileNotFoundError(f"无法找到数据文件，尝试的路径: {[main_path] + alternative_paths}")
            
        self.logger.info(f"从 {data_path} 加载数据")
        
        # 加载数据
        if data_path.endswith('.pq') or data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
            
        # 快速测试模式：只使用前100行
        if quick_test:
            df = df.head(100)
            self.logger.info(f"快速测试模式：使用 {len(df)} 行数据")
        
        # 获取特征列
        feature_columns = self._get_feature_columns(df)
        
        self.logger.info(f"数据形状: {df.shape}, 特征列数: {len(feature_columns)}")
        
        return df, feature_columns
        
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列名"""
        config = self.data_config.config
        
        # 如果配置中指定了特征列
        if config['data_schema'].get('feature_columns'):
            return config['data_schema']['feature_columns']
            
        # 自动检测特征列（包含@符号的列）
        feature_columns = [col for col in df.columns if '@' in col]
        
        # 如果没有@符号的列，使用数值列（排除必需列和排除列）
        if not feature_columns:
            required_cols = config['data_schema']['required_columns']
            exclude_cols = config['data_schema'].get('exclude_columns', [])
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols 
                             if col not in required_cols and col not in exclude_cols]
        
        return feature_columns

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        预处理数据
        
        Args:
            df: 原始数据
            feature_columns: 特征列名
            
        Returns:
            (周频数据, 日频数据, 日内数据)元组和标签张量
        """
        config = self.data_config.config
        
        # 处理缺失值
        df_processed = df.copy()
        
        # 填充特征列的缺失值
        for col in feature_columns:
            if col in df_processed.columns:
                if df_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    df_processed[col] = df_processed[col].fillna(0)
        
        # 提取特征
        X = df_processed[feature_columns].values.astype(np.float32)
        
        # 处理标签
        target_column = config['data_schema'].get('target_column', 'class')
        if target_column in df_processed.columns:
            y = df_processed[target_column].values
        else:
            # 如果没有指定目标列，使用最后一列
            y = df_processed.iloc[:, -1].values
            
        # 标签编码
        if y.dtype == 'object' or len(np.unique(y)) <= 10:
            # 分类任务
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        
        # 特征标准化
        if config['data_preprocessing'].get('normalize_features', True):
            X = self._normalize_features(X)
            
        # 将特征分割为三个频率
        # 简单分割：假设特征按顺序分为三部分
        n_features = X.shape[1]
        weekly_size = n_features // 3
        daily_size = n_features // 3
        intraday_size = n_features - weekly_size - daily_size
        
        weekly_data = X[:, :weekly_size]
        daily_data = X[:, weekly_size:weekly_size + daily_size]
        intraday_data = X[:, weekly_size + daily_size:]
        
        # 转换为张量并添加序列维度（LSTM需要3D输入）
        weekly_tensor = torch.FloatTensor(weekly_data).unsqueeze(1)  # [batch, 1, features]
        daily_tensor = torch.FloatTensor(daily_data).unsqueeze(1)
        intraday_tensor = torch.FloatTensor(intraday_data).unsqueeze(1)
        y_tensor = torch.LongTensor(y)
        
        self.logger.info(f"预处理完成 - 周频形状: {weekly_tensor.shape}, 日频形状: {daily_tensor.shape}, 日内形状: {intraday_tensor.shape}")
        self.logger.info(f"标签形状: {y_tensor.shape}, 标签分布: {np.bincount(y)}")
        
        return (weekly_tensor, daily_tensor, intraday_tensor), y_tensor
        
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """标准化特征"""
        # Z-score标准化
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # 避免除零
        X_normalized = (X - mean) / std
        return X_normalized

class Trainer:
    """训练器"""
    
    def __init__(self, model: nn.Module, model_config: ModelConfig, data_config: DataConfig):
        self.model = model
        self.model_config = model_config
        self.data_config = data_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)
        
        # 初始化训练组件
        self._setup_training_components()
        
    def _setup_training_components(self):
        """设置训练组件"""
        config = self.model_config.get_training_config()
        
        # 优化器
        optimizer_config = config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")
            
        # 损失函数
        loss_config = config['loss']
        if loss_config['type'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_config['type'] == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_config['type']}")
            
        # 学习率调度器
        scheduler_config = config.get('scheduler')
        if scheduler_config and scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config and scheduler_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max']
            )
        else:
            self.scheduler = None
            
    def train(self, X: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], y: torch.Tensor, quick_test: bool = False) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X: (周频数据, 日频数据, 日内数据)元组
            y: 标签张量
            quick_test: 是否为快速测试模式
            
        Returns:
            训练历史
        """
        config = self.model_config.get_training_config()
        
        # 快速测试模式参数调整
        if quick_test:
            epochs = 2
            batch_size = 16
            validation_split = 0.2
            self.logger.info("快速测试模式：epochs=2, batch_size=16")
        else:
            epochs = config['epochs']
            batch_size = config['batch_size']
            validation_split = config.get('validation_split', 0.2)
            
        # 数据分割
        weekly_data, daily_data, intraday_data = X
        dataset_size = len(weekly_data)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        dataset = TensorDataset(weekly_data, daily_data, intraday_data, y)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = TorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = TorchDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger.info(f"开始训练 - 设备: {self.device}, 训练样本: {train_size}, 验证样本: {val_size}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # 验证阶段
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model('best_model.pth')
                
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                f"time: {epoch_time:.2f}s"
            )
            
            # 快速测试模式：提前结束
            if quick_test and epoch >= 0:  # 至少运行1个epoch
                self.logger.info("快速测试模式：提前结束训练")
                break
                
        total_time = time.time() - start_time
        self.logger.info(f"训练完成 - 总时间: {total_time:.2f}s, 最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存最终模型
        self._save_model('final_model.pth')
        
        return history
        
    def _train_epoch(self, train_loader: TorchDataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (weekly, daily, intraday, target) in enumerate(train_loader):
            weekly = weekly.to(self.device)
            daily = daily.to(self.device)
            intraday = intraday.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(weekly, daily, intraday)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def _validate_epoch(self, val_loader: TorchDataLoader) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for weekly, daily, intraday, target in val_loader:
                weekly = weekly.to(self.device)
                daily = daily.to(self.device)
                intraday = intraday.to(self.device)
                target = target.to(self.device)
                output = self.model(weekly, daily, intraday)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def _save_model(self, filename: str):
        """保存模型"""
        save_path = os.path.join('checkpoints', filename)
        os.makedirs('checkpoints', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model_config.config,
        }, save_path)
        
        self.logger.info(f"模型已保存到: {save_path}")

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def quick_validation():
    """快速验证模式：使用最小预算验证代码正确性"""
    print("🚀 开始 training.py 快速验证...")
    
    try:
        # 设置日志
        setup_logging("INFO")
        logger = logging.getLogger(__name__)
        
        # 加载配置
        logger.info("加载配置文件...")
        model_config = ModelConfig("config.yaml")
        data_config = DataConfig("data.yaml")
        
        # 加载数据
        logger.info("加载数据...")
        data_loader = CustomDataLoader(data_config)
        df, feature_columns = data_loader.load_data(quick_test=True)
        
        # 预处理数据
        logger.info("预处理数据...")
        preprocessor = DataPreprocessor(data_config)
        X, y = preprocessor.preprocess(df, feature_columns)
        
        # 创建模型
        logger.info("创建模型...")
        model = MixedFrequencyFactorModel(model_config)
        
        # 创建训练器
        logger.info("创建训练器...")
        trainer = Trainer(model, model_config, data_config)
        
        # 快速训练
        logger.info("开始快速训练...")
        history = trainer.train(X, y, quick_test=True)
        
        logger.info("✅ 快速验证成功完成！")
        logger.info(f"训练历史: {history}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='混合频率量价因子模型训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='模型配置文件路径')
    parser.add_argument('--data-config', type=str, default='data.yaml', help='数据配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--quick-test', action='store_true', help='快速验证模式：使用最小预算验证训练脚本正确性')
    
    args = parser.parse_args()
    
    # 快速验证模式
    if args.quick_test:
        success = quick_validation()
        sys.exit(0 if success else 1)
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        logger.info("加载配置文件...")
        model_config = ModelConfig(args.config)
        data_config = DataConfig(args.data_config)
        
        # 加载数据
        logger.info("加载数据...")
        data_loader = CustomDataLoader(data_config)
        df, feature_columns = data_loader.load_data()
        
        # 预处理数据
        logger.info("预处理数据...")
        preprocessor = DataPreprocessor(data_config)
        X, y = preprocessor.preprocess(df, feature_columns)
        
        # 创建模型
        logger.info("创建模型...")
        model = MixedFrequencyFactorModel(model_config)
        
        # 创建训练器
        logger.info("创建训练器...")
        trainer = Trainer(model, model_config, data_config)
        
        # 训练模型
        logger.info("开始训练...")
        history = trainer.train(X, y)
        
        # 保存训练历史
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
            
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()