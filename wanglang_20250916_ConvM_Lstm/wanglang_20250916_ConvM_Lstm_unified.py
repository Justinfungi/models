#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wanglang_20250916_ConvM_Lstm_unified.py - 统一模型实现
整合训练、推理和文档生成功能，支持Optuna超参数优化
"""

import os
import sys
import time
import yaml
import json
import pickle
import logging
import argparse
import warnings
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# =============================================================================
# 基础配置和工具函数
# =============================================================================

@dataclass
class wanglang_20250916_ConvM_LstmConfig:
    """wanglang_20250916_ConvM_Lstm模型配置类"""
    # 模型基础配置
    model_name: str = "wanglang_20250916_ConvM_Lstm"
    model_type: str = "classification"
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-5
    
    # 性能优化配置
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    gradient_clip_value: float = 1.0
    
    # 模型架构配置
    seq_len: int = 60
    num_features: int = 20
    input_dim: Optional[int] = None  # 动态设置
    
    # 卷积分支配置
    conv_num_branches: int = 4
    conv_kernel_sizes: List[List[int]] = None
    conv_out_channels: int = 32
    
    # LSTM配置
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_bidirectional: bool = False
    lstm_dropout: float = 0.2
    
    # 全连接层配置
    fc_hidden_dim: int = 256
    fc_dropout: float = 0.3
    num_classes: int = 2
    
    # 设备配置
    device: str = "auto"
    seed: int = 42
    
    # 路径配置
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [[1, self.num_features], [3, self.num_features], 
                                    [5, self.num_features], [7, self.num_features]]

def set_all_seeds(seed: int = 42) -> None:
    """设置所有随机种子确保可重现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device(device_choice: str = "auto") -> torch.device:
    """自动设备配置并记录详细的GPU信息"""
    logger = logging.getLogger(__name__)

    if device_choice == "cpu":
        device = torch.device('cpu')
        print("🖥️  强制使用CPU")
        logger.info("设备配置: 强制使用CPU")
    elif device_choice == "cuda":
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 强制使用GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"设备配置: 强制使用GPU - {gpu_name} ({gpu_memory:.1f}GB)")
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA不可用，回退到CPU")
            logger.warning("CUDA不可用，回退到CPU")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            cached_memory = torch.cuda.memory_reserved() / 1024**3

            print(f"🚀 自动检测使用GPU: {gpu_name}")
            print(f"   📊 GPU内存: {gpu_memory:.1f}GB 总量")
            print(f"   📊 GPU数量: {gpu_count}")
            print(f"   📊 当前使用: {current_memory:.2f}GB")
            print(f"   📊 缓存使用: {cached_memory:.2f}GB")

            logger.info(f"设备配置: 自动检测使用GPU")
            logger.info(f"GPU信息: {gpu_name}, {gpu_memory:.1f}GB总内存, {gpu_count}个GPU")
            logger.info(f"GPU内存使用: {current_memory:.2f}GB已用, {cached_memory:.2f}GB缓存")

            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            print("🖥️  GPU不可用，使用CPU")
            logger.info("设备配置: GPU不可用，使用CPU")

    return device

def setup_logging(log_dir: str = "./logs", prefix: str = "unified", log_filename: str = None) -> logging.Logger:
    """设置日志系统"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_filename:
        log_file = log_dir / log_filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{prefix}_{timestamp}.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger

def create_directories(config: wanglang_20250916_ConvM_LstmConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 数据处理类
# =============================================================================

class wanglang_20250916_ConvM_LstmDataLoaderManager:
    """数据加载器管理类"""
    
    def __init__(self, data_config_path: str, config: wanglang_20250916_ConvM_LstmConfig):
        self.data_config_path = data_config_path
        self.config = config
        self.data_config = self._load_data_config()
        self.data_split_info = self.data_config.get('data_split', {})
        self.scaler = StandardScaler()
        
    def _load_data_config(self) -> Dict[str, Any]:
        """加载数据配置文件"""
        try:
            with open(self.data_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'phase_1' in config:
                phase_1_config = config['phase_1']
                config['data_split'] = {
                    "workflow_type": "time_series",
                    "train": {
                        "start_date": phase_1_config['train_data']['start_date'],
                        "end_date": phase_1_config['train_data']['end_date'],
                        "samples": phase_1_config['train_data']['samples'],
                        "duration_years": phase_1_config['train_data']['duration_years'],
                        "file": phase_1_config['train_data']['file']
                    },
                    "validation": {
                        "start_date": phase_1_config['valid_data']['start_date'],
                        "end_date": phase_1_config['valid_data']['end_date'],
                        "samples": phase_1_config['valid_data']['samples'],
                        "duration_years": phase_1_config['valid_data']['duration_years'],
                        "file": phase_1_config['valid_data']['file']
                    },
                    "test": {
                        "start_date": phase_1_config['test_data']['start_date'],
                        "end_date": phase_1_config['test_data']['end_date'],
                        "samples": phase_1_config['test_data']['samples'],
                        "duration_years": phase_1_config['test_data']['duration_years'],
                        "file": phase_1_config['test_data']['file']
                    }
                }
                print(f"✅ 使用Phase 1配置作为默认数据分割")
            
            return config
        except Exception as e:
            raise ValueError(f"无法加载数据配置文件 {self.data_config_path}: {e}")
    
    def get_input_dim_from_dataframe(self, df: pd.DataFrame) -> int:
        """从DataFrame获取特征维度"""
        feature_cols = [col for col in df.columns if '@' in col]
        if not feature_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = numeric_cols[:-1] if len(numeric_cols) > 1 else numeric_cols
        
        features = len(feature_cols)
        print(f"🔍 检测到输入特征维度: {features}")
        return features
    
    def _create_dataloader(self, dataset, shuffle: bool = False, batch_size: int = 32) -> DataLoader:
        """创建优化的数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def _load_real_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """从真实parquet文件加载数据"""
        try:
            data_split = self.data_split_info
            
            # 定义可能的数据路径
            possible_data_paths = [
                "/home/feng.hao.jie/deployment/model_explorer/b_model_reproduction_agent/data/feature_set/",
                "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/data/feature_set/",
                "./data/feature_set/",
                "./",
                ""
            ]
            
            def find_data_file(filename):
                """查找数据文件的实际路径"""
                # 如果是绝对路径且存在，直接返回
                if os.path.isabs(filename) and os.path.exists(filename):
                    return filename
                
                # 如果是相对路径且存在，直接返回
                if os.path.exists(filename):
                    return filename
                
                # 在可能的路径中搜索
                for base_path in possible_data_paths:
                    full_path = os.path.join(base_path, filename)
                    if os.path.exists(full_path):
                        print(f"🔍 找到数据文件: {full_path}")
                        return full_path
                
                # 如果都找不到，抛出错误
                raise FileNotFoundError(f"无法找到数据文件: {filename}")
            
            # 加载训练数据
            train_file = data_split['train']['file']
            train_file_path = find_data_file(train_file)
            train_df = pd.read_parquet(train_file_path)
            
            # 加载验证数据
            valid_file = data_split['validation']['file']
            valid_file_path = find_data_file(valid_file)
            valid_df = pd.read_parquet(valid_file_path)
            
            # 加载测试数据
            test_file = data_split['test']['file']
            test_file_path = find_data_file(test_file)
            test_df = pd.read_parquet(test_file_path)
            
            print(f"✅ 数据加载完成")
            print(f"   训练集: {train_df.shape}")
            print(f"   验证集: {valid_df.shape}")
            print(f"   测试集: {test_df.shape}")
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            raise RuntimeError(f"加载真实数据失败: {e}")
    
    def load_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载数据加载器"""
        train_df, valid_df, test_df = self._load_real_dataframes()
        
        # 动态检测特征维度
        feature_cols = [col for col in train_df.columns if '@' in col]
        if not feature_cols:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = numeric_cols[:-1] if len(numeric_cols) > 1 else numeric_cols
        
        input_dim = len(feature_cols)
        print(f"🔍 检测到 {input_dim} 个特征")
        
        # 更新配置 - 确保所有相关维度都正确设置
        self.config.input_dim = input_dim
        self.config.num_features = input_dim
        
        # 调整序列长度以适应特征数量
        # 如果特征数量较多，可能需要调整序列长度
        if input_dim > 60:
            # 对于大量特征，使用较小的序列长度
            self.config.seq_len = max(10, min(30, input_dim // 3))
        else:
            # 对于较少特征，使用标准序列长度
            self.config.seq_len = min(60, max(10, input_dim))
        
        print(f"🔧 调整序列长度为: {self.config.seq_len}")
        print(f"🔧 设置特征维度为: {self.config.num_features}")
        
        # 提取特征和标签
        target_col = 'label'
        if target_col not in train_df.columns:
            target_col = train_df.columns[-1]
        
        # 处理训练数据
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        
        # 处理验证数据
        X_valid = valid_df[feature_cols].values
        y_valid = valid_df[target_col].values
        
        # 处理测试数据
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # 确保标签是整数类型
        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        y_test = y_test.astype(int)
        
        # 数据标准化
        X_train = self.scaler.fit_transform(X_train)
        X_valid = self.scaler.transform(X_valid)
        X_test = self.scaler.transform(X_test)
        
        # 重塑为时间序列格式
        seq_len = self.config.seq_len
        
        def reshape_to_sequences(X, y):
            # 确保我们有足够的特征来重塑
            available_features = X.shape[1]
            
            # 如果可用特征数量少于需要的，进行填充
            if available_features < input_dim:
                padding = np.zeros((X.shape[0], input_dim - available_features))
                X = np.concatenate([X, padding], axis=1)
                print(f"⚠️ 特征不足，从 {available_features} 填充到 {input_dim}")
            elif available_features > input_dim:
                # 如果特征太多，截取前input_dim个
                X = X[:, :input_dim]
                print(f"⚠️ 特征过多，从 {available_features} 截取到 {input_dim}")
            
            # 现在重塑为序列格式
            # 每个样本重复seq_len次来创建时间序列
            X_expanded = np.tile(X[:, np.newaxis, :], (1, seq_len, 1))
            
            return X_expanded, y
        
        X_train_seq, y_train = reshape_to_sequences(X_train, y_train)
        X_valid_seq, y_valid = reshape_to_sequences(X_valid, y_valid)
        X_test_seq, y_test = reshape_to_sequences(X_test, y_test)
        
        # 转换为张量
        train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train))
        valid_dataset = TensorDataset(torch.FloatTensor(X_valid_seq), torch.LongTensor(y_valid))
        test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test))
        
        # 创建数据加载器
        train_loader = self._create_dataloader(train_dataset, shuffle=True, batch_size=batch_size)
        valid_loader = self._create_dataloader(valid_dataset, shuffle=False, batch_size=batch_size)
        test_loader = self._create_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        
        return train_loader, valid_loader, test_loader
    
    def get_data_split_info(self) -> Dict[str, Any]:
        """获取数据分割信息"""
        return self.data_split_info
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据相关信息"""
        return {
            "workflow_type": self.data_split_info.get("workflow_type", "unknown"),
            "train_period": self.data_split_info.get("train", {}),
            "validation_period": self.data_split_info.get("validation", {}),
            "test_period": self.data_split_info.get("test", {}),
            "data_config": self.data_config
        }

# =============================================================================
# 模型实现
# =============================================================================

class wanglang_20250916_ConvM_LstmModel(nn.Module):
    """wanglang_20250916_ConvM_Lstm模型实现"""
    
    def __init__(self, config: wanglang_20250916_ConvM_LstmConfig):
        super().__init__()
        self.config = config
        
        # 动态获取输入维度
        self.input_dim = getattr(config, 'input_dim', None)
        if self.input_dim is None:
            self.input_dim = config.num_features
            print("⚠️ 配置中未指定input_dim，使用num_features")
        
        self.seq_len = config.seq_len
        self.num_features = self.input_dim
        
        # 卷积分支配置
        self.num_branches = config.conv_num_branches
        self.kernel_sizes = config.conv_kernel_sizes
        self.out_channels = config.conv_out_channels
        
        # 更新kernel_sizes以匹配实际特征数
        self.kernel_sizes = [[k[0], self.num_features] for k in self.kernel_sizes]
        
        # LSTM配置
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm_num_layers = config.lstm_num_layers
        self.lstm_bidirectional = config.lstm_bidirectional
        self.lstm_dropout = config.lstm_dropout
        
        # 全连接层配置
        self.fc_hidden_dim = config.fc_hidden_dim
        self.fc_dropout = config.fc_dropout
        self.num_classes = config.num_classes
        
        # 构建模型层
        self._build_conv_branches()
        self._build_lstm()
        self._build_classifier()
        
    def _build_conv_branches(self):
        """构建多分支卷积层"""
        self.conv_branches = nn.ModuleList()
        
        for i, kernel_size in enumerate(self.kernel_sizes):
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size[0]//2, 0)
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.conv_branches.append(branch)
            
    def _build_lstm(self):
        """构建LSTM层"""
        lstm_input_dim = self.num_branches * self.out_channels + self.num_features
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            bidirectional=self.lstm_bidirectional,
            dropout=self.lstm_dropout if self.lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        self.lstm_output_dim = self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)
        
    def _build_classifier(self):
        """构建分类器"""
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, self.fc_hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_hidden_dim, self.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, num_features = x.shape
        
        # 准备卷积输入
        conv_input = x.view(batch_size * seq_len, 1, 1, num_features)
        
        # 多分支卷积特征提取
        conv_features = []
        for branch in self.conv_branches:
            branch_output = branch(conv_input)
            branch_output = branch_output.squeeze(-1).squeeze(-1)
            conv_features.append(branch_output)
        
        # 合并卷积特征
        conv_output = torch.cat(conv_features, dim=1)
        conv_output = conv_output.view(batch_size, seq_len, -1)
        
        # 原始特征
        original_features = x
        
        # 合并卷积特征和原始特征
        lstm_input = torch.cat([conv_output, original_features], dim=2)
        
        # LSTM处理
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # 使用最后一个时间步的输出
        if self.lstm_bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        
        # 分类
        output = self.classifier(final_hidden)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "num_classes": self.num_classes
        }

# =============================================================================
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: wanglang_20250916_ConvM_LstmConfig, model: wanglang_20250916_ConvM_LstmModel, 
                 data_loader_manager: wanglang_20250916_ConvM_LstmDataLoaderManager, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.checkpoint_dir = checkpoint_dir
        
        # 设备设置
        self.device = setup_device(getattr(config, 'device', 'auto'))
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 训练组件
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 混合精度训练
        self.scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        
        # 训练状态
        self.best_val_score = 0.0
        self.train_history = []
        
        # Checkpoint跟踪器
        self.checkpoint_tracker = {
            'global_best': {'epoch': -1, 'val_loss': float('inf'), 'path': None},
            'early_best': {'epoch': -1, 'val_loss': float('inf'), 'path': None},
            'mid_best': {'epoch': -1, 'val_loss': float('inf'), 'path': None},
            'late_best': {'epoch': -1, 'val_loss': float('inf'), 'path': None}
        }
        
        # 移动模型到设备
        self.model.to(self.device)
        self._log_device_info()
        
        if self.use_amp:
            self.logger.info("✅ 混合精度训练已启用")
        
        # 记录数据分割信息
        data_info = self.data_loader_manager.get_data_info()
        self.logger.info(f"数据分割信息: {data_info['workflow_type']}")
    
    def _log_device_info(self):
        """记录设备和GPU使用信息"""
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            
            self.logger.info(f"🚀 使用GPU训练: {gpu_name}")
            self.logger.info(f"📊 GPU总内存: {gpu_memory:.1f}GB")
            self.logger.info(f"📊 模型占用内存: {current_memory:.2f}GB")
        else:
            self.logger.info("🖥️  使用CPU训练")
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"📊 模型总参数: {total_params:,}")
        self.logger.info(f"📊 可训练参数: {trainable_params:,}")
    
    def setup_training_components(self):
        """设置训练组件"""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def _save_checkpoint(self, epoch, val_loss, checkpoint_type):
        """保存checkpoint"""
        old_path = self.checkpoint_tracker[checkpoint_type]['path']
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
        
        checkpoint_name = f"checkpoint_{checkpoint_type}_epoch{epoch:03d}_val{val_loss:.4f}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'checkpoint_type': checkpoint_type
        }, checkpoint_path)
        
        self.checkpoint_tracker[checkpoint_type] = {
            'epoch': epoch,
            'val_loss': val_loss,
            'path': checkpoint_path
        }
        self.logger.info(f"💾 保存{checkpoint_type} checkpoint: epoch {epoch}, val_loss {val_loss:.4f}")
    
    def _save_interval_best(self, checkpoint_type, epoch_range):
        """在区间结束时，复制区间最佳checkpoint为interval_best"""
        source_path = self.checkpoint_tracker[checkpoint_type]['path']
        
        if source_path and os.path.exists(source_path):
            interval_name = f"interval_best_{epoch_range}.pt"
            interval_path = os.path.join(self.checkpoint_dir, interval_name)
            
            shutil.copy2(source_path, interval_path)
            
            epoch = self.checkpoint_tracker[checkpoint_type]['epoch']
            val_loss = self.checkpoint_tracker[checkpoint_type]['val_loss']
            
            self.logger.info(f"📦 复制区间最佳 [{epoch_range}]: epoch {epoch}, val_loss {val_loss:.4f}")
        else:
            self.logger.warning(f"⚠️ 区间 [{epoch_range}] 没有找到最佳checkpoint")
    
    def save_checkpoint_if_best(self, epoch, val_loss):
        """保存最佳checkpoint"""
        # 更新全局最佳
        if val_loss < self.checkpoint_tracker['global_best']['val_loss']:
            self._save_checkpoint(epoch, val_loss, 'global_best')
        
        # 更新区间最佳
        if 0 <= epoch < 30:
            if val_loss < self.checkpoint_tracker['early_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'early_best')
        elif 30 <= epoch < 60:
            if val_loss < self.checkpoint_tracker['mid_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'mid_best')
        elif 60 <= epoch < 100:
            if val_loss < self.checkpoint_tracker['late_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'late_best')
        
        # 在区间结束时复制区间最佳
        if epoch == 29:
            self._save_interval_best('early_best', '00-30')
        elif epoch == 59:
            self._save_interval_best('mid_best', '30-60')
        elif epoch == 99:
            self._save_interval_best('late_best', '60-100')
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if hasattr(self.config, 'gradient_clip_value'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if hasattr(self.config, 'gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
        
        metrics = {"loss": total_loss / len(train_loader)}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        
        # 计算AUC
        if len(np.unique(all_targets)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
            except Exception:
                metrics["auc"] = 0.0
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
        
        metrics = {"loss": total_loss / len(val_loader)}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
        
        # 计算AUC
        if len(np.unique(all_targets)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
                # 添加验证预测数据用于Optuna优化
                metrics["y_true_val"] = all_targets
                metrics["y_prob_val"] = all_probs_array[:, 1].tolist()
            except Exception:
                metrics["auc"] = 0.0
        
        return metrics
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """完整训练流程"""
        batch_size = batch_size or self.config.batch_size
        train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(batch_size=batch_size)
        
        self.setup_training_components()
        
        self.logger.info("开始训练...")
        self.logger.info(f"模型信息: {self.model.get_model_info()}")
        
        start_time = time.time()
        epochs = self.config.epochs
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader)
            
            # 记录历史
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "lr": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - epoch_start,
                "train_acc": train_metrics.get("accuracy", 0),
                "val_acc": val_metrics.get("accuracy", 0),
                "train_auc": train_metrics.get("auc", 0),
                "val_auc": val_metrics.get("auc", 0),
                "val_f1": val_metrics.get("f1", 0)
            }
            
            self.train_history.append(epoch_info)
            
            current_score = val_metrics.get("accuracy", 0)
            
            # 学习率调度
            self.scheduler.step(current_score)
            
            # 保存最佳模型
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                if not no_save_model:
                    self.save_checkpoint_if_best(epoch, val_metrics["loss"])
            
            # 日志输出
            log_msg = (
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
            )
            
            if 'auc' in val_metrics:
                log_msg += f", Val AUC: {val_metrics['auc']:.4f}"
            
            log_msg += f", Time: {epoch_info['time']:.2f}s"
            self.logger.info(log_msg)
        
        total_time = time.time() - start_time
        
        # 保存训练历史
        self.save_training_history()
        
        return {
            "best_val_score": self.best_val_score,
            "total_epochs": len(self.train_history),
            "total_time": total_time,
            "train_history": self.train_history
        }
    
    def save_training_history(self):
        """保存训练历史"""
        # 确保结果目录存在
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        self.logger.info(f"训练历史保存到: {history_path}")

# =============================================================================
# 推理器类
# =============================================================================

class UnifiedInferencer:
    """统一推理器"""
    
    def __init__(self, config: wanglang_20250916_ConvM_LstmConfig, model: wanglang_20250916_ConvM_LstmModel, 
                 data_loader_manager: wanglang_20250916_ConvM_LstmDataLoaderManager):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"成功加载模型检查点: {checkpoint_path}")
        
        return checkpoint
    
    def predict_batch(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """批量推理"""
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def evaluate(self) -> Dict[str, float]:
        """模型评估"""
        _, _, test_loader = self.data_loader_manager.load_data_loaders()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
        
        # 计算指标
        metrics = {}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        metrics["precision"] = precision_score(all_targets, all_preds, average='weighted')
        metrics["recall"] = recall_score(all_targets, all_preds, average='weighted')
        metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
        
        # AUC指标计算
        if len(np.unique(all_targets)) == 2:
            all_probs_positive = np.array(all_probs)[:, 1]
            metrics["auc"] = roc_auc_score(all_targets, all_probs_positive)
        
        return metrics
    
    def generate_predictions(self, start_date: str = None, end_date: str = None, 
                           output_path: str = None, output_format: str = "parquet") -> pd.DataFrame:
        """生成预测数据文件用于回测"""
        _, _, test_loader = self.data_loader_manager.load_data_loaders()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
        
        # 生成日期序列
        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date, periods=len(all_preds))
        else:
            date_range = pd.date_range(start='2020-01-01', periods=len(all_preds), freq='D')
        
        # 创建预测DataFrame
        all_probs_array = np.array(all_probs)
        predictions_df = pd.DataFrame({
            'date': date_range,
            'prediction': all_preds,
            'confidence': all_probs_array[:, 1] if all_probs_array.shape[1] > 1 else all_probs_array[:, 0],
            'probability_class_0': all_probs_array[:, 0] if all_probs_array.shape[1] > 1 else 1 - all_probs_array[:, 0],
            'probability_class_1': all_probs_array[:, 1] if all_probs_array.shape[1] > 1 else all_probs_array[:, 0],
            'model_name': self.config.model_name,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保存文件
        if output_path:
            if output_format == "parquet":
                predictions_df.to_parquet(output_path, index=False)
            else:
                predictions_df.to_csv(output_path, index=False)
            
            self.logger.info(f"预测文件已保存: {output_path}")
        
        return predictions_df

# =============================================================================
# 文档生成器
# =============================================================================

class ModelDocumentationGenerator:
    """模型文档生成器"""
    
    def __init__(self, config: wanglang_20250916_ConvM_LstmConfig, model: wanglang_20250916_ConvM_LstmModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个结合多分支卷积和LSTM的时间序列分类模型。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB
- **输入维度**: {model_info['input_dim']}
- **序列长度**: {model_info['seq_len']}

## 模型架构

### 核心组件

1. **多分支卷积层**: 使用不同kernel size提取多尺度特征
2. **LSTM层**: 捕获时间序列的长期依赖关系
3. **全连接层**: 进行最终分类

### 网络结构

- 卷积分支数: {self.config.conv_num_branches}
- LSTM隐藏层大小: {self.config.lstm_hidden_size}
- 分类器隐藏层: {self.config.fc_hidden_dim}

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}
- 权重衰减: {self.config.weight_decay}

### 数据配置
- 序列长度: {self.config.seq_len}
- 特征数量: {self.config.num_features}

## 使用方法

### 训练模型

```bash
python wanglang_20250916_ConvM_Lstm_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```bash
python wanglang_20250916_ConvM_Lstm_unified.py inference --checkpoint best_model.pth
```

### 生成文档

```bash
python wanglang_20250916_ConvM_Lstm_unified.py docs
```

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# 主要接口函数
# =============================================================================

def apply_optuna_config(config: wanglang_20250916_ConvM_LstmConfig, optuna_config_path: str):
    """应用Optuna配置文件中的超参数"""
    if not optuna_config_path or not os.path.exists(optuna_config_path):
        return config
    
    try:
        with open(optuna_config_path, 'r', encoding='utf-8') as f:
            optuna_config = yaml.safe_load(f)
        
        # 应用超参数覆盖
        if 'hyperparameters' in optuna_config:
            hyperparams = optuna_config['hyperparameters']
            for key, value in hyperparams.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"🔧 应用Optuna超参数: {key} = {value}")
                else:
                    print(f"⚠️ 忽略无效的Optuna超参数: {key} = {value}")
        
        return config
    except Exception as e:
        print(f"⚠️ 应用Optuna配置失败: {e}")
        return config

def main_train(config_path: str = "config.yaml", data_config_path: str = "data.yaml",
               optuna_config_path: str = None, device: str = "auto",
               epochs_override: int = None, no_save_model: bool = False, seed: int = 42,
               checkpoint_dir: str = "checkpoints"):
    """主训练函数"""
    # 设置随机种子
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置固定的日志路径
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/wanglang_20250916_ConvM_Lstm_2609327506"
    logger = setup_logging(log_dir=log_dir, log_filename="log.txt")
    
    logger.info(f"🎲 设置随机种子: {seed}")
    logger.info(f"💾 Checkpoint保存目录: {checkpoint_dir}")
    
    # 确保checkpoint目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except:
        config_dict = {}
    
    # 过滤掉配置类中不存在的参数
    import inspect
    valid_params = set(inspect.signature(wanglang_20250916_ConvM_LstmConfig.__init__).parameters.keys())
    valid_params.discard('self')  # 移除self参数
    
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_params}
    if len(filtered_config_dict) != len(config_dict):
        ignored_keys = set(config_dict.keys()) - set(filtered_config_dict.keys())
        logger.info(f"⚠️ 忽略配置文件中的无效参数: {ignored_keys}")
    
    # 创建配置对象
    config = wanglang_20250916_ConvM_LstmConfig(**filtered_config_dict)
    
    # 应用Optuna配置
    config = apply_optuna_config(config, optuna_config_path)
    
    # 覆盖配置
    if device != "auto":
        config.device = device
    if epochs_override:
        config.epochs = epochs_override
    
    config.checkpoint_dir = checkpoint_dir
    
    # 创建数据加载器管理器
    data_loader_manager = wanglang_20250916_ConvM_LstmDataLoaderManager(data_config_path, config)
    
    # 预加载数据以获取正确的维度信息
    try:
        train_loader, _, _ = data_loader_manager.load_data_loaders(batch_size=config.batch_size)
        logger.info(f"✅ 数据预加载完成，配置已更新")
        logger.info(f"📊 最终配置 - 特征维度: {config.input_dim}, 序列长度: {config.seq_len}")
    except Exception as e:
        logger.error(f"❌ 数据预加载失败: {e}")
        raise
    
    # 创建模型（现在配置已经是正确的）
    model = wanglang_20250916_ConvM_LstmModel(config)
    
    # 创建训练器
    trainer = UnifiedTrainer(config, model, data_loader_manager, checkpoint_dir=checkpoint_dir)
    
    # 执行训练
    results = trainer.train(no_save_model=no_save_model)
    
    # 输出JSON结果
    final_results = {
        "final_results": {
            "best_val_score": results['best_val_score'],
            "total_epochs": results['total_epochs'],
            "total_time": results['total_time'],
            "final_train_acc": results['train_history'][-1].get('train_acc', 0),
            "final_val_acc": results['train_history'][-1].get('val_acc', 0),
            "train_auc": results['train_history'][-1].get('train_auc', 0),
            "val_auc": results['train_history'][-1].get('val_auc', 0),
            "train_losses": [epoch['train_loss'] for epoch in results['train_history']],
            "val_losses": [epoch['val_loss'] for epoch in results['train_history']],
            "train_accuracies": [epoch.get('train_acc', 0) for epoch in results['train_history']],
            "val_accuracies": [epoch.get('val_acc', 0) for epoch in results['train_history']]
        }
    }
    print(json.dumps(final_results, indent=2))

def main_inference(config_path: str = "config.yaml", data_config_path: str = "data.yaml", 
                  checkpoint_path: str = "best_model.pth", mode: str = "eval",
                  start_date: str = None, end_date: str = None, 
                  output_path: str = None, output_format: str = "parquet"):
    """主推理函数"""
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except:
        config_dict = {}
    
    # 过滤掉配置类中不存在的参数
    import inspect
    valid_params = set(inspect.signature(wanglang_20250916_ConvM_LstmConfig.__init__).parameters.keys())
    valid_params.discard('self')  # 移除self参数
    
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_params}
    
    config = wanglang_20250916_ConvM_LstmConfig(**filtered_config_dict)
    
    # 创建数据加载器管理器
    data_loader_manager = wanglang_20250916_ConvM_LstmDataLoaderManager(data_config_path, config)
    
    # 创建模型
    model = wanglang_20250916_ConvM_LstmModel(config)
    
    # 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = inferencer.load_checkpoint(checkpoint_path)
    
    if mode == "eval":
        # 传统模型评估
        metrics = inferencer.evaluate()
        print(f"评估结果: {metrics}")
    elif mode == "test":
        # 生成预测数据文件
        predictions_df = inferencer.generate_predictions(
            start_date=start_date, 
            end_date=end_date, 
            output_path=output_path, 
            output_format=output_format
        )
        print(f"生成预测文件完成，包含 {len(predictions_df)} 条记录")

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except:
        config_dict = {}
    
    # 过滤掉配置类中不存在的参数
    import inspect
    valid_params = set(inspect.signature(wanglang_20250916_ConvM_LstmConfig.__init__).parameters.keys())
    valid_params.discard('self')  # 移除self参数
    
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_params}
    
    config = wanglang_20250916_ConvM_LstmConfig(**filtered_config_dict)
    
    # 创建模型
    model = wanglang_20250916_ConvM_LstmModel(config)
    
    # 生成文档
    doc_generator = ModelDocumentationGenerator(config, model)
    doc_generator.generate_model_md("MODEL.md")

# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="wanglang_20250916_ConvM_Lstm统一模型训练、推理和文档生成工具")
    parser.add_argument("mode", choices=["train", "inference", "docs"], help="运行模式")
    parser.add_argument("--config", default="config.yaml", help="模型配置文件路径")
    parser.add_argument("--data-config", default="data.yaml", help="数据配置文件路径")
    parser.add_argument("--checkpoint", default="best_model.pth", help="模型检查点路径")
    parser.add_argument("--output", help="输出文件路径")
    
    # Optuna优化相关参数
    parser.add_argument("--optuna-config", help="Optuna试验配置文件路径")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto", help="训练设备")
    parser.add_argument("--epochs", type=int, help="训练轮数（覆盖配置文件）")
    parser.add_argument("--no-save-model", action="store_true", help="不保存模型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确保可复现性）")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint保存目录")
    
    # 推理相关参数
    parser.add_argument("--inference-mode", choices=["test", "eval"], default="eval", 
                       help="推理模式：test(生成预测文件) 或 eval(评估)")
    parser.add_argument("--start-date", help="推理开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="推理结束日期 (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", 
                       help="输出格式")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "train":
        main_train(
            config_path=args.config,
            data_config_path=args.data_config,
            optuna_config_path=args.optuna_config,
            device=args.device,
            epochs_override=args.epochs,
            no_save_model=args.no_save_model,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir
        )
    elif args.mode == "inference":
        main_inference(
            config_path=args.config,
            data_config_path=args.data_config,
            checkpoint_path=args.checkpoint,
            mode=args.inference_mode,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
            output_format=args.format
        )
    elif args.mode == "docs":
        main_generate_docs(args.config, args.data_config)