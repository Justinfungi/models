#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wanglang_20250916_Conv_Trans_unified.py - 统一模型实现
结合卷积神经网络和Transformer的混合架构，支持Optuna超参数优化
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
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import math

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
class wanglang_20250916_Conv_TransConfig:
    """wanglang_20250916_Conv_Trans配置类"""
    # 模型基础配置
    model_name: str = "wanglang_20250916_Conv_Trans"
    model_type: str = "classification"
    architecture: str = "conv_transformer"  # 添加架构参数
    
    # 动态输入维度配置
    input_dim: Optional[int] = None  # 将从数据中自动检测
    seq_len: int = 100
    output_dim: int = 2
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-4
    
    # 性能优化配置
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    gradient_clip_value: float = 1.0
    
    # 卷积分支配置
    num_branches: int = 4
    kernel_sizes: List[List[int]] = None
    conv_out_channels: int = 32
    conv_activation: str = "relu"
    conv_batch_norm: bool = True
    conv_pooling: str = "adaptive_avg"
    
    # Transformer配置
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    transformer_activation: str = "relu"
    
    # 分类器配置
    hidden_dim: int = 256
    classifier_activation: str = "tanh"
    classifier_dropout: float = 0.3
    
    # 设备配置
    device: str = "auto"
    seed: int = 42
    
    # 路径配置
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [[1, 20], [3, 20], [5, 20], [7, 20]]

def set_all_seeds(seed: int = 42) -> None:
    """设置所有随机种子确保可重现性"""
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
    
    # 清除之前的handlers，避免重复日志
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

def create_directories(config: wanglang_20250916_Conv_TransConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 数据处理类
# =============================================================================

class wanglang_20250916_Conv_TransDataLoaderManager:
    """数据加载器管理类"""
    
    def __init__(self, data_config_path: str, config: wanglang_20250916_Conv_TransConfig):
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
    
    def get_data_split_info(self) -> Dict[str, Any]:
        """获取数据分割信息"""
        return self.data_split_info
    
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
            # 从data_split配置中获取文件路径
            train_file = self.data_split_info.get('train', {}).get('file')
            valid_file = self.data_split_info.get('validation', {}).get('file')
            test_file = self.data_split_info.get('test', {}).get('file')
            
            if not all([train_file, valid_file, test_file]):
                raise ValueError("数据配置中缺少文件路径信息")
            
            # 加载数据文件
            train_df = pd.read_parquet(train_file)
            valid_df = pd.read_parquet(valid_file)
            test_df = pd.read_parquet(test_file)
            
            print(f"📊 加载真实数据:")
            print(f"  训练集: {train_df.shape} - {train_file}")
            print(f"  验证集: {valid_df.shape} - {valid_file}")
            print(f"  测试集: {test_df.shape} - {test_file}")
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            print(f"❌ 加载真实数据失败: {e}")
            raise

    def get_input_dim_from_dataframe(self, df: pd.DataFrame) -> int:
        """从DataFrame获取特征维度"""
        # 提取特征列（包含@符号的列）
        feature_cols = [col for col in df.columns if '@' in col]
        if not feature_cols:
            # 如果没有@符号的列，使用除了最后一列的所有数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'class' in numeric_cols:
                numeric_cols.remove('class')
            feature_cols = numeric_cols[:-1] if len(numeric_cols) > 1 else numeric_cols
        
        input_dim = len(feature_cols)
        print(f"🔍 检测到输入特征维度: {input_dim}")
        return input_dim

    def load_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载数据加载器"""
        # 加载真实数据
        train_df, valid_df, test_df = self._load_real_dataframes()
        
        # 动态检测特征维度
        input_dim = self.get_input_dim_from_dataframe(train_df)
        self.config.input_dim = input_dim
        
        # 提取特征和标签
        feature_cols = [col for col in train_df.columns if '@' in col]
        if not feature_cols:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'class' in numeric_cols:
                numeric_cols.remove('class')
            feature_cols = numeric_cols[:-1] if len(numeric_cols) > 1 else numeric_cols
        
        target_col = 'class' if 'class' in train_df.columns else train_df.columns[-1]
        
        # 处理训练集
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_col].values.astype(np.int64)
        X_train = np.nan_to_num(X_train, nan=0.0)
        
        # 处理验证集
        X_valid = valid_df[feature_cols].values.astype(np.float32)
        y_valid = valid_df[target_col].values.astype(np.int64)
        X_valid = np.nan_to_num(X_valid, nan=0.0)
        
        # 处理测试集
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.int64)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_valid = self.scaler.transform(X_valid)
        X_test = self.scaler.transform(X_test)
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train)
        X_valid = torch.FloatTensor(X_valid)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_valid = torch.LongTensor(y_valid)
        y_test = torch.LongTensor(y_test)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        valid_dataset = TensorDataset(X_valid, y_valid)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 创建数据加载器
        train_loader = self._create_dataloader(train_dataset, shuffle=True, batch_size=batch_size)
        valid_loader = self._create_dataloader(valid_dataset, shuffle=False, batch_size=batch_size)
        test_loader = self._create_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        
        print(f"📈 数据加载器创建完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(valid_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, valid_loader, test_loader
    
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
# 模型架构组件
# =============================================================================

class ConvBranch(nn.Module):
    """卷积分支模块"""
    
    def __init__(self, 
                 input_dim: int,
                 kernel_size: List[int],
                 out_channels: int,
                 activation: str = "relu",
                 batch_norm: bool = True,
                 pooling: str = "adaptive_avg"):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=tuple(kernel_size),
            padding=(kernel_size[0]//2, kernel_size[1]//2)
        )
        
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        if pooling == "adaptive_avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "adaptive_max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        x = self.pooling(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        return x

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

# =============================================================================
# 模型基类和具体实现
# =============================================================================

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, config: wanglang_20250916_Conv_TransConfig):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 子类必须实现"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "input_dim": getattr(self.config, 'input_dim', None)
        }

class wanglang_20250916_Conv_TransModel(BaseModel):
    """wanglang_20250916_Conv_Trans混合模型架构"""
    
    def __init__(self, config: wanglang_20250916_Conv_TransConfig):
        super().__init__(config)
        
        # 输入维度处理
        self.input_dim = getattr(config, 'input_dim', None)
        if self.input_dim is None:
            print("⚠️ 配置中未指定input_dim，将在数据加载时自动检测")
        
        # 延迟初始化网络层，等待input_dim确定
        self.layers_initialized = False
        self.conv_branches = None
        self.input_projection = None
        self.pos_encoding = None
        self.transformer_encoder = None
        self.global_pool = None
        self.classifier = None
        
        # 保存配置用于延迟初始化
        self.seq_len = config.seq_len
        self.output_dim = config.output_dim
        self.num_branches = config.num_branches
        self.kernel_sizes = config.kernel_sizes
        self.conv_out_channels = config.conv_out_channels
        self.conv_activation = config.conv_activation
        self.conv_batch_norm = config.conv_batch_norm
        self.conv_pooling = config.conv_pooling
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.dim_feedforward = config.dim_feedforward
        self.transformer_dropout = config.transformer_dropout
        self.transformer_activation = config.transformer_activation
        self.hidden_dim = config.hidden_dim
        self.classifier_activation = config.classifier_activation
        self.classifier_dropout = config.classifier_dropout
        
        # 如果已知input_dim，立即初始化
        if self.input_dim is not None:
            self._build_layers()
    
    def _build_layers(self):
        """构建网络层"""
        if self.input_dim is None:
            raise ValueError("input_dim必须在构建网络层之前设置")
        
        print(f"🔧 构建网络层，输入维度: {self.input_dim}")
        
        # 创建卷积分支
        self.conv_branches = nn.ModuleList([
            ConvBranch(
                input_dim=self.input_dim,
                kernel_size=self.kernel_sizes[i] if i < len(self.kernel_sizes) else [3, 20],
                out_channels=self.conv_out_channels,
                activation=self.conv_activation,
                batch_norm=self.conv_batch_norm,
                pooling=self.conv_pooling
            ) for i in range(self.num_branches)
        ])
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.transformer_activation,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征融合维度
        conv_feature_dim = self.num_branches * self.conv_out_channels
        total_feature_dim = conv_feature_dim + self.d_model
        
        # 分类器
        if self.classifier_activation.lower() == 'tanh':
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.ReLU()
            
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, self.hidden_dim),
            activation_fn,
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        self.layers_initialized = True
        print(f"✅ 网络层构建完成")
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 首次前向传播时自动检测并构建网络
        if not self.layers_initialized:
            if self.input_dim is None:
                self.input_dim = x.shape[-1]
                print(f"🔍 自动检测输入维度: {self.input_dim}")
            self._build_layers()
        
        batch_size, input_dim = x.shape
        
        # 为序列处理添加序列维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 卷积分支处理
        conv_features = []
        
        # 为卷积准备输入: [batch_size, 1, seq_len, input_dim]
        conv_input = x.unsqueeze(1)  # [batch_size, 1, 1, input_dim]
        
        for branch in self.conv_branches:
            branch_output = branch(conv_input)
            conv_features.append(branch_output)
        
        # 拼接卷积特征
        conv_output = torch.cat(conv_features, dim=1)  # [batch_size, total_conv_features]
        
        # Transformer分支处理
        # 输入投影
        trans_input = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 转换为Transformer期望的格式: [seq_len, batch_size, d_model]
        trans_input = trans_input.transpose(0, 1)
        
        # 添加位置编码
        trans_input = self.pos_encoding(trans_input)
        
        # Transformer编码
        trans_output = self.transformer_encoder(trans_input)  # [seq_len, batch_size, d_model]
        
        # 转换回 [batch_size, seq_len, d_model]
        trans_output = trans_output.transpose(0, 1)
        
        # 全局池化: [batch_size, d_model, seq_len] -> [batch_size, d_model, 1]
        trans_output = trans_output.transpose(1, 2)
        trans_output = self.global_pool(trans_output)
        trans_output = trans_output.squeeze(-1)  # [batch_size, d_model]
        
        # 特征融合
        combined_features = torch.cat([conv_output, trans_output], dim=1)
        
        # 分类
        output = self.classifier(combined_features)
        
        return output

# =============================================================================
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: wanglang_20250916_Conv_TransConfig, model: BaseModel, 
                 data_loader_manager: wanglang_20250916_Conv_TransDataLoaderManager, 
                 checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.checkpoint_dir = checkpoint_dir
        
        # 设备设置
        self.device = setup_device(getattr(config, 'device', 'auto'))
        self.rank = 0
        self.world_size = 1
        
        # 设置日志
        self.logger = setup_logging(config.logs_dir, "training")
        
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
        
        # 设置随机种子
        set_all_seeds(config.seed)
        
        # 创建目录
        create_directories(config)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 移动模型到设备并记录GPU信息
        self.model.to(self.device)
        self._log_device_info()
        
        # 记录性能优化设置
        if self.use_amp:
            self.logger.info("✅ 混合精度训练已启用")
        
        # 记录数据分割信息
        data_info = self.data_loader_manager.get_data_info()
        self.logger.info(f"数据分割信息: {data_info['workflow_type']}")
        self.logger.info(f"训练期间: {data_info['train_period']}")
        self.logger.info(f"验证期间: {data_info['validation_period']}")
        self.logger.info(f"测试期间: {data_info['test_period']}")
    
    def _log_device_info(self):
        """记录设备和GPU使用信息"""
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            
            self.logger.info(f"🚀 使用GPU训练: {gpu_name}")
            self.logger.info(f"📊 GPU总内存: {gpu_memory:.1f}GB")
            self.logger.info(f"📊 模型占用内存: {current_memory:.2f}GB")
            self.logger.info(f"📊 可用内存: {gpu_memory - current_memory:.2f}GB")
        else:
            self.logger.info("🖥️  使用CPU训练")
            
        # 记录模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"📊 模型总参数: {total_params:,}")
        self.logger.info(f"📊 可训练参数: {trainable_params:,}")
        self.logger.info(f"📊 模型大小: {total_params * 4 / 1024**2:.1f}MB")
    
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
            
            # 混合精度训练
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
            
            # 计算预测和概率
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
        metrics = {"loss": total_loss / len(train_loader)}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        
        # 计算AUC
        try:
            if len(np.unique(all_targets)) == 2:
                all_probs_array = np.array(all_probs)
                if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
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
        try:
            if len(np.unique(all_targets)) == 2:
                all_probs_array = np.array(all_probs)
                if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                    metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
                    # 添加验证预测数据用于Optuna优化
                    metrics["y_true_val"] = all_targets
                    metrics["y_prob_val"] = all_probs_array[:, 1].tolist()
        except Exception:
            metrics["auc"] = 0.0
        
        return metrics
    
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
        
        # 在区间结束时，复制区间最佳为interval_best
        if epoch == 29:
            self._save_interval_best('early_best', '00-30')
        elif epoch == 59:
            self._save_interval_best('mid_best', '30-60')
        elif epoch == 99:
            self._save_interval_best('late_best', '60-100')
    
    def _save_checkpoint(self, epoch, val_loss, checkpoint_type):
        """保存checkpoint"""
        # 删除旧checkpoint
        old_path = self.checkpoint_tracker[checkpoint_type]['path']
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
        
        # 保存新checkpoint
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
            self.logger.info(f"   源文件: {os.path.basename(source_path)}")
            self.logger.info(f"   目标文件: {interval_name}")
        else:
            self.logger.warning(f"⚠️ 区间 [{epoch_range}] 没有找到最佳checkpoint")
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """完整训练流程"""
        # 获取数据加载器
        batch_size = batch_size or self.config.batch_size
        train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(
            batch_size=batch_size)
        
        # 确保模型已完全初始化（通过一次前向传播）
        if not self.model.layers_initialized:
            sample_batch = next(iter(train_loader))
            sample_data = sample_batch[0][:1]  # 取一个样本，先不移动到GPU
            with torch.no_grad():
                _ = self.model(sample_data)  # 触发模型初始化
            # 确保模型参数在正确的设备上
            self.model.to(self.device)
            print(f"🔧 模型初始化完成，参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
                "val_f1": val_metrics.get("f1", 0),
                "train_auc": train_metrics.get("auc", 0),
                "val_auc": val_metrics.get("auc", 0)
            }
            
            current_score = val_metrics.get("accuracy", 0)
            self.train_history.append(epoch_info)
            
            # 学习率调度
            self.scheduler.step(current_score)
            
            # 保存最佳模型
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                if not no_save_model:
                    best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_score': self.best_val_score,
                        'config': asdict(self.config)
                    }, best_path)
            
            # 保存checkpoint
            if not no_save_model:
                self.save_checkpoint_if_best(epoch, val_metrics["loss"])
            
            # GPU内存监控
            if self.device.type == 'cuda' and (epoch + 1) % 10 == 0:
                current_memory = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                cached_memory = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"📊 GPU内存使用 (Epoch {epoch+1}): 当前 {current_memory:.2f}GB, 峰值 {max_memory:.2f}GB, 缓存 {cached_memory:.2f}GB")
            
            # 日志输出
            log_msg = (
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
            )

            if 'auc' in val_metrics and val_metrics['auc'] > 0:
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
        history_path = Path(self.config.results_dir) / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        self.logger.info(f"训练历史保存到: {history_path}")

# =============================================================================
# 推理器类
# =============================================================================

class UnifiedInferencer:
    """统一推理器"""
    
    def __init__(self, config: wanglang_20250916_Conv_TransConfig, model: BaseModel, 
                 data_loader_manager: wanglang_20250916_Conv_TransDataLoaderManager):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.device = setup_device()
        self.logger = setup_logging(config.logs_dir, "inference")
        
        # 移动模型到设备
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
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def predict_single(self, data: np.ndarray) -> Tuple[Union[int, float], np.ndarray]:
        """单样本推理"""
        data_tensor = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(data_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            return pred, probs.cpu().numpy()[0]
    
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
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
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
        elif len(np.unique(all_targets)) > 2:
            try:
                metrics["auc"] = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
            except ValueError:
                pass
        
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
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # 创建预测DataFrame
        predictions = np.array(all_preds)
        probabilities = np.array(all_probs)
        
        # 生成日期序列（如果指定了日期范围）
        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date, periods=len(predictions))
        else:
            # 使用默认日期序列
            date_range = pd.date_range(start='2023-01-01', periods=len(predictions), freq='D')
        
        predictions_df = pd.DataFrame({
            'date': date_range,
            'prediction': predictions,
            'confidence': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0],
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1] if probabilities.shape[1] > 1 else 1 - probabilities[:, 0],
            'model_name': self.config.model_name,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保存预测文件
        if output_path:
            if output_format == "parquet":
                predictions_df.to_parquet(output_path, index=False)
            else:
                predictions_df.to_csv(output_path, index=False)
            self.logger.info(f"预测结果保存到: {output_path}")
        
        return predictions_df
    
    def save_predictions(self, predictions: np.ndarray, probabilities: np.ndarray, 
                        output_path: str):
        """保存预测结果"""
        results = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"预测结果保存到: {output_path}")

# =============================================================================
# 文档生成器
# =============================================================================

class ModelDocumentationGenerator:
    """模型文档生成器"""
    
    def __init__(self, config: wanglang_20250916_Conv_TransConfig, model: BaseModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个结合卷积神经网络和Transformer的混合架构模型，专门用于时间序列分类任务。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB
- **输入维度**: {model_info.get('input_dim', 'Dynamic')}

## 模型架构

### 核心组件

1. **卷积分支**: {self.config.num_branches}个并行卷积分支，使用不同的卷积核大小
2. **Transformer编码器**: {self.config.num_layers}层Transformer编码器
3. **特征融合**: 卷积特征和Transformer特征的融合
4. **分类器**: 全连接分类器

### 网络结构

```
输入 -> [卷积分支1, 卷积分支2, ..., 卷积分支N] -> 卷积特征
     -> Transformer编码器 -> Transformer特征
     -> 特征融合 -> 分类器 -> 输出
```

## 技术原理

### 卷积分支
- 使用多个不同尺寸的卷积核捕获不同时间尺度的特征
- 每个分支包含卷积层、批归一化、激活函数和自适应池化

### Transformer编码器
- 使用位置编码处理序列信息
- 多头自注意力机制捕获长距离依赖关系

### 特征融合
- 将卷积特征和Transformer特征拼接
- 通过全连接层进行最终分类

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}
- 权重衰减: {self.config.weight_decay}

### 模型配置
- 卷积分支数: {self.config.num_branches}
- Transformer维度: {self.config.d_model}
- 注意力头数: {self.config.nhead}
- Transformer层数: {self.config.num_layers}

## 使用方法

### 训练模型

```bash
python wanglang_20250916_Conv_Trans_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```bash
python wanglang_20250916_Conv_Trans_unified.py inference --checkpoint best_model.pth
```

### 生成预测文件

```bash
python wanglang_20250916_Conv_Trans_unified.py inference --inference-mode test --start-date 2023-01-01 --end-date 2023-12-31
```

## 性能特性

- 支持混合精度训练
- 自动GPU检测和使用
- 动态输入维度适配
- 多checkpoint保存策略
- Optuna超参数优化支持

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# 主要接口函数
# =============================================================================

def apply_optuna_config(config: wanglang_20250916_Conv_TransConfig, optuna_config_path: str) -> None:
    """应用Optuna配置文件中的超参数"""
    if not optuna_config_path or not os.path.exists(optuna_config_path):
        return
    
    try:
        with open(optuna_config_path, 'r', encoding='utf-8') as f:
            optuna_config = json.load(f)
        
        # 获取配置类的有效字段
        from dataclasses import fields
        valid_fields = {field.name for field in fields(wanglang_20250916_Conv_TransConfig)}
        
        # 应用超参数覆盖，只应用有效的参数
        for key, value in optuna_config.items():
            if key in valid_fields and hasattr(config, key):
                setattr(config, key, value)
                print(f"🔧 Optuna覆盖参数: {key} = {value}")
            elif key not in valid_fields:
                print(f"⚠️ 忽略不支持的Optuna参数: {key}")
    except Exception as e:
        print(f"⚠️ 加载Optuna配置失败: {e}")

def main_train(config_path: str = "config.yaml", data_config_path: str = "data.yaml",
               optuna_config_path: str = None, device: str = "auto",
               epochs_override: int = None, no_save_model: bool = False, seed: int = 42,
               checkpoint_dir: str = "checkpoints"):
    """主训练函数"""
    # 设置所有随机种子以确保可复现性
    set_all_seeds(seed)
    
    # 设置固定的日志路径
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/wanglang_20250916_Conv_Trans_3658411800"
    logger = setup_logging(log_dir=log_dir, log_filename="log.txt")
    
    logger.info(f"🎲 设置随机种子: {seed}")
    logger.info(f"💾 Checkpoint保存目录: {checkpoint_dir}")
    
    # 确保checkpoint目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        # 创建默认配置
        config_dict = {}
    
    # 处理architecture部分
    if config_dict and 'architecture' in config_dict:
        arch = config_dict['architecture']
        # 将architecture中的参数映射到配置类
        architecture_mapping = {
            # 卷积分支配置
            'conv_branches': {
                'num_branches': 'num_branches',
                'kernel_sizes': 'kernel_sizes', 
                'out_channels': 'conv_out_channels',
                'activation': 'conv_activation',
                'batch_norm': 'conv_batch_norm',
                'pooling': 'conv_pooling'
            },
            # Transformer配置
            'transformer': {
                'd_model': 'd_model',
                'nhead': 'nhead',
                'num_layers': 'num_layers',
                'dim_feedforward': 'dim_feedforward',
                'dropout': 'transformer_dropout',
                'activation': 'transformer_activation'
            },
            # 分类器配置
            'classifier': {
                'hidden_dim': 'hidden_dim',
                'activation': 'classifier_activation',
                'dropout': 'classifier_dropout'
            }
        }
        
        # 提取architecture参数到顶层
        for section_name, section_mapping in architecture_mapping.items():
            if section_name in arch:
                section_config = arch[section_name]
                for arch_key, config_key in section_mapping.items():
                    if arch_key in section_config:
                        config_dict[config_key] = section_config[arch_key]
        
        # 处理其他直接映射的参数
        direct_mappings = {
            'input_dim': 'input_dim',
            'output_dim': 'output_dim',
            'model_name': 'model_name',
            'model_type': 'model_type'
        }
        
        for arch_key, config_key in direct_mappings.items():
            if arch_key in arch:
                config_dict[config_key] = arch[arch_key]
        
        # 移除architecture部分
        del config_dict['architecture']
    
    # 过滤配置字典，只保留配置类支持的参数
    if config_dict:
        # 获取配置类的有效字段
        from dataclasses import fields
        valid_fields = {field.name for field in fields(wanglang_20250916_Conv_TransConfig)}
        
        # 过滤掉不支持的参数
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # 记录被过滤掉的参数
        filtered_out = {k: v for k, v in config_dict.items() if k not in valid_fields}
        if filtered_out:
            logger.info(f"⚠️ 过滤掉不支持的配置参数: {list(filtered_out.keys())}")
        
        config = wanglang_20250916_Conv_TransConfig(**filtered_config)
    else:
        config = wanglang_20250916_Conv_TransConfig()
    
    # 应用Optuna配置
    if optuna_config_path:
        apply_optuna_config(config, optuna_config_path)
    
    # 覆盖配置
    if device != "auto":
        config.device = device
    if epochs_override:
        config.epochs = epochs_override
    
    # 创建数据加载器管理器
    data_loader_manager = wanglang_20250916_Conv_TransDataLoaderManager(data_config_path, config)
    
    # 创建模型
    model = wanglang_20250916_Conv_TransModel(config)
    
    # 创建训练器
    trainer = UnifiedTrainer(config, model, data_loader_manager, checkpoint_dir=checkpoint_dir)
    
    # 执行训练
    results = trainer.train(no_save_model=no_save_model)
    
    # 输出JSON格式的训练结果
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
    except FileNotFoundError:
        config_dict = {}
    
    # 处理architecture部分
    if config_dict and 'architecture' in config_dict:
        arch = config_dict['architecture']
        # 将architecture中的参数映射到配置类
        architecture_mapping = {
            # 卷积分支配置
            'conv_branches': {
                'num_branches': 'num_branches',
                'kernel_sizes': 'kernel_sizes', 
                'out_channels': 'conv_out_channels',
                'activation': 'conv_activation',
                'batch_norm': 'conv_batch_norm',
                'pooling': 'conv_pooling'
            },
            # Transformer配置
            'transformer': {
                'd_model': 'd_model',
                'nhead': 'nhead',
                'num_layers': 'num_layers',
                'dim_feedforward': 'dim_feedforward',
                'dropout': 'transformer_dropout',
                'activation': 'transformer_activation'
            },
            # 分类器配置
            'classifier': {
                'hidden_dim': 'hidden_dim',
                'activation': 'classifier_activation',
                'dropout': 'classifier_dropout'
            }
        }
        
        # 提取architecture参数到顶层
        for section_name, section_mapping in architecture_mapping.items():
            if section_name in arch:
                section_config = arch[section_name]
                for arch_key, config_key in section_mapping.items():
                    if arch_key in section_config:
                        config_dict[config_key] = section_config[arch_key]
        
        # 处理其他直接映射的参数
        direct_mappings = {
            'input_dim': 'input_dim',
            'output_dim': 'output_dim',
            'model_name': 'model_name',
            'model_type': 'model_type'
        }
        
        for arch_key, config_key in direct_mappings.items():
            if arch_key in arch:
                config_dict[config_key] = arch[arch_key]
        
        # 移除architecture部分
        del config_dict['architecture']
    
    # 过滤配置字典，只保留配置类支持的参数
    if config_dict:
        from dataclasses import fields
        valid_fields = {field.name for field in fields(wanglang_20250916_Conv_TransConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = wanglang_20250916_Conv_TransConfig(**filtered_config)
    else:
        config = wanglang_20250916_Conv_TransConfig()
    
    # 创建数据加载器管理器
    data_loader_manager = wanglang_20250916_Conv_TransDataLoaderManager(data_config_path, config)
    
    # 创建模型
    model = wanglang_20250916_Conv_TransModel(config)
    
    # 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = inferencer.load_checkpoint(checkpoint_path)
    
    if mode == "eval":
        # 传统模型评估
        metrics = inferencer.evaluate()
        print("评估结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    elif mode == "test":
        # 生成预测数据文件
        predictions_df = inferencer.generate_predictions(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            output_format=output_format
        )
        print(f"生成预测文件: {len(predictions_df)} 条记录")

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        config_dict = {}
    
    # 处理architecture部分
    if config_dict and 'architecture' in config_dict:
        arch = config_dict['architecture']
        # 将architecture中的参数映射到配置类
        architecture_mapping = {
            # 卷积分支配置
            'conv_branches': {
                'num_branches': 'num_branches',
                'kernel_sizes': 'kernel_sizes', 
                'out_channels': 'conv_out_channels',
                'activation': 'conv_activation',
                'batch_norm': 'conv_batch_norm',
                'pooling': 'conv_pooling'
            },
            # Transformer配置
            'transformer': {
                'd_model': 'd_model',
                'nhead': 'nhead',
                'num_layers': 'num_layers',
                'dim_feedforward': 'dim_feedforward',
                'dropout': 'transformer_dropout',
                'activation': 'transformer_activation'
            },
            # 分类器配置
            'classifier': {
                'hidden_dim': 'hidden_dim',
                'activation': 'classifier_activation',
                'dropout': 'classifier_dropout'
            }
        }
        
        # 提取architecture参数到顶层
        for section_name, section_mapping in architecture_mapping.items():
            if section_name in arch:
                section_config = arch[section_name]
                for arch_key, config_key in section_mapping.items():
                    if arch_key in section_config:
                        config_dict[config_key] = section_config[arch_key]
        
        # 处理其他直接映射的参数
        direct_mappings = {
            'input_dim': 'input_dim',
            'output_dim': 'output_dim',
            'model_name': 'model_name',
            'model_type': 'model_type'
        }
        
        for arch_key, config_key in direct_mappings.items():
            if arch_key in arch:
                config_dict[config_key] = arch[arch_key]
        
        # 移除architecture部分
        del config_dict['architecture']
    
    # 过滤配置字典，只保留配置类支持的参数
    if config_dict:
        from dataclasses import fields
        valid_fields = {field.name for field in fields(wanglang_20250916_Conv_TransConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = wanglang_20250916_Conv_TransConfig(**filtered_config)
    else:
        config = wanglang_20250916_Conv_TransConfig()
    
    # 创建模型
    model = wanglang_20250916_Conv_TransModel(config)
    
    # 生成文档
    doc_generator = ModelDocumentationGenerator(config, model)
    doc_generator.generate_model_md("MODEL.md")

# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="wanglang_20250916_Conv_Trans统一模型训练、推理和文档生成工具")
    parser.add_argument("mode", choices=["train", "inference", "docs"], help="运行模式")
    parser.add_argument("--config", default="config.yaml", help="模型配置文件路径")
    parser.add_argument("--data-config", default="data.yaml", help="数据配置文件路径")
    parser.add_argument("--checkpoint", default="best_model.pth", help="模型检查点路径")
    parser.add_argument("--data", help="数据文件路径")
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