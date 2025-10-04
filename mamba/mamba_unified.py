#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mamba_unified.py - 统一Mamba模型实现
基于选择性状态空间模型的高效序列处理架构，支持Optuna超参数优化
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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# 基础配置和工具函数
# =============================================================================

@dataclass
class MambaConfig:
    """Mamba模型配置类"""
    # 模型基础配置
    model_name: str = "mamba"
    model_type: str = "classification"
    
    # Mamba特有配置
    d_model: int = 256
    n_layer: int = 8
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    
    # 选择性扫描配置
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    
    # 训练配置
    learning_rate: float = 0.0003
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-5
    
    # 性能优化配置
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    gradient_clip_value: float = 1.0
    
    # 正则化
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    
    # 初始化配置
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True
    
    # 任务配置
    num_classes: int = 2
    input_dim: Optional[int] = None
    
    # 设备配置
    device: str = "auto"
    seed: int = 42
    
    # 路径配置
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        """配置后处理"""
        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)
        
        # 计算内部维度
        self.d_inner = int(self.expand * self.d_model)

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
    import logging
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

def create_directories(config: MambaConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Mamba模型核心组件
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps) if eps is not None else 1e-5
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    """Mamba核心块：实现选择性状态空间模型"""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        
        # 1D卷积层
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        
        # SSM参数投影
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # 状态空间参数
        A_log = torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(config.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # 输入投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # 1D卷积
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        
        # 激活函数
        x = F.silu(x)
        
        # SSM计算
        y = self.ssm(x)
        
        # 门控机制
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """选择性状态空间模型核心计算"""
        batch_size, seq_len, d_inner = x.shape
        
        # 计算delta, B, C参数
        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        
        # 计算delta
        delta = F.softplus(self.dt_proj(delta))
        
        # 获取A矩阵
        A = -torch.exp(self.A_log.float())
        
        # 选择性扫描算法
        y = self.selective_scan(x, delta, A, B, C, self.D)
        
        return y
    
    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, 
                      B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """选择性扫描算法实现"""
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[-1]
        
        # 离散化
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        
        # 状态更新
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        # 添加跳跃连接
        y = y + u * D
        
        return y

class MambaLayer(nn.Module):
    """完整的Mamba层，包含残差连接和层归一化"""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.mamba = MambaBlock(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + x

# =============================================================================
# 数据处理类
# =============================================================================

class MambaDataLoaderManager:
    """Mamba数据加载器管理类"""
    
    def __init__(self, data_config_path: str, config: MambaConfig):
        self.data_config_path = data_config_path
        self.config = config
        self.data_config = self._load_data_config()
        self.data_split_info = self.data_config.get('data_split', {})
        self.scaler = None
        
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
            # 获取数据文件路径
            train_file = self.data_split_info['train']['file']
            valid_file = self.data_split_info['validation']['file']
            test_file = self.data_split_info['test']['file']
            
            # 获取数据文件夹路径
            data_folder = self.data_config.get('data_paths', {}).get('data_folder', '')
            
            # 构建完整的文件路径
            if not os.path.isabs(train_file):
                train_file = os.path.join(data_folder, train_file)
            if not os.path.isabs(valid_file):
                valid_file = os.path.join(data_folder, valid_file)
            if not os.path.isabs(test_file):
                test_file = os.path.join(data_folder, test_file)
            
            print(f"📊 加载训练数据: {train_file}")
            train_df = pd.read_parquet(train_file)
            
            print(f"📊 加载验证数据: {valid_file}")
            valid_df = pd.read_parquet(valid_file)
            
            print(f"📊 加载测试数据: {test_file}")
            test_df = pd.read_parquet(test_file)
            
            print(f"✅ 数据加载完成")
            print(f"   训练集: {train_df.shape}")
            print(f"   验证集: {valid_df.shape}")
            print(f"   测试集: {test_df.shape}")
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            raise ValueError(f"加载真实数据失败: {e}")
    
    def get_input_dim_from_dataframe(self, df: pd.DataFrame) -> int:
        """从DataFrame获取特征维度"""
        # 自动识别特征列（包含@符号的列）
        feature_columns = [col for col in df.columns if '@' in col]
        if not feature_columns:
            # 如果没有@符号列，假设最后一列是标签，其余都是特征
            features = df.shape[1] - 1
        else:
            features = len(feature_columns)
        
        print(f"🔍 检测到输入特征维度: {features}")
        return features
    
    def validate_input_dimensions(self, config: MambaConfig, actual_input_dim: int) -> int:
        """验证配置文件中的输入维度与实际数据是否一致"""
        config_input_dim = getattr(config, 'input_dim', None)
        
        if config_input_dim is None:
            print(f"📋 配置文件中未指定input_dim，使用实际数据维度: {actual_input_dim}")
            config.input_dim = actual_input_dim
            return actual_input_dim
        
        if config_input_dim != actual_input_dim:
            print(f"⚠️ 维度不匹配！配置: {config_input_dim}, 实际: {actual_input_dim}")
            print(f"🔧 使用实际数据维度: {actual_input_dim}")
            config.input_dim = actual_input_dim
            return actual_input_dim
        
        print(f"✅ 输入维度验证通过: {actual_input_dim}")
        return actual_input_dim
    
    def load_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载真实数据并创建DataLoader"""
        # 加载真实数据
        train_df, valid_df, test_df = self._load_real_dataframes()
        
        # 动态检测特征维度
        input_dim = self.get_input_dim_from_dataframe(train_df)
        
        # 验证并更新配置
        self.validate_input_dimensions(self.config, input_dim)
        
        # 提取特征和标签
        feature_columns = [col for col in train_df.columns if '@' in col]
        if not feature_columns:
            feature_columns = train_df.columns[:-1].tolist()
        
        label_column = train_df.columns[-1]
        
        # 准备数据
        X_train = train_df[feature_columns].values.astype(np.float32)
        y_train = train_df[label_column].values.astype(np.int64)
        
        X_valid = valid_df[feature_columns].values.astype(np.float32)
        y_valid = valid_df[label_column].values.astype(np.int64)
        
        X_test = test_df[feature_columns].values.astype(np.float32)
        y_test = test_df[label_column].values.astype(np.int64)
        
        # 处理缺失值
        if np.isnan(X_train).any():
            print("⚠️ 检测到缺失值，使用均值填充")
            X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
            X_valid = np.nan_to_num(X_valid, nan=np.nanmean(X_valid))
            X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))
        
        # 标准化
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_valid = self.scaler.transform(X_valid)
        X_test = self.scaler.transform(X_test)
        
        # 保存scaler
        scaler_path = Path(self.config.checkpoint_dir) / 'scaler.pkl'
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Scaler已保存到: {scaler_path}")
        
        # 创建数据集
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        valid_dataset = TensorDataset(torch.FloatTensor(X_valid), torch.LongTensor(y_valid))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        # 创建DataLoader
        train_loader = self._create_dataloader(train_dataset, shuffle=True, batch_size=batch_size)
        valid_loader = self._create_dataloader(valid_dataset, shuffle=False, batch_size=batch_size)
        test_loader = self._create_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        
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
# 模型基类
# =============================================================================

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, config: MambaConfig):
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
            "model_size_mb": total_params * 4 / (1024 * 1024)
        }

class MambaModel(BaseModel):
    """完整的Mamba模型实现"""
    
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config = config
        
        # 输入维度动态配置
        self.input_dim = getattr(config, 'input_dim', None)
        
        if self.input_dim is None:
            print("⚠️ 配置中未指定input_dim，将在数据加载时自动检测")
        
        # 延迟初始化网络层
        self.layers = None
        self.feature_projection = None
        self.norm_f = None
        self.classifier = None
        
    def _build_layers(self, input_dim: int):
        """使用动态输入维度构建网络"""
        print(f"🔧 构建网络层，输入维度: {input_dim}")
        
        # 获取当前设备
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征投影层
        self.feature_projection = nn.Linear(input_dim, self.config.d_model).to(device)
        
        # Mamba层堆叠
        self.layers = nn.ModuleList([
            MambaLayer(self.config) for _ in range(self.config.n_layer)
        ]).to(device)
        
        # 最终层归一化
        self.norm_f = RMSNorm(self.config.d_model, eps=self.config.layer_norm_epsilon).to(device)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_classes)
        ).to(device)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 移动新创建的层到正确的设备
        # 检查是否有其他参数来确定设备
        existing_params = [p for name, p in self.named_parameters() if not any(layer_name in name for layer_name in ['feature_projection', 'layers', 'norm_f', 'classifier'])]
        if existing_params:
            device = existing_params[0].device
            self.to(device)
    
    def _init_weights(self, module: nn.Module) -> None:
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 首次前向传播时自动检测并构建网络
        if self.layers is None:
            if self.input_dim is None:
                self.input_dim = x.shape[-1]
                print(f"🔍 自动检测输入维度: {self.input_dim}")
            self._build_layers(self.input_dim)
        
        # 特征投影
        if x.dim() == 2:
            x = self.feature_projection(x)
            x = x.unsqueeze(1)  # 添加序列维度
        
        # 确保输入是3D张量
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 通过Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 最终归一化
        x = self.norm_f(x)
        
        # 序列池化
        if x.shape[1] > 1:
            x = x.mean(dim=1)
        else:
            x = x.squeeze(1)
        
        # 分类
        logits = self.classifier(x)
        
        return logits

# =============================================================================
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: MambaConfig, model: BaseModel, data_loader_manager: MambaDataLoaderManager,
                 checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.checkpoint_dir = checkpoint_dir  # Checkpoint保存目录
        
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
        
        # Checkpoint跟踪器（支持多区间最佳）
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
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-5
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
            
            # 收集预测结果
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        metrics = {"loss": total_loss / len(train_loader)}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        
        # 计算AUC
        if len(np.unique(all_targets)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                    metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
            except Exception:
                pass
        
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
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        metrics = {"loss": total_loss / len(val_loader)}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
        
        # 计算AUC
        if len(np.unique(all_targets)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                    metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
                    # 添加验证预测数据用于Optuna优化
                    metrics["y_true_val"] = all_targets
                    metrics["y_prob_val"] = all_probs_array[:, 1].tolist()
            except Exception:
                pass
        
        return metrics
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """完整训练流程"""
        # 获取数据加载器
        batch_size = batch_size or self.config.batch_size
        train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(batch_size=batch_size)
        
        # 初始化模型（通过一次前向传播）
        sample_batch = next(iter(train_loader))
        sample_data = sample_batch[0][:1].to(self.device)  # 取一个样本
        with torch.no_grad():
            _ = self.model(sample_data)  # 触发模型初始化
        
        # 确保模型在正确的设备上
        self.model.to(self.device)
        
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
                "time": time.time() - epoch_start
            }
            
            epoch_info.update({
                "train_acc": train_metrics.get("accuracy", 0),
                "val_acc": val_metrics.get("accuracy", 0),
                "val_f1": val_metrics.get("f1", 0)
            })
            
            if "auc" in train_metrics:
                epoch_info["train_auc"] = train_metrics["auc"]
            if "auc" in val_metrics:
                epoch_info["val_auc"] = val_metrics["auc"]
            
            current_score = val_metrics.get("accuracy", 0)
            
            self.train_history.append(epoch_info)
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                if not no_save_model:
                    self.save_checkpoint(epoch + 1, is_best=True)
            
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'config': asdict(self.config)
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
    
    def save_checkpoint_if_best(self, epoch: int, val_loss: float):
        """智能checkpoint保存：全局最佳 + 区间最佳"""
        # 更新全局最佳
        if val_loss < self.checkpoint_tracker['global_best']['val_loss']:
            self._save_checkpoint(epoch, val_loss, 'global_best')
        
        # 更新区间最佳（实时更新）
        if 0 <= epoch < 30:
            if val_loss < self.checkpoint_tracker['early_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'early_best')
        elif 30 <= epoch < 60:
            if val_loss < self.checkpoint_tracker['mid_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'mid_best')
        elif 60 <= epoch < 100:
            if val_loss < self.checkpoint_tracker['late_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'late_best')
        
        # 💡 关键：在区间结束时，复制区间最佳为interval_best
        if epoch == 29:  # epoch从0开始，29是第30个epoch
            self._save_interval_best('early_best', '00-30')
        elif epoch == 59:
            self._save_interval_best('mid_best', '30-60')
        elif epoch == 99:
            self._save_interval_best('late_best', '60-100')
    
    def _save_checkpoint(self, epoch: int, val_loss: float, checkpoint_type: str):
        """保存单个checkpoint"""
        import os
        
        # 删除旧checkpoint
        old_path = self.checkpoint_tracker[checkpoint_type]['path']
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
        
        # 保存新checkpoint
        checkpoint_name = f"checkpoint_{checkpoint_type}_epoch{epoch:03d}_val{val_loss:.4f}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'checkpoint_type': checkpoint_type,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoint_tracker[checkpoint_type] = {
            'epoch': epoch,
            'val_loss': val_loss,
            'path': checkpoint_path
        }
        
        self.logger.info(f"💾 保存{checkpoint_type} checkpoint: epoch {epoch}, val_loss {val_loss:.4f}")
    
    def _save_interval_best(self, checkpoint_type: str, epoch_range: str):
        """在区间结束时，复制区间最佳checkpoint为interval_best"""
        import shutil
        import os
        
        # 获取当前区间最佳的checkpoint路径
        source_path = self.checkpoint_tracker[checkpoint_type]['path']
        
        if source_path and os.path.exists(source_path):
            # 创建interval_best文件名
            interval_name = f"interval_best_{epoch_range}.pt"
            interval_path = os.path.join(self.checkpoint_dir, interval_name)
            
            # 复制checkpoint
            shutil.copy2(source_path, interval_path)
            
            epoch = self.checkpoint_tracker[checkpoint_type]['epoch']
            val_loss = self.checkpoint_tracker[checkpoint_type]['val_loss']
            
            self.logger.info(f"📦 复制区间最佳 [{epoch_range}]: epoch {epoch}, val_loss {val_loss:.4f}")
            self.logger.info(f"   源文件: {os.path.basename(source_path)}")
            self.logger.info(f"   目标文件: {interval_name}")
        else:
            self.logger.warning(f"⚠️ 区间 [{epoch_range}] 没有找到最佳checkpoint")
    
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
    
    def __init__(self, config: MambaConfig, model: BaseModel, data_loader_manager: MambaDataLoaderManager,
                 checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.checkpoint_dir = checkpoint_dir  # Checkpoint保存目录
        self.device = setup_device()
        self.logger = setup_logging(config.logs_dir, "inference")
        
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
                all_preds.extend(preds.cpu().numpy())
        
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
        
        return metrics

# =============================================================================
# 文档生成器
# =============================================================================

class ModelDocumentationGenerator:
    """模型文档生成器"""
    
    def __init__(self, config: MambaConfig, model: BaseModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个基于选择性状态空间模型(Mamba)的{self.config.model_type}模型。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB

## 模型架构

### Mamba核心组件

- **选择性状态空间模型**: 高效的序列处理架构
- **RMSNorm**: Root Mean Square Layer Normalization
- **门控机制**: 选择性信息传递
- **1D卷积**: 局部特征提取

### 网络结构

```
输入 -> 特征投影 -> Mamba层堆叠 -> 归一化 -> 分类头 -> 输出
```

## 技术原理

Mamba模型基于选择性状态空间模型，通过选择性扫描算法实现高效的序列建模。

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}
- 权重衰减: {self.config.weight_decay}

### 模型配置
- 模型维度: {self.config.d_model}
- 层数: {self.config.n_layer}
- 状态维度: {self.config.d_state}
- 卷积核大小: {self.config.d_conv}

## 使用方法

### 训练模型

```bash
python mamba_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```bash
python mamba_unified.py inference --config config.yaml --data-config data.yaml --checkpoint best_model.pth
```

### 模型评估

```bash
python mamba_unified.py inference --config config.yaml --data-config data.yaml
```

## 性能指标

模型支持以下评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- AUC (Area Under Curve)

## 注意事项

- 模型支持动态输入维度检测
- 支持混合精度训练以提升性能
- 支持GPU加速训练和推理

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# Checkpoint管理方法（添加到UnifiedTrainer类中）
# =============================================================================

# 注意：以下方法应该添加到UnifiedTrainer类中，这里作为参考

def _add_checkpoint_methods_to_trainer():
    """
    将以下方法添加到UnifiedTrainer类中：
    
    def save_checkpoint_if_best(self, epoch: int, val_loss: float):
        '''智能checkpoint保存：全局最佳 + 区间最佳'''
        if val_loss < self.checkpoint_tracker['global_best']['val_loss']:
            self._save_checkpoint(epoch, val_loss, 'global_best')
        
        if 0 <= epoch < 30:
            if val_loss < self.checkpoint_tracker['early_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'early_best')
        elif 30 <= epoch < 60:
            if val_loss < self.checkpoint_tracker['mid_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'mid_best')
        elif 60 <= epoch < 100:
            if val_loss < self.checkpoint_tracker['late_best']['val_loss']:
                self._save_checkpoint(epoch, val_loss, 'late_best')
        
        if epoch == 29:
            self._save_interval_best('early_best', '00-30')
        elif epoch == 59:
            self._save_interval_best('mid_best', '30-60')
        elif epoch == 99:
            self._save_interval_best('late_best', '60-100')
    """
    pass

# =============================================================================
# 主要接口函数
# =============================================================================

def create_model_factory(config: MambaConfig) -> BaseModel:
    """模型工厂函数"""
    return MambaModel(config)

def create_data_loader_manager(data_config_path: str, config: MambaConfig) -> MambaDataLoaderManager:
    """数据加载器管理器工厂函数"""
    return MambaDataLoaderManager(data_config_path, config)

def main_train(config_path: str = "config.yaml", data_config_path: str = "data.yaml",
               optuna_config_path: str = None, device: str = "auto",
               epochs_override: int = None, no_save_model: bool = False,
               seed: int = 42, checkpoint_dir: str = "checkpoints"):
    # 设置随机种子
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"🎲 设置随机种子: {seed}")
    logger.info(f"💾 Checkpoint保存目录: {checkpoint_dir}")
    
    # 确保checkpoint目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    """主训练函数"""
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        config_dict = {}
    
    # 提取相关配置并创建配置对象
    mamba_config_dict = {}
    if 'architecture' in config_dict:
        arch_config = config_dict['architecture']
        mamba_config_dict.update({
            'd_model': arch_config.get('d_model', 256),
            'n_layer': arch_config.get('n_layer', 8),
            'd_state': arch_config.get('d_state', 16),
            'd_conv': arch_config.get('d_conv', 4),
            'expand': arch_config.get('expand', 2),
            'dropout': arch_config.get('dropout', 0.1),
        })
    
    if 'training' in config_dict:
        train_config = config_dict['training']
        mamba_config_dict.update({
            'learning_rate': train_config.get('learning_rate', 0.0003),
            'batch_size': train_config.get('batch_size', 32),
            'epochs': train_config.get('epochs', 100),
            'weight_decay': train_config.get('weight_decay', 1e-5),
        })
    
    config = MambaConfig(**mamba_config_dict)
    
    # 应用Optuna配置覆盖
    if optuna_config_path and os.path.exists(optuna_config_path):
        with open(optuna_config_path, 'r', encoding='utf-8') as f:
            optuna_config = yaml.safe_load(f)
        
        # 应用超参数覆盖
        for key, value in optuna_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"🔧 Optuna覆盖参数: {key} = {value}")
    
    # 覆盖配置
    if device != "auto":
        config.device = device
    if epochs_override:
        config.epochs = epochs_override
    
    # 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 创建模型
    model = create_model_factory(config)
    
    # 创建训练器
    trainer = UnifiedTrainer(config, model, data_loader_manager, checkpoint_dir=checkpoint_dir)
    
    # 执行训练
    results = trainer.train(no_save_model=no_save_model)
    
    # 输出JSON格式结果供Optuna解析
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
    
    return results

def main_inference(config_path: str = "config.yaml", data_config_path: str = "data.yaml", 
                  checkpoint_path: str = "best_model.pth"):
    """主推理函数"""
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        config_dict = {}
    
    # 提取相关配置并创建配置对象
    mamba_config_dict = {}
    if 'architecture' in config_dict:
        arch_config = config_dict['architecture']
        mamba_config_dict.update({
            'd_model': arch_config.get('d_model', 256),
            'n_layer': arch_config.get('n_layer', 8),
            'd_state': arch_config.get('d_state', 16),
            'd_conv': arch_config.get('d_conv', 4),
            'expand': arch_config.get('expand', 2),
            'dropout': arch_config.get('dropout', 0.1),
        })
    
    config = MambaConfig(**mamba_config_dict)
    
    # 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 创建模型
    model = create_model_factory(config)
    
    # 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = inferencer.load_checkpoint(checkpoint_path)
    else:
        print(f"⚠️ 检查点文件 {checkpoint_path} 不存在，使用随机初始化模型")
    
    # 执行推理和评估
    metrics = inferencer.evaluate()
    
    print("推理结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        config_dict = {}
    
    # 提取相关配置并创建配置对象
    mamba_config_dict = {}
    if 'architecture' in config_dict:
        arch_config = config_dict['architecture']
        mamba_config_dict.update({
            'd_model': arch_config.get('d_model', 256),
            'n_layer': arch_config.get('n_layer', 8),
            'd_state': arch_config.get('d_state', 16),
            'd_conv': arch_config.get('d_conv', 4),
            'expand': arch_config.get('expand', 2),
            'dropout': arch_config.get('dropout', 0.1),
        })
    
    config = MambaConfig(**mamba_config_dict)
    
    # 创建模型
    model = create_model_factory(config)
    
    # 生成文档
    doc_generator = ModelDocumentationGenerator(config, model)
    doc_generator.generate_model_md("MODEL.md")
    
    print("文档生成完成")

# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="统一Mamba模型训练、推理和文档生成工具")
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
    
    return parser.parse_args()

if __name__ == "__main__":
    # 设置固定的日志路径
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/mamba_3493354528"
    logger = setup_logging(log_dir=log_dir, log_filename="log.txt")
    
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
        main_inference(args.config, args.data_config, args.checkpoint)
    elif args.mode == "docs":
        main_generate_docs(args.config, args.data_config)