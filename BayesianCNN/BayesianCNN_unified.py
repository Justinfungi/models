#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BayesianCNN_unified.py - 统一模型实现
基于贝叶斯卷积神经网络的统一模型实现，支持不确定性量化和Optuna超参数优化
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
from torch.nn import Parameter
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
class BayesianCNNConfig:
    """BayesianCNN配置类"""
    # 模型基础配置
    model_name: str = "BayesianCNN"
    model_type: str = "classification"
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # 性能优化配置
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    gradient_clip_value: float = 1.0
    
    # 贝叶斯配置
    prior_mu: float = 0.0
    prior_sigma: float = 0.1
    posterior_mu_initial: List[float] = None
    posterior_rho_initial: List[float] = None
    kl_weight: float = 0.1
    
    # 网络配置
    input_dim: Optional[int] = None
    hidden_dims: List[int] = None
    output_dim: Optional[int] = None
    dropout: float = 0.2
    activation: str = 'relu'
    
    # 集成配置
    train_samples: int = 1
    val_samples: int = 5
    test_samples: int = 10
    
    # 设备配置
    device: str = "auto"
    seed: int = 42
    
    # 路径配置
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        if self.posterior_mu_initial is None:
            self.posterior_mu_initial = [0.0, 0.1]
        if self.posterior_rho_initial is None:
            self.posterior_rho_initial = [-3.0, 0.1]
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

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

def create_directories(config: BayesianCNNConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 贝叶斯层实现
# =============================================================================

class ModuleWrapper(nn.Module):
    """贝叶斯层的基础包装类"""
    
    def __init__(self):
        super(ModuleWrapper, self).__init__()
    
    def kl_divergence(self) -> torch.Tensor:
        """计算KL散度，子类需要实现此方法"""
        raise NotImplementedError("子类必须实现kl_divergence方法")

class BBBLinear(ModuleWrapper):
    """贝叶斯线性层实现"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 prior_mu: float = 0.0, prior_sigma: float = 0.1,
                 posterior_mu_initial: List[float] = None, 
                 posterior_rho_initial: List[float] = None):
        super(BBBLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # 设置先验分布参数
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.posterior_mu_initial = posterior_mu_initial or [0.0, 0.1]
        self.posterior_rho_initial = posterior_rho_initial or [-3.0, 0.1]
        
        # 权重参数
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        
        # 偏置参数
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """重置参数到初始值"""
        self.weight_mu.data.normal_(
            self.posterior_mu_initial[0], 
            self.posterior_mu_initial[1]
        )
        self.weight_rho.data.normal_(
            self.posterior_rho_initial[0], 
            self.posterior_rho_initial[1]
        )
        
        if self.use_bias:
            self.bias_mu.data.normal_(
                self.posterior_mu_initial[0], 
                self.posterior_mu_initial[1]
            )
            self.bias_rho.data.normal_(
                self.posterior_rho_initial[0], 
                self.posterior_rho_initial[1]
            )
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """前向传播"""
        if sample:
            # 采样权重
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_eps = torch.randn_like(self.weight_mu).to(self.weight_mu.device)
            weight = self.weight_mu + weight_sigma * weight_eps
            
            # 采样偏置
            if self.use_bias:
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_eps = torch.randn_like(self.bias_mu).to(self.bias_mu.device)
                bias = self.bias_mu + bias_sigma * bias_eps
            else:
                bias = None
        else:
            # 使用均值
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """计算KL散度"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_kl = self._kl_divergence_gaussian(
            self.weight_mu, weight_sigma, self.prior_mu, self.prior_sigma
        )
        
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_kl = self._kl_divergence_gaussian(
                self.bias_mu, bias_sigma, self.prior_mu, self.prior_sigma
            )
            return weight_kl + bias_kl
        else:
            return weight_kl
    
    def _kl_divergence_gaussian(self, mu_q: torch.Tensor, sigma_q: torch.Tensor, 
                               mu_p: float, sigma_p: float) -> torch.Tensor:
        """计算两个高斯分布之间的KL散度"""
        kl = torch.log(sigma_p / sigma_q) + \
             (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p**2) - 0.5
        return kl.sum()

# =============================================================================
# 数据处理类
# =============================================================================

class BayesianCNNDataLoaderManager:
    """BayesianCNN数据加载器管理类"""
    
    def __init__(self, data_config_path: str, config: BayesianCNNConfig):
        self.data_config_path = data_config_path
        self.config = config
        self.data_config = self._load_data_config()
        self.data_split_info = self.data_config.get('data_split', {})
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
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
            data_split = self.data_split_info
            
            train_file = data_split['train']['file']
            valid_file = data_split['validation']['file']
            test_file = data_split['test']['file']
            
            # 加载数据文件
            train_df = pd.read_parquet(train_file)
            valid_df = pd.read_parquet(valid_file)
            test_df = pd.read_parquet(test_file)
            
            print(f"✅ 数据加载成功:")
            print(f"  训练集: {train_df.shape}")
            print(f"  验证集: {valid_df.shape}")
            print(f"  测试集: {test_df.shape}")
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            print(f"⚠️ 无法加载真实数据文件: {e}")
            print("🔄 生成模拟数据用于测试...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """生成模拟数据用于测试"""
        n_samples = 1000
        n_features = 69
        n_classes = 2
        
        # 生成特征数据
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        # 创建特征列名
        feature_columns = [f"feature_{i}@test" for i in range(n_features)]
        
        # 创建DataFrame
        df = pd.DataFrame(X, columns=feature_columns)
        df['label'] = y
        
        # 分割数据
        train_size = int(0.6 * n_samples)
        valid_size = int(0.2 * n_samples)
        
        train_df = df[:train_size].copy()
        valid_df = df[train_size:train_size+valid_size].copy()
        test_df = df[train_size+valid_size:].copy()
        
        print(f"✅ 模拟数据生成成功:")
        print(f"  训练集: {train_df.shape}")
        print(f"  验证集: {valid_df.shape}")
        print(f"  测试集: {test_df.shape}")
        
        return train_df, valid_df, test_df
    
    def get_input_dim_from_dataframe(self, df: pd.DataFrame) -> int:
        """从DataFrame获取特征维度"""
        # 识别特征列（包含@符号的列）
        feature_cols = [col for col in df.columns if '@' in col]
        if not feature_cols:
            # 如果没有@符号，假设最后一列是标签，其余都是特征
            feature_cols = [col for col in df.columns if col != 'label']
        
        features = len(feature_cols)
        print(f"🔍 检测到输入特征维度: {features}")
        return features
    
    def validate_input_dimensions(self, config: BayesianCNNConfig, actual_input_dim: int) -> int:
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
        """加载数据加载器"""
        # 加载真实数据
        train_df, valid_df, test_df = self._load_real_dataframes()
        
        # 动态检测特征维度
        feature_cols = [col for col in train_df.columns if '@' in col]
        if not feature_cols:
            feature_cols = [col for col in train_df.columns if col != 'label']
        
        input_dim = len(feature_cols)
        print(f"🔍 检测到 {input_dim} 个特征")
        
        # 验证并更新配置
        self.validate_input_dimensions(self.config, input_dim)
        
        # 准备数据
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values if 'label' in train_df.columns else train_df.iloc[:, -1].values
        
        X_valid = valid_df[feature_cols].values
        y_valid = valid_df['label'].values if 'label' in valid_df.columns else valid_df.iloc[:, -1].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values if 'label' in test_df.columns else test_df.iloc[:, -1].values
        
        # 数据预处理
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 标签编码
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_valid_encoded = self.label_encoder.transform(y_valid)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 更新输出维度
        self.config.output_dim = len(np.unique(y_train_encoded))
        
        # 转换为张量
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_encoded)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(X_valid_scaled),
            torch.LongTensor(y_valid_encoded)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.LongTensor(y_test_encoded)
        )
        
        # 创建数据加载器
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
# 模型实现
# =============================================================================

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, config: BayesianCNNConfig):
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

class BayesianCNNModel(BaseModel):
    """BayesianCNN模型实现"""
    
    def __init__(self, config: BayesianCNNConfig):
        super().__init__(config)
        
        # 获取网络参数
        self.input_dim = getattr(config, 'input_dim', None)
        self.hidden_dims = config.hidden_dims
        self.output_dim = getattr(config, 'output_dim', None)
        self.dropout = config.dropout
        self.activation = config.activation
        
        # 贝叶斯配置
        self.prior_mu = config.prior_mu
        self.prior_sigma = config.prior_sigma
        self.posterior_mu_initial = config.posterior_mu_initial
        self.posterior_rho_initial = config.posterior_rho_initial
        
        # 延迟初始化网络层
        self.layers = None
        self.dropout_layers = None
        
        # 激活函数
        self.activation_fn = self._get_activation_function(self.activation)
        
        # 如果输入输出维度已知，立即构建网络
        if self.input_dim is not None and self.output_dim is not None:
            self._build_layers()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        
        if activation.lower() not in activation_map:
            print(f"⚠️ 未知的激活函数: {activation}，使用ReLU")
            return nn.ReLU()
        
        return activation_map[activation.lower()]
    
    def _build_layers(self):
        """构建网络层"""
        if self.input_dim is None or self.output_dim is None:
            print("⚠️ 输入或输出维度未指定，无法构建网络")
            return
        
        print(f"🔧 构建网络层，输入维度: {self.input_dim}, 输出维度: {self.output_dim}")
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(
            BBBLinear(
                self.input_dim, 
                self.hidden_dims[0],
                prior_mu=self.prior_mu,
                prior_sigma=self.prior_sigma,
                posterior_mu_initial=self.posterior_mu_initial,
                posterior_rho_initial=self.posterior_rho_initial
            )
        )
        self.dropout_layers.append(nn.Dropout(self.dropout))
        
        # 隐藏层
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(
                BBBLinear(
                    self.hidden_dims[i], 
                    self.hidden_dims[i + 1],
                    prior_mu=self.prior_mu,
                    prior_sigma=self.prior_sigma,
                    posterior_mu_initial=self.posterior_mu_initial,
                    posterior_rho_initial=self.posterior_rho_initial
                )
            )
            self.dropout_layers.append(nn.Dropout(self.dropout))
        
        # 输出层
        self.layers.append(
            BBBLinear(
                self.hidden_dims[-1], 
                self.output_dim,
                prior_mu=self.prior_mu,
                prior_sigma=self.prior_sigma,
                posterior_mu_initial=self.posterior_mu_initial,
                posterior_rho_initial=self.posterior_rho_initial
            )
        )
        
        # 确保新构建的层在正确的设备上
        if hasattr(self, '_device'):
            self.layers.to(self._device)
            self.dropout_layers.to(self._device)
            print(f"🔧 新构建的层已移动到设备: {self._device}")
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """前向传播"""
        # 首次前向传播时自动检测并构建网络
        if self.layers is None:
            if self.input_dim is None:
                self.input_dim = x.shape[-1]
                self.config.input_dim = self.input_dim
                print(f"🔍 自动检测输入维度: {self.input_dim}")
            
            if self.output_dim is None:
                self.output_dim = 2  # 默认二分类
                self.config.output_dim = self.output_dim
                print(f"🔍 默认输出维度: {self.output_dim}")
            
            self._build_layers()
        
        # 前向传播
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, sample=sample)
            x = self.activation_fn(x)
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
        
        # 输出层（不使用激活函数和dropout）
        if len(self.layers) > 0:
            x = self.layers[-1](x, sample=sample)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """计算所有贝叶斯层的总KL散度"""
        if self.layers is None:
            return torch.tensor(0.0)
        
        kl_sum = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for layer in self.layers:
            if isinstance(layer, BBBLinear):
                kl_sum += layer.kl_divergence()
        
        return kl_sum
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用蒙特卡洛采样进行不确定性量化预测"""
        self.eval()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # 计算均值和标准差
        pred_mean = predictions.mean(dim=0)
        pred_std = predictions.std(dim=0)
        
        return pred_mean, pred_std

# =============================================================================
# 损失函数
# =============================================================================

class ELBOLoss(nn.Module):
    """Evidence Lower Bound (ELBO) 损失函数"""
    
    def __init__(self, train_size: int):
        super().__init__()
        self.train_size = train_size
        self.nll_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                kl_divergence: torch.Tensor, beta: float) -> torch.Tensor:
        """计算ELBO损失"""
        nll = self.nll_loss(outputs, targets)
        kl_scaled = kl_divergence / self.train_size
        return nll + beta * kl_scaled

def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """稳定的log-mean-exp计算"""
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])

# =============================================================================
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: BayesianCNNConfig, model: BayesianCNNModel, data_loader_manager: BayesianCNNDataLoaderManager,
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
    
    def _initialize_model_layers(self, train_loader: DataLoader):
        """通过一次前向传播初始化模型网络层"""
        if self.model.layers is None:
            # 获取一个批次的数据来初始化模型
            for data, target in train_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # 设置模型的设备信息，以便在构建层时使用正确的设备
                self.model._device = self.device
                
                # 执行一次前向传播来构建网络层
                with torch.no_grad():
                    _ = self.model(data, sample=False)
                
                # 确保模型完全移动到设备
                self.model.to(self.device)
                print(f"🔧 模型已移动到设备: {self.device}")
                
                # 更新输出维度
                if self.model.output_dim is None or self.model.output_dim == 2:
                    unique_labels = torch.unique(target)
                    actual_output_dim = len(unique_labels)
                    if actual_output_dim != self.model.output_dim:
                        self.model.output_dim = actual_output_dim
                        self.config.output_dim = actual_output_dim
                        print(f"🔍 更新输出维度为: {actual_output_dim}")
                        # 重新构建网络层
                        self.model._build_layers()
                        # 再次确保模型在正确设备上
                        self.model.to(self.device)
                
                break  # 只需要一个批次来初始化
            
            # 重新记录模型参数信息
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"📊 模型初始化后总参数: {total_params:,}")
            self.logger.info(f"📊 模型初始化后可训练参数: {trainable_params:,}")
            self.logger.info(f"📊 模型初始化后大小: {total_params * 4 / 1024**2:.1f}MB")
    
    def setup_training_components(self, train_size: int):
        """设置训练组件"""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        self.criterion = ELBOLoss(train_size).to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_kl = 0.0
        all_preds = []
        all_targets = []
        
        num_ens = self.config.train_samples
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 集成采样
            if num_ens > 1:
                outputs_list = []
                kl_total = 0.0
                
                for j in range(num_ens):
                    net_out = self.model(data, sample=True)
                    kl = self.model.kl_divergence()
                    kl_total += kl
                    outputs_list.append(F.log_softmax(net_out, dim=1))
                
                outputs = torch.stack(outputs_list, dim=2)
                log_outputs = logmeanexp(outputs, dim=2)
                kl_avg = kl_total / num_ens
            else:
                # 单次采样
                outputs = self.model(data, sample=True)
                log_outputs = F.log_softmax(outputs, dim=1)
                kl_avg = self.model.kl_divergence()
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    loss = self.criterion(torch.exp(log_outputs), target, kl_avg, self.config.kl_weight)
                
                self.scaler.scale(loss).backward()
                
                if hasattr(self.config, 'gradient_clip_value'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.criterion(torch.exp(log_outputs), target, kl_avg, self.config.kl_weight)
                loss.backward()
                
                if hasattr(self.config, 'gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            total_kl += kl_avg.item()
            
            # 记录预测
            preds = torch.argmax(log_outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        return {"loss": avg_loss, "accuracy": accuracy, "kl_divergence": avg_kl}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        num_ens = self.config.val_samples
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 集成预测
                if num_ens > 1:
                    outputs_list = []
                    kl_total = 0.0
                    
                    for j in range(num_ens):
                        net_out = self.model(data, sample=True)
                        kl = self.model.kl_divergence()
                        kl_total += kl
                        outputs_list.append(F.log_softmax(net_out, dim=1))
                    
                    outputs = torch.stack(outputs_list, dim=2)
                    log_outputs = logmeanexp(outputs, dim=2)
                    kl_avg = kl_total / num_ens
                else:
                    outputs = self.model(data, sample=True)
                    log_outputs = F.log_softmax(outputs, dim=1)
                    kl_avg = self.model.kl_divergence()
                
                # 混合精度推理
                if self.use_amp:
                    with autocast():
                        loss = self.criterion(torch.exp(log_outputs), target, kl_avg, self.config.kl_weight)
                else:
                    loss = self.criterion(torch.exp(log_outputs), target, kl_avg, self.config.kl_weight)
                
                total_loss += loss.item()
                
                # 记录预测
                probs = torch.exp(log_outputs)
                preds = torch.argmax(log_outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = {"loss": total_loss / len(val_loader)}
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
        
        # 计算AUC
        if len(np.unique(all_targets)) == 2:
            try:
                all_probs_array = np.array(all_probs)
                if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                    metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
            except Exception:
                pass
        
        return metrics
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """完整训练流程"""
        # 获取数据加载器
        batch_size = batch_size or self.config.batch_size
        train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(batch_size=batch_size)
        
        # 初始化模型网络层（通过一次前向传播）
        self._initialize_model_layers(train_loader)
        
        self.setup_training_components(len(train_loader.dataset))
        
        self.logger.info("开始训练...")
        self.logger.info(f"模型信息: {self.model.get_model_info()}")
        
        start_time = time.time()
        epochs = self.config.epochs
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader)
            
            # 记录历史
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_acc": train_metrics.get("accuracy", 0),
                "val_acc": val_metrics.get("accuracy", 0),
                "train_auc": train_metrics.get("auc", 0),
                "val_auc": val_metrics.get("auc", 0),
                "lr": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - epoch_start
            }
            
            current_score = val_metrics.get("accuracy", 0)
            self.train_history.append(epoch_info)
            
            # 学习率调度
            self.scheduler.step(current_score)
            
            # 保存checkpoint（使用新的智能策略）
            if not no_save_model:
                val_loss = epoch_info.get('val_loss', float('inf'))
                self.save_checkpoint_if_best(epoch, val_loss)
            
            # 更新best_val_score（用于向后兼容）
            if current_score > self.best_val_score:
                self.best_val_score = current_score
            
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点（向后兼容的旧方法）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'config': asdict(self.config)
        }
        
        # 使用传入的checkpoint目录（优先），否则使用配置中的
        checkpoint_dir = Path(self.checkpoint_dir) if self.checkpoint_dir else Path(self.config.checkpoint_dir)
        
        # 保存最新检查点
        checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
    
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
    
    def __init__(self, config: BayesianCNNConfig, model: BayesianCNNModel, data_loader_manager: BayesianCNNDataLoaderManager):
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
        
        # 检查模型是否已经构建网络层
        if self.model.layers is None:
            # 从checkpoint中获取配置信息来构建网络层
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                if 'input_dim' in saved_config and 'output_dim' in saved_config:
                    self.model.input_dim = saved_config['input_dim']
                    self.model.output_dim = saved_config['output_dim']
                    self.model._device = self.device  # 设置设备信息
                    self.model._build_layers()
                    self.model.to(self.device)  # 确保新构建的层在正确设备上
                    self.logger.info(f"根据checkpoint配置构建网络层: input_dim={self.model.input_dim}, output_dim={self.model.output_dim}")
                else:
                    self.logger.warning("checkpoint中缺少input_dim或output_dim配置")
            else:
                # 尝试从state_dict推断网络结构
                state_dict = checkpoint['model_state_dict']
                layer_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
                if layer_keys:
                    # 从第一层推断input_dim
                    first_layer_weight_key = 'layers.0.weight_mu'
                    if first_layer_weight_key in state_dict:
                        input_dim = state_dict[first_layer_weight_key].shape[1]
                        self.model.input_dim = input_dim
                        self.logger.info(f"从checkpoint推断input_dim: {input_dim}")
                    
                    # 从最后一层推断output_dim
                    last_layer_idx = max([int(k.split('.')[1]) for k in layer_keys if '.' in k and k.split('.')[1].isdigit()])
                    last_layer_weight_key = f'layers.{last_layer_idx}.weight_mu'
                    if last_layer_weight_key in state_dict:
                        output_dim = state_dict[last_layer_weight_key].shape[0]
                        self.model.output_dim = output_dim
                        self.logger.info(f"从checkpoint推断output_dim: {output_dim}")
                    
                    # 构建网络层
                    if hasattr(self.model, 'input_dim') and hasattr(self.model, 'output_dim'):
                        self.model._device = self.device  # 设置设备信息
                        self.model._build_layers()
                        self.model.to(self.device)  # 确保新构建的层在正确设备上
                        self.logger.info("根据checkpoint state_dict构建网络层")
        
        # 加载状态字典
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 确保模型完全在正确的设备上
            self.model.to(self.device)
            self.logger.info(f"成功加载模型检查点: {checkpoint_path}")
            self.logger.info(f"模型已移动到设备: {self.device}")
        except RuntimeError as e:
            self.logger.error(f"加载checkpoint失败: {e}")
            raise
        
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
        # 获取测试数据加载器
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
        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, average='weighted'),
            "recall": recall_score(all_targets, all_preds, average='weighted'),
            "f1": f1_score(all_targets, all_preds, average='weighted')
        }
        
        # AUC指标计算
        if len(np.unique(all_targets)) == 2:
            all_probs_positive = np.array(all_probs)[:, 1]
            metrics["auc"] = roc_auc_score(all_targets, all_probs_positive)
        
        return metrics
    
    def generate_predictions(self, start_date: str = None, end_date: str = None, 
                           output_path: str = None, output_format: str = "parquet") -> pd.DataFrame:
        """生成预测数据文件用于回测"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # 获取数据分割信息
        data_split_info = self.data_loader_manager.get_data_split_info()
        
        # 确定使用哪个数据集
        if start_date and end_date:
            # 使用指定日期范围的数据
            print(f"  🎯 使用指定日期范围: {start_date} 到 {end_date}")
            
            # 检查日期范围属于哪个数据集
            test_start = data_split_info.get('test', {}).get('start_date')
            test_end = data_split_info.get('test', {}).get('end_date')
            
            if test_start and test_end and start_date >= test_start and end_date <= test_end:
                print(f"  📊 使用测试集数据")
                _, _, data_loader = self.data_loader_manager.load_data_loaders()
            else:
                print(f"  ⚠️ 指定日期范围不在测试集内，使用测试集数据")
                _, _, data_loader = self.data_loader_manager.load_data_loaders()
        else:
            # 默认使用测试集
            print(f"  📊 使用测试集数据")
            _, _, data_loader = self.data_loader_manager.load_data_loaders()
            
            # 使用测试集的日期范围
            test_info = data_split_info.get('test', {})
            start_date = test_info.get('start_date', '2020-07-01')
            end_date = test_info.get('end_date', '2020-12-31')
        
        # 执行预测
        predictions, probabilities = self.predict_batch(data_loader)
        
        # 生成日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 如果预测数量与日期数量不匹配，调整日期范围
        if len(predictions) != len(date_range):
            print(f"  🔧 调整日期范围：预测数量={len(predictions)}, 日期数量={len(date_range)}")
            if len(predictions) < len(date_range):
                date_range = date_range[:len(predictions)]
            else:
                # 扩展日期范围
                additional_days = len(predictions) - len(date_range)
                end_date_dt = pd.to_datetime(end_date)
                extended_dates = pd.date_range(
                    start=end_date_dt + timedelta(days=1),
                    periods=additional_days,
                    freq='D'
                )
                date_range = date_range.append(extended_dates)
        
        # 创建预测数据DataFrame
        predictions_df = pd.DataFrame({
            'date': date_range[:len(predictions)],
            'prediction': predictions,
            'confidence': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0],
            'probability_class_0': probabilities[:, 0] if probabilities.shape[1] > 1 else 1 - probabilities[:, 0],
            'probability_class_1': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        })
        
        # 添加元数据
        predictions_df['model_name'] = self.config.model_name
        predictions_df['timestamp'] = datetime.now().isoformat()
        
        # 保存文件
        if output_path:
            if output_format == "parquet":
                predictions_df.to_parquet(output_path, index=False)
            elif output_format == "csv":
                predictions_df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
            print(f"  💾 预测数据已保存: {output_path}")
            print(f"  📊 数据形状: {predictions_df.shape}")
            print(f"  📅 日期范围: {predictions_df['date'].min()} 到 {predictions_df['date'].max()}")
        
        return predictions_df
    
    def save_predictions(self, predictions: np.ndarray, probabilities: np.ndarray, output_path: str):
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
    
    def __init__(self, config: BayesianCNNConfig, model: BayesianCNNModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个基于贝叶斯深度学习的{self.config.model_type}模型，支持不确定性量化。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB
- **不确定性量化**: 支持贝叶斯推理

## 模型架构

### 贝叶斯网络结构

- 输入维度: {self.config.input_dim}
- 隐藏层: {self.config.hidden_dims}
- 输出维度: {self.config.output_dim}
- 激活函数: {self.config.activation}
- Dropout率: {self.config.dropout}

### 贝叶斯配置

- 先验均值: {self.config.prior_mu}
- 先验标准差: {self.config.prior_sigma}
- KL权重: {self.config.kl_weight}

## 技术原理

该模型使用变分贝叶斯推理，通过BBBLinear层实现权重的概率分布建模，
支持不确定性量化和鲁棒性预测。

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}
- 混合精度: {self.config.use_mixed_precision}

### 集成配置
- 训练采样: {self.config.train_samples}
- 验证采样: {self.config.val_samples}
- 测试采样: {self.config.test_samples}

## 使用方法

### 训练模型

```python
python BayesianCNN_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```python
python BayesianCNN_unified.py inference --checkpoint best_model.pth
```

### 生成文档

```python
python BayesianCNN_unified.py docs
```

## 性能特点

- 支持GPU加速训练
- 混合精度训练优化
- 不确定性量化能力
- 贝叶斯集成预测

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# 配置处理函数
# =============================================================================

def _flatten_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """将嵌套的配置字典扁平化为BayesianCNNConfig可接受的格式"""
    flattened = {}
    
    # 处理architecture部分
    if 'architecture' in config_dict:
        arch = config_dict['architecture']
        
        # 处理贝叶斯配置
        if 'bayesian_config' in arch:
            bayesian = arch['bayesian_config']
            flattened.update({
                'prior_mu': bayesian.get('prior_mu', 0.0),
                'prior_sigma': bayesian.get('prior_sigma', 0.1),
                'posterior_mu_initial': bayesian.get('posterior_mu_initial', [0.0, 0.1]),
                'posterior_rho_initial': bayesian.get('posterior_rho_initial', [-3.0, 0.1])
            })
        
        # 处理网络配置
        if 'network' in arch:
            network = arch['network']
            flattened.update({
                'input_dim': network.get('input_dim'),
                'hidden_dims': network.get('hidden_dims', [128, 64]),
                'output_dim': network.get('output_dim'),
                'dropout': network.get('dropout', 0.2),
                'activation': network.get('activation', 'relu')
            })
    
    # 处理training部分
    if 'training' in config_dict:
        training = config_dict['training']
        
        # 基本训练参数
        flattened.update({
            'epochs': training.get('epochs', 100),
            'batch_size': training.get('batch_size', 32)
        })
        
        # 优化器配置
        if 'optimizer' in training:
            optimizer = training['optimizer']
            flattened.update({
                'learning_rate': optimizer.get('learning_rate', 0.001)
            })
        
        # 损失函数配置
        if 'loss' in training:
            loss = training['loss']
            flattened.update({
                'kl_weight': loss.get('kl_weight', 0.1)
            })
    
    # 处理inference部分
    if 'inference' in config_dict:
        inference = config_dict['inference']
        
        # 采样配置
        if 'sampling' in inference:
            sampling = inference['sampling']
            flattened.update({
                'train_samples': sampling.get('num_samples', 1),
                'val_samples': sampling.get('num_samples', 5),
                'test_samples': sampling.get('num_samples', 10)
            })
    
    # 处理顶层配置（直接映射）
    direct_mappings = [
        'model_name', 'model_type', 'device', 'seed', 'use_mixed_precision',
        'dataloader_num_workers', 'pin_memory', 'gradient_clip_value',
        'data_path', 'checkpoint_dir', 'results_dir', 'logs_dir'
    ]
    
    for key in direct_mappings:
        if key in config_dict:
            flattened[key] = config_dict[key]
    
    return flattened

# =============================================================================
# 主要接口函数
# =============================================================================

def create_model_factory(config: BayesianCNNConfig) -> BayesianCNNModel:
    """模型工厂函数"""
    return BayesianCNNModel(config)

def create_data_loader_manager(data_config_path: str, config: BayesianCNNConfig) -> BayesianCNNDataLoaderManager:
    """数据加载器管理器工厂函数"""
    return BayesianCNNDataLoaderManager(data_config_path, config)

def main_train(config_path: str = "config.yaml", data_config_path: str = "data.yaml",
               optuna_config_path: str = None, device: str = "auto",
               epochs_override: int = None, no_save_model: bool = False,
               seed: int = 42, checkpoint_dir: str = "checkpoints"):
    """主训练函数"""
    # 设置随机种子
    import random
    import numpy as np
    import torch
    import logging
    logger = logging.getLogger(__name__)
    
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
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件不存在: {config_path}，使用默认配置")
        config_dict = {}
    
    # 解析嵌套配置结构
    flattened_config = _flatten_config(config_dict)
    
    # 创建配置对象
    config = BayesianCNNConfig(**flattened_config)
    
    # 应用Optuna配置覆盖
    if optuna_config_path and os.path.exists(optuna_config_path):
        try:
            with open(optuna_config_path, 'r', encoding='utf-8') as f:
                optuna_config = yaml.safe_load(f)
            
            # 应用超参数覆盖
            for key, value in optuna_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"🔧 Optuna覆盖参数: {key} = {value}")
        except Exception as e:
            print(f"⚠️ 无法加载Optuna配置: {e}")
    
    # 覆盖配置
    if device != "auto":
        config.device = device
    if epochs_override:
        config.epochs = epochs_override
    
    # 设置固定的日志路径
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/BayesianCNN_3356644015"
    logger = setup_logging(log_dir=log_dir, log_filename="log.txt")
    
    # 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 创建模型
    model = create_model_factory(config)
    
    # 创建训练器（传递checkpoint目录）
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
    
    return results

def main_inference(config_path: str = "config.yaml", data_config_path: str = "data.yaml", 
                  checkpoint_path: str = "best_model.pth", mode: str = "eval",
                  start_date: str = None, end_date: str = None, 
                  output_path: str = None, output_format: str = "parquet"):
    """主推理函数 - 支持评估和预测文件生成"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 解析嵌套配置结构
    flattened_config = _flatten_config(config_dict)
    config = BayesianCNNConfig(**flattened_config)
    
    # 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 创建模型
    model = create_model_factory(config)
    
    # 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 加载检查点
    checkpoint = inferencer.load_checkpoint(checkpoint_path)
    
    if mode == "test":
        # 生成预测文件模式
        if not output_path:
            output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        
        print(f"🔮 生成预测文件模式")
        print(f"  📅 日期范围: {start_date} 到 {end_date}")
        print(f"  📄 输出文件: {output_path}")
        print(f"  📊 输出格式: {output_format}")
        
        # 生成预测数据
        predictions_data = inferencer.generate_predictions(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            output_format=output_format
        )
        
        print(f"✅ 预测文件生成完成: {output_path}")
        return {"predictions_file": output_path, "predictions_count": len(predictions_data)}
        
    else:
        # 评估模式
        print(f"📊 模型评估模式")
        metrics = inferencer.evaluate()
        print(f"推理结果: {metrics}")
        return metrics

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 解析嵌套配置结构
    flattened_config = _flatten_config(config_dict)
    config = BayesianCNNConfig(**flattened_config)
    
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
    parser = argparse.ArgumentParser(description="BayesianCNN统一模型训练、推理和文档生成工具")
    parser.add_argument("mode", choices=["train", "inference", "docs"], help="运行模式")
    parser.add_argument("--config", default="config.yaml", help="模型配置文件路径")
    parser.add_argument("--data-config", default="data.yaml", help="数据配置文件路径")
    parser.add_argument("--checkpoint", default="best_model.pth", help="模型检查点路径")
    parser.add_argument("--data", help="数据文件路径")
    parser.add_argument("--output", help="输出文件路径")
    
    # 推理相关参数
    parser.add_argument("--inference-mode", choices=["test", "eval"], default="eval", help="推理模式：test(生成预测文件) 或 eval(评估)")
    parser.add_argument("--start-date", help="推理开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="推理结束日期 (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="输出格式")
    
    # Optuna优化相关参数
    parser.add_argument("--optuna-config", help="Optuna试验配置文件路径")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto", help="训练设备")
    parser.add_argument("--epochs", type=int, help="训练轮数（覆盖配置文件）")
    parser.add_argument("--no-save-model", action="store_true", help="不保存模型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确保可复现性）")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint保存目录")
    
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