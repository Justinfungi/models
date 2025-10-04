#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM_unified.py - HRM (Hierarchical Reasoning Model) 统一模型实现
整合训练、推理和文档生成功能的统一模板，支持Optuna超参数优化
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
import glob
import random
import math

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

warnings.filterwarnings('ignore')

# =============================================================================
# 基础配置和工具函数
# =============================================================================

@dataclass
class HRMConfig:
    """HRM模型配置类"""
    # 模型基础配置
    model_name: str = "HRM"
    model_type: str = "classification"  # classification, regression
    
    # 模型架构配置
    input_dim: Optional[int] = None  # 动态推断
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 8
    n_heads: int = 8
    hierarchical_levels: int = 3
    reasoning_depth: int = 4
    h_cycles: int = 3
    l_cycles: int = 2
    halt_max_steps: int = 10
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = 'swiglu'
    norm_type: str = 'rms'
    norm_eps: float = 1e-6
    num_classes: int = 2
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-5
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    gradient_clip_norm: float = 1.0
    
    # 性能优化配置
    use_mixed_precision: bool = True  # 混合精度训练
    dataloader_num_workers: int = 4  # 数据加载并发数
    pin_memory: bool = True  # 固定内存
    gradient_clip_value: float = 1.0  # 梯度裁剪
    
    # 设备配置
    device: str = "auto"
    seed: int = 42
    
    # 路径配置
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"

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
        # 使用固定的日志文件名
        log_file = log_dir / log_filename
    else:
        # 使用带时间戳的日志文件名（原有逻辑）
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
        force=True  # 强制重新配置
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger

def create_directories(config: HRMConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# HRM模型架构组件
# =============================================================================

class RMSNorm(nn.Module):
    """RMS归一化层"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (norm + self.eps) * self.weight

class SwiGLU(nn.Module):
    """SwiGLU激活函数"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.w3 = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x))) * self.w3(x)

class HierarchicalAttention(nn.Module):
    """层级注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        # 计算Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out, attn_weights

class ReasoningBlock(nn.Module):
    """推理模块"""
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1, 
                 activation: str = 'swiglu', norm_type: str = 'rms', norm_eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        
        # 注意力层
        self.self_attn = HierarchicalAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        if activation == 'swiglu':
            self.ffn = SwiGLU(d_model)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
        
        # 归一化层
        if norm_type == 'rms':
            self.norm1 = RMSNorm(d_model, float(norm_eps))
            self.norm2 = RMSNorm(d_model, float(norm_eps))
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=float(norm_eps))
            self.norm2 = nn.LayerNorm(d_model, eps=float(norm_eps))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 自注意力 + 残差连接
        attn_out, attn_weights = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # 前馈网络 + 残差连接
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x, attn_weights

class FeatureExtractor(nn.Module):
    """特征提取器"""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影到模型维度
        x = self.input_projection(x)
        x = self.dropout(x)
        x = self.norm(x)
        
        # 添加序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        return x

# =============================================================================
# 数据处理类
# =============================================================================

class HRMDataset(Dataset):
    """HRM模型专用数据集类"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
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

class HRMDataLoaderManager:
    """HRM数据加载器管理类"""
    
    def __init__(self, data_config_path: str, config: HRMConfig):
        self.data_config_path = data_config_path
        self.config = config
        self.data_config = self._load_data_config()
        self.data_split_info = self.data_config.get('data_split', {})
        self.scaler = StandardScaler()
        
    def _load_data_config(self) -> Dict[str, Any]:
        """加载数据配置文件，默认使用Phase 1配置"""
        try:
            with open(self.data_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 如果配置文件中有phase_1配置，使用它作为默认数据分割信息
            if 'phase_1' in config:
                phase_1_config = config['phase_1']
                # 构建标准的data_split格式
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
                print(f"  训练期: {config['data_split']['train']['start_date']} 到 {config['data_split']['train']['end_date']}")
                print(f"  验证期: {config['data_split']['validation']['start_date']} 到 {config['data_split']['validation']['end_date']}")
                print(f"  测试期: {config['data_split']['test']['start_date']} 到 {config['data_split']['test']['end_date']}")
            
            return config
        except Exception as e:
            raise ValueError(f"无法加载数据配置文件 {self.data_config_path}: {e}")
    
    def get_input_dim_from_dataframe(self, df):
        """从DataFrame获取特征维度（排除标签列）"""
        # 识别特征列（包含@符号的列）
        feature_cols = [col for col in df.columns if '@' in col]
        
        if not feature_cols:
            # 如果没有@符号的列，使用除了已知非特征列外的所有数值列
            exclude_cols = ['symbol', 'date', 'time', 'class', 'target', 'label']
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        features = len(feature_cols)
        print(f"🔍 检测到输入特征维度: {features}")
        return features
    
    def validate_input_dimensions(self, config, actual_input_dim):
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
    
    def _create_dataloader(self, dataset, shuffle: bool = False, batch_size: int = 32) -> DataLoader:
        """创建优化的数据加载器"""
        # 根据设备类型调整数据加载器配置
        pin_memory = torch.cuda.is_available()
        num_workers = 4 if torch.cuda.is_available() else 0  # CPU时减少worker数量避免问题
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # 只有在有worker时才启用
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    def _load_real_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """从真实parquet文件加载数据"""
        try:
            # 获取数据路径
            data_folder = self.data_config.get('data_paths', {}).get('data_folder', './data/feature_set')
            data_phase = self.data_config.get('data_paths', {}).get('data_phase', 1)
            
            # 尝试多个可能的数据路径
            possible_paths = [
                data_folder,
                '../../../b_model_reproduction_agent/data_1/feature_set',
                '../../../data_1/feature_set',
                '../../../data/feature_set',
                '../../data/feature_set',
                './data/feature_set',
                '../data/feature_set',
                '/home/feng.hao.jie/deployment/model_explorer/b_model_reproduction_agent/data_1/feature_set'
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
                raise FileNotFoundError(f"未找到数据文件，搜索模式: {pattern}")
            
            # 加载数据
            print(f"📂 加载数据文件: {data_files[0]}")
            if data_files[0].endswith('.pq'):
                df = pd.read_parquet(data_files[0])
            else:
                df = pd.read_csv(data_files[0])
            
            # 识别特征列（包含@符号的列）
            feature_cols = [col for col in df.columns if '@' in col]
            
            if not feature_cols:
                # 如果没有@符号的列，使用除了已知非特征列外的所有数值列
                exclude_cols = ['symbol', 'date', 'time', 'class', 'target', 'label']
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not feature_cols:
                raise ValueError("未找到有效的特征列")
            
            # 提取特征和标签
            X = df[feature_cols].values
            y = df['class'].values
            
            print(f"📊 数据形状: X={X.shape}, y={y.shape}")
            print(f"📊 特征列数: {len(feature_cols)}")
            print(f"📊 类别分布: {np.bincount(y)}")
            
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
            
            # 创建DataFrame用于返回
            train_df = pd.DataFrame(X_train, columns=feature_cols)
            train_df['class'] = y_train
            
            val_df = pd.DataFrame(X_val, columns=feature_cols)
            val_df['class'] = y_val
            
            test_df = pd.DataFrame(X_test, columns=feature_cols)
            test_df['class'] = y_test
            
            # 更新配置中的特征数量
            input_dim = self.validate_input_dimensions(self.config, X.shape[1])
            self.config.num_classes = len(np.unique(y))
            
            print(f"✅ 数据加载完成 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def load_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载真实数据的DataLoader"""
        # 从真实数据文件加载
        train_df, val_df, test_df = self._load_real_dataframes()
        
        # 分离特征和标签
        feature_cols = [col for col in train_df.columns if col != 'class']
        
        X_train = train_df[feature_cols].values
        y_train = train_df['class'].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df['class'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['class'].values
        
        # 创建数据集
        train_dataset = HRMDataset(X_train, y_train)
        val_dataset = HRMDataset(X_val, y_val)
        test_dataset = HRMDataset(X_test, y_test)
        
        # 创建数据加载器
        train_loader = self._create_dataloader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = self._create_dataloader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = self._create_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
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
# 模型基类
# =============================================================================

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, config: HRMConfig):
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
            "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设float32
        }

# =============================================================================
# HRM模型实现
# =============================================================================

class HRMModel(BaseModel):
    """HRM主模型类"""
    
    def __init__(self, config: HRMConfig):
        super().__init__(config)
        
        # 输入维度必须从配置中获取，不能硬编码
        self.input_dim = getattr(config, 'input_dim', None)
        
        # 如果配置中没有input_dim，必须在训练时动态设置
        if self.input_dim is None:
            print("⚠️ 配置中未指定input_dim，将在数据加载时自动检测")
        
        # 模型参数
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hierarchical_levels = config.hierarchical_levels
        self.reasoning_depth = config.reasoning_depth
        self.h_cycles = config.h_cycles
        self.l_cycles = config.l_cycles
        
        # 延迟初始化网络层，等待input_dim确定
        self.layers = None
        
    def _build_layers(self, input_dim):
        """使用动态输入维度构建网络，绝对不能硬编码"""
        print(f"🔧 构建网络层，输入维度: {input_dim}")
        
        # 特征提取器
        feature_extractor = FeatureExtractor(input_dim, self.d_model, self.config.dropout)
        
        # 推理层
        reasoning_layers = nn.ModuleList([
            ReasoningBlock(
                d_model=self.d_model,
                d_ff=self.config.d_ff,
                n_heads=self.n_heads,
                dropout=self.config.dropout,
                activation=self.config.activation,
                norm_type=self.config.norm_type,
                norm_eps=self.config.norm_eps
            ) for _ in range(self.n_layers)
        ])
        
        # 层级推理模块
        hierarchical_reasoning = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            ) for _ in range(self.hierarchical_levels)
        ])
        
        # 输出投影
        output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.d_model // 2, self.config.num_classes)
        )
        
        return nn.ModuleDict({
            'feature_extractor': feature_extractor,
            'reasoning_layers': reasoning_layers,
            'hierarchical_reasoning': hierarchical_reasoning,
            'output_projection': output_projection
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 确保输入数据在正确的设备上
        try:
            model_params = list(self.parameters())
            if len(model_params) > 0:
                model_device = next(iter(model_params)).device
                if x.device != model_device:
                    x = x.to(model_device)
        except StopIteration:
            # 如果模型还没有参数，跳过设备检查
            pass
        
        # 首次前向传播时自动检测并构建网络
        if self.layers is None:
            if self.input_dim is None:
                self.input_dim = x.shape[-1]  # 从输入数据自动检测
                print(f"🔍 自动检测输入维度: {self.input_dim}")
            self.layers = self._build_layers(self.input_dim)
            # 确保新构建的层在正确的设备上
            self.layers.to(x.device)
            print(f"🔧 网络层构建完成，已移动到设备: {x.device}")
        
        # 特征提取
        features = self.layers['feature_extractor'](x)  # [batch_size, 1, d_model]
        
        # 存储注意力权重
        attention_weights = []
        
        # 层级推理循环
        for h_cycle in range(self.h_cycles):
            # 高层推理
            for layer in self.layers['reasoning_layers']:
                features, attn_weights = layer(features)
                attention_weights.append(attn_weights)
            
            # 层级特征融合
            for level, hierarchical_layer in enumerate(self.layers['hierarchical_reasoning']):
                if level < len(self.layers['hierarchical_reasoning']) - 1:
                    features = features + hierarchical_layer(features)
        
        # 低层推理循环
        for l_cycle in range(self.l_cycles):
            for layer in self.layers['reasoning_layers'][:self.reasoning_depth]:
                features, attn_weights = layer(features)
                attention_weights.append(attn_weights)
        
        # 全局池化
        features = features.mean(dim=1)  # [batch_size, d_model]
        
        # 输出投影
        output = self.layers['output_projection'](features)  # [batch_size, num_classes]
        
        # 存储注意力权重用于分析
        self._last_attention_weights = attention_weights
        
        return output

# =============================================================================
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: HRMConfig, model: HRMModel, data_loader_manager: HRMDataLoaderManager,
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
        
        # 混合精度训练（只在GPU上启用）
        self.scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() and self.device.type == 'cuda' else None
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available() and self.device.type == 'cuda'
        
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
        
        # 确保损失函数也在正确的设备上
        if hasattr(self.criterion, 'to'):
            self.criterion.to(self.device)
        
        self._log_device_info()
        
        # 记录性能优化设置
        if self.use_amp:
            self.logger.info("✅ 混合精度训练已启用")
        else:
            self.logger.info("ℹ️ 混合精度训练未启用（需要GPU支持）")
        
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
    
    def _check_device_consistency(self, data: torch.Tensor, target: torch.Tensor) -> bool:
        """检查设备一致性"""
        try:
            # 安全地获取模型设备，处理模型参数为空的情况
            model_params = list(self.model.parameters())
            if len(model_params) == 0:
                # 如果模型还没有参数（延迟初始化），使用训练器的设备
                model_device = self.device
            else:
                model_device = next(iter(model_params)).device
            
            data_device = data.device
            target_device = target.device
            
            if model_device != data_device or model_device != target_device:
                self.logger.warning(f"设备不匹配！模型: {model_device}, 数据: {data_device}, 标签: {target_device}")
                return False
            return True
        except Exception as e:
            # 如果检查失败，记录警告但不中断训练
            self.logger.warning(f"设备一致性检查失败: {e}")
            return False
    
    def setup_training_components(self):
        """设置训练组件"""
        # 损失函数（可以提前初始化）
        if self.config.model_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # 确保损失函数在正确的设备上
        self.criterion.to(self.device)
        
        # 优化器和调度器将在模型网络层构建后延迟初始化
        self.optimizer = None
        self.scheduler = None
    
    def _setup_optimizer_and_scheduler(self):
        """延迟初始化优化器和学习率调度器（在模型网络层构建后）"""
        if self.optimizer is not None:
            return  # 已经初始化过了
        
        # 确保模型有参数
        model_params = list(self.model.parameters())
        if len(model_params) == 0:
            raise RuntimeError("模型没有可训练参数，无法创建优化器")
        
        # 优化器
        if self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                model_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                model_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # 学习率调度器
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        
        self.logger.info(f"✅ 优化器和调度器初始化完成，模型参数数量: {len(model_params)}")
        
        # 记录模型信息
        model_info = self.model.get_model_info()
        self.logger.info(f"模型信息: {model_info}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                # 确保数据和目标都在正确的设备上
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # 在第一次前向传播后初始化优化器
                if self.optimizer is None:
                    # 先进行一次前向传播以构建模型网络层
                    with torch.no_grad():
                        # 确保模型和数据在同一设备上
                        test_data = data[:1].to(self.device)
                        _ = self.model(test_data)  # 只用一个样本来触发网络构建
                    # 现在初始化优化器和调度器
                    self._setup_optimizer_and_scheduler()
                    self.logger.info(f"✅ 优化器初始化完成，开始正式训练")
                
                # 检查设备一致性，如果不一致则强制移动
                if not self._check_device_consistency(data, target):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    self.logger.debug(f"已强制移动数据到设备: {self.device}")
                
                self.optimizer.zero_grad()
                
                # 混合精度训练
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    # 检查loss是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"检测到无效loss值: {loss.item()}, 跳过此batch")
                        continue
                    
                    # 混合精度反向传播
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    if hasattr(self.config, 'gradient_clip_value'):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                    
                    # 优化器步进
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准训练
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # 检查loss是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"检测到无效loss值: {loss.item()}, 跳过此batch")
                        continue
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    if hasattr(self.config, 'gradient_clip_value'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                    
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                if self.config.model_type == "classification":
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                
                # 每100个batch记录一次进度
                if batch_idx % 100 == 0:
                    self.logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                self.logger.error(f"训练batch {batch_idx} 时发生错误: {e}")
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 继续下一个batch而不是中断整个训练
                continue
        
        metrics = {"loss": total_loss / len(train_loader) if len(train_loader) > 0 else 0.0}
        
        if self.config.model_type == "classification" and len(all_targets) > 0:
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                try:
                    # 确保数据和目标都在正确的设备上
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    # 检查设备一致性，如果不一致则强制移动
                    if not self._check_device_consistency(data, target):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        self.logger.debug(f"验证时已强制移动数据到设备: {self.device}")
                    
                    # 混合精度推理
                    if self.use_amp:
                        with autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    # 检查loss是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"验证时检测到无效loss值: {loss.item()}, 跳过此batch")
                        continue
                    
                    total_loss += loss.item()
                    
                    if self.config.model_type == "classification":
                        probs = torch.softmax(output, dim=1)
                        preds = torch.argmax(output, dim=1)
                        all_probs.extend(probs.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                        
                except Exception as e:
                    self.logger.error(f"验证batch {batch_idx} 时发生错误: {e}")
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 继续下一个batch
                    continue
        
        metrics = {"loss": total_loss / len(val_loader) if len(val_loader) > 0 else 0.0}
        
        if self.config.model_type == "classification" and len(all_targets) > 0:
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
            metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
            
            # 计算AUC（如果有概率输出）
            if len(all_probs) > 0 and len(np.unique(all_targets)) == 2:
                try:
                    all_probs_array = np.array(all_probs)
                    if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                        metrics["auc"] = roc_auc_score(all_targets, all_probs_array[:, 1])
                        # 添加验证预测数据用于Optuna优化
                        metrics["y_true_val"] = all_targets
                        metrics["y_prob_val"] = all_probs_array[:, 1].tolist()
                except Exception:
                    pass  # 如果无法计算AUC，跳过
        
        return metrics
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """完整训练流程"""
        try:
            # 获取数据加载器
            batch_size = batch_size or self.config.batch_size
            train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(
                batch_size=batch_size)
            
            self.setup_training_components()
            
            self.logger.info("🚀 开始训练...")
            self.logger.info(f"📊 训练配置: epochs={self.config.epochs}, batch_size={batch_size}, lr={self.config.learning_rate}")
            self.logger.info(f"📊 数据集大小: 训练={len(train_loader.dataset)}, 验证={len(val_loader.dataset)}")
            
            start_time = time.time()
            epochs = self.config.epochs
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                try:
                    # 训练
                    self.logger.info(f"🔄 开始训练 Epoch {epoch+1}/{epochs}")
                    train_metrics = self.train_epoch(train_loader)
                    
                    # 验证
                    self.logger.info(f"🔍 开始验证 Epoch {epoch+1}/{epochs}")
                    val_metrics = self.validate_epoch(val_loader)
                    
                    # 记录历史
                    epoch_info = {
                        "epoch": epoch + 1,
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "lr": self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else self.config.learning_rate,
                        "time": time.time() - epoch_start
                    }
                    
                    if self.config.model_type == "classification":
                        epoch_info.update({
                            "train_acc": train_metrics.get("accuracy", 0),
                            "val_acc": val_metrics.get("accuracy", 0),
                            "val_f1": val_metrics.get("f1", 0)
                        })
                        if 'auc' in val_metrics:
                            epoch_info["train_auc"] = 0  # 简化，训练时不计算AUC
                            epoch_info["val_auc"] = val_metrics["auc"]
                        current_score = val_metrics.get("accuracy", 0)
                    else:
                        current_score = -val_metrics["loss"]  # 回归任务使用负损失
                    
                    self.train_history.append(epoch_info)
                    
                    # 学习率调度（确保调度器已初始化）
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(current_score)
                        else:
                            self.scheduler.step()
                    
                    # 保存最佳模型
                    if current_score > self.best_val_score:
                        self.best_val_score = current_score
                        if not no_save_model:
                            self.save_checkpoint(epoch + 1, is_best=True)
                    
                    # GPU内存监控（每10个epoch记录一次）
                    if self.device.type == 'cuda' and (epoch + 1) % 10 == 0:
                        current_memory = torch.cuda.memory_allocated() / 1024**3
                        max_memory = torch.cuda.max_memory_allocated() / 1024**3
                        cached_memory = torch.cuda.memory_reserved() / 1024**3
                        self.logger.info(f"📊 GPU内存使用 (Epoch {epoch+1}): 当前 {current_memory:.2f}GB, 峰值 {max_memory:.2f}GB, 缓存 {cached_memory:.2f}GB")
                    
                    # 日志输出
                    log_msg = (
                        f"✅ Epoch {epoch+1}/{epochs} 完成 - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}"
                    )

                    if self.config.model_type == "classification":
                        log_msg += f", Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                        if 'auc' in val_metrics:
                            log_msg += f", Val AUC: {val_metrics['auc']:.4f}"

                    log_msg += f", Time: {epoch_info['time']:.2f}s"
                    self.logger.info(log_msg)
                    
                except Exception as e:
                    self.logger.error(f"❌ Epoch {epoch+1} 训练失败: {e}")
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 继续下一个epoch而不是完全停止训练
                    continue
            
            total_time = time.time() - start_time
            
            # 保存训练历史
            self.save_training_history()
            
            self.logger.info(f"🎉 训练完成！总时间: {total_time:.2f}s, 最佳验证分数: {self.best_val_score:.4f}")
            
            return {
                "best_val_score": self.best_val_score,
                "total_epochs": len(self.train_history),
                "total_time": total_time,
                "train_history": self.train_history
            }
            
        except Exception as e:
            self.logger.error(f"❌ 训练过程发生严重错误: {e}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_score': self.best_val_score,
            'config': asdict(self.config)
        }
        
        # 只有在优化器和调度器存在时才保存它们的状态
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        # 使用传入的checkpoint目录（优先），否则使用配置中的
        checkpoint_dir = Path(self.checkpoint_dir) if self.checkpoint_dir else Path(self.config.checkpoint_dir)
        
        checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
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
    
    def __init__(self, config: HRMConfig, model: HRMModel, data_loader_manager: HRMDataLoaderManager):
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
                # 确保数据在正确的设备上
                data = data.to(self.device, non_blocking=True)
                outputs = self.model(data)
                
                if self.config.model_type == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                else:
                    all_preds.extend(outputs.cpu().numpy())
                    all_probs.extend(outputs.cpu().numpy())  # 回归任务概率即为预测值
        
        return np.array(all_preds), np.array(all_probs)
    
    def predict_single(self, data: np.ndarray) -> Tuple[Union[int, float], np.ndarray]:
        """单样本推理"""
        data_tensor = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(data_tensor)
            
            if self.config.model_type == "classification":
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                return pred, probs.cpu().numpy()[0]
            else:
                pred = output.item()
                return pred, np.array([pred])
    
    def evaluate(self) -> Dict[str, float]:
        """模型评估"""
        # 获取测试数据加载器
        _, _, test_loader = self.data_loader_manager.load_data_loaders()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                # 确保数据和目标在正确的设备上
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                outputs = self.model(data)
                
                if self.config.model_type == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                else:
                    all_preds.extend(outputs.cpu().numpy())
                
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        metrics = {}
        
        if self.config.model_type == "classification":
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
            metrics["precision"] = precision_score(all_targets, all_preds, average='weighted')
            metrics["recall"] = recall_score(all_targets, all_preds, average='weighted')
            metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
            
            # AUC指标计算
            if len(np.unique(all_targets)) == 2:  # 二分类
                all_probs_positive = np.array(all_probs)[:, 1]
                metrics["auc"] = roc_auc_score(all_targets, all_probs_positive)
            elif len(np.unique(all_targets)) > 2:  # 多分类
                try:
                    metrics["auc"] = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
                except ValueError:
                    # 如果无法计算多分类AUC，跳过
                    pass
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics["mse"] = mean_squared_error(all_targets, all_preds)
            metrics["mae"] = mean_absolute_error(all_targets, all_preds)
            metrics["rmse"] = np.sqrt(metrics["mse"])
        
        return metrics
    
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
    
    def __init__(self, config: HRMConfig, model: HRMModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个基于深度学习的{self.config.model_type}模型，采用层级推理机制。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB

## 模型架构

### 核心组件

- **特征提取器**: 将输入特征投影到模型维度
- **层级推理模块**: 多层次的推理机制
- **注意力机制**: 层级注意力机制
- **输出投影**: 最终分类/回归输出

### 网络结构

```
输入特征 -> 特征提取器 -> 层级推理循环 -> 输出投影 -> 预测结果
```

## 技术原理

HRM模型采用层级推理机制，通过多个推理循环和注意力机制来处理复杂的特征关系。

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}
- 优化器: {self.config.optimizer}

### 模型配置
- 模型维度: {self.config.d_model}
- 注意力头数: {self.config.n_heads}
- 层级数量: {self.config.hierarchical_levels}
- 推理深度: {self.config.reasoning_depth}

## 使用方法

### 训练模型

```python
python HRM_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```python
python HRM_unified.py inference --checkpoint best_model.pth
```

### 模型评估

```python
python HRM_unified.py inference --checkpoint best_model.pth --evaluate
```

## 性能指标

模型支持以下评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- AUC (Area Under Curve)

## 注意事项

- 模型支持动态输入维度适配
- 支持混合精度训练以提升性能
- 支持GPU加速训练和推理

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# 主要接口函数
# =============================================================================

def create_model_factory(config: HRMConfig) -> HRMModel:
    """模型工厂函数"""
    return HRMModel(config)

def create_data_loader_manager(data_config_path: str, config: HRMConfig) -> HRMDataLoaderManager:
    """数据加载器管理器工厂函数"""
    return HRMDataLoaderManager(data_config_path, config)

def main_train(config_path: str = "config.yaml", data_config_path: str = "data.yaml",
               optuna_config_path: str = None, device: str = "auto",
               epochs_override: int = None, no_save_model: bool = False,
               seed: int = 42, checkpoint_dir: str = "checkpoints"):
    """主训练函数"""
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
    
    # 1. 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        config_dict = {}
    
    # 处理architecture部分
    if 'architecture' in config_dict:
        arch = config_dict['architecture']
        # 将architecture中的参数提取到顶层
        for key, value in arch.items():
            if hasattr(HRMConfig, key):
                config_dict[key] = value
        # 移除architecture部分
        del config_dict['architecture']
    
    # 移除其他不属于HRMConfig的部分
    valid_keys = {field.name for field in HRMConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    # 创建配置对象
    config = HRMConfig(**filtered_config)
    
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
    
    # 2. 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 3. 创建模型
    model = create_model_factory(config)
    
    # 4. 创建训练器（传递checkpoint目录）
    trainer = UnifiedTrainer(config, model, data_loader_manager, checkpoint_dir=checkpoint_dir)
    
    # 5. 执行训练
    results = trainer.train(no_save_model=no_save_model)
    
    # 6. 输出JSON格式结果供Optuna解析
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
                  checkpoint_path: str = "best_model.pth"):
    """主推理函数"""
    # 1. 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        config_dict = {}
    
    # 处理architecture部分
    if 'architecture' in config_dict:
        arch = config_dict['architecture']
        # 将architecture中的参数提取到顶层
        for key, value in arch.items():
            if hasattr(HRMConfig, key):
                config_dict[key] = value
        # 移除architecture部分
        del config_dict['architecture']
    
    # 移除其他不属于HRMConfig的部分
    valid_keys = {field.name for field in HRMConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    config = HRMConfig(**filtered_config)
    
    # 2. 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 3. 创建模型
    model = create_model_factory(config)
    
    # 4. 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 5. 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = inferencer.load_checkpoint(checkpoint_path)
    else:
        print(f"⚠️ 检查点文件 {checkpoint_path} 不存在，使用随机初始化模型")
    
    # 6. 执行推理和评估
    metrics = inferencer.evaluate()
    
    # 7. 输出结果
    print("评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    # 1. 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        config_dict = {}
    
    # 处理architecture部分
    if 'architecture' in config_dict:
        arch = config_dict['architecture']
        # 将architecture中的参数提取到顶层
        for key, value in arch.items():
            if hasattr(HRMConfig, key):
                config_dict[key] = value
        # 移除architecture部分
        del config_dict['architecture']
    
    # 移除其他不属于HRMConfig的部分
    valid_keys = {field.name for field in HRMConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    config = HRMConfig(**filtered_config)
    
    # 2. 创建模型
    model = create_model_factory(config)
    
    # 3. 生成文档
    doc_generator = ModelDocumentationGenerator(config, model)
    doc_generator.generate_model_md("MODEL.md")

# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="HRM统一模型训练、推理和文档生成工具")
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
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/HRM_5021096712"
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