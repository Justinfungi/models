#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TKAN_unified.py - 统一TKAN模型实现
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
import math
import random

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
class TKANConfig:
    """TKAN模型配置类"""
    # 模型基础配置
    model_name: str = "TKAN"
    model_type: str = "classification"  # classification, regression, time_series
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # 性能优化配置
    use_mixed_precision: bool = True  # 混合精度训练
    dataloader_num_workers: int = 4  # 数据加载并发数
    pin_memory: bool = True  # 固定内存
    gradient_clip_value: float = 1.0  # 梯度裁剪
    
    # TKAN特定配置
    units: int = 128
    return_sequences: bool = False
    stateful: bool = False
    dropout: float = 0.2
    recurrent_dropout: float = 0.1
    use_bias: bool = True
    num_layers: int = 2
    hidden_dims: List[int] = None
    activation: str = 'tanh'
    kan_sublayers: int = 4
    grid_size: int = 5
    spline_order: int = 3
    output_dim: int = 1
    output_activation: str = 'sigmoid'
    
    # 数据配置
    input_dim: int = None  # 动态设置
    seq_len: int = 96
    pred_len: int = 24
    
    # 设备配置
    device: str = "auto"
    seed: int = 42
    
    # 路径配置
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]
        
        # 验证hidden_dims配置
        if not isinstance(self.hidden_dims, list) or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims必须是非空列表")
        
        for i, dim in enumerate(self.hidden_dims):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"hidden_dims[{i}]必须是正整数，当前值: {dim}")
        
        # 验证其他关键参数
        if self.output_dim <= 0:
            raise ValueError(f"output_dim必须是正整数，当前值: {self.output_dim}")
        
        if self.grid_size <= 0:
            raise ValueError(f"grid_size必须是正整数，当前值: {self.grid_size}")
        
        if self.spline_order <= 0:
            raise ValueError(f"spline_order必须是正整数，当前值: {self.spline_order}")

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
    """设置日志系统
    
    Args:
        log_dir: 日志目录路径
        prefix: 日志文件前缀（当log_filename为None时使用）
        log_filename: 固定的日志文件名（如果提供，将忽略prefix和时间戳）
    """
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

def create_directories(config: TKANConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# TKAN模型架构实现
# =============================================================================

class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network 线性层实现"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: str = 'silu',
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1, 1)
    ):
        """KAN 线性层初始化"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        
        # 创建网格
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        self.register_buffer('grid', grid)
        
        # 初始化参数
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * scale_base)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * scale_spline
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.randn(out_features, in_features) * scale_spline)
        
        # 基础激活函数
        if base_activation == 'silu':
            self.base_activation_fn = F.silu
        elif base_activation == 'relu':
            self.base_activation_fn = F.relu
        elif base_activation == 'tanh':
            self.base_activation_fn = torch.tanh
        else:
            self.base_activation_fn = F.silu
    
    def b_splines(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """B-spline 基函数计算"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # 确保grid和x在同一设备上
        device = x.device
        if grid.device != device:
            grid = grid.to(device)
        
        batch_size = x.size(0)
        
        # 计算每个输入点在网格上的位置
        x_expanded = x.unsqueeze(-1)  # [batch_size, in_features, 1]
        grid_expanded = grid.unsqueeze(0).unsqueeze(0).expand(batch_size, self.in_features, -1)  # [batch_size, in_features, grid_size+1]
        
        # 初始化B-spline基函数 (0阶)
        bases = torch.zeros(batch_size, self.in_features, self.grid_size + self.spline_order, device=device, dtype=x.dtype)
        
        # 找到每个点所在的区间
        for i in range(self.grid_size):
            if i + 1 < grid_expanded.size(-1):
                mask = (x_expanded >= grid_expanded[:, :, i:i+1]) & (x_expanded < grid_expanded[:, :, i+1:i+2])
                bases[:, :, i] = mask.squeeze(-1).float()
        
        # 递归计算高阶B-spline
        for k in range(1, self.spline_order + 1):
            new_bases = torch.zeros_like(bases, device=device, dtype=x.dtype)
            for i in range(bases.size(-1) - k):
                if i + k < grid.size(0):
                    # 左侧项
                    denom1 = grid[i + k] - grid[i] + 1e-8
                    alpha1 = (x - grid[i].to(device)) / denom1.to(device)
                    new_bases[:, :, i] += alpha1 * bases[:, :, i]
                    
                    # 右侧项
                    if i + k + 1 < grid.size(0):
                        denom2 = grid[i + k + 1] - grid[i + 1] + 1e-8
                        alpha2 = (grid[i + k + 1].to(device) - x) / denom2.to(device)
                        new_bases[:, :, i] += alpha2 * bases[:, :, i + 1]
            bases = new_bases
        
        # 返回正确维度的基函数，确保与spline_weight兼容
        return bases[:, :, :self.grid_size + self.spline_order]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # 确保所有张量在同一设备上
        device = x.device
        
        # 确保所有模型参数都在正确的设备上
        if self.base_weight.device != device:
            self.base_weight = self.base_weight.to(device)
        if self.spline_weight.device != device:
            self.spline_weight = self.spline_weight.to(device)
        if hasattr(self, 'spline_scaler') and self.spline_scaler.device != device:
            self.spline_scaler = self.spline_scaler.to(device)
        
        # 基础变换
        base_output = F.linear(self.base_activation_fn(x), self.base_weight)
        
        # B-spline变换
        grid = self.grid.to(device)
        spline_basis = self.b_splines(x, grid)
        spline_output = torch.einsum('bik,oik->bo', spline_basis, self.spline_weight)
        
        if self.enable_standalone_scale_spline:
            spline_output = spline_output * self.spline_scaler.sum(dim=1, keepdim=True).T
        
        return base_output + spline_output

class TKANCell(nn.Module):
    """TKAN 单元格实现"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kan_degree: int = 3,
        kan_grid_size: int = 5,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        use_bias: bool = True
    ):
        """TKAN 单元格初始化"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        
        # KAN层用于输入变换
        self.input_kan = KANLinear(
            input_size, 
            hidden_size * 4,  # 4个门：forget, input, candidate, output
            grid_size=kan_grid_size,
            spline_order=kan_degree
        )
        
        # KAN层用于隐藏状态变换
        self.hidden_kan = KANLinear(
            hidden_size,
            hidden_size * 4,
            grid_size=kan_grid_size,
            spline_order=kan_degree
        )
        
        # 偏置项
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size * 4))
        else:
            self.register_parameter('bias', None)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        self.recurrent_dropout_layer = nn.Dropout(recurrent_dropout)
    
    def forward(self, input: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """单步前向传播"""
        batch_size = input.size(0)
        device = input.device
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=input.dtype)
        else:
            # 确保hidden在正确的设备上
            if hidden.device != device:
                hidden = hidden.to(device)
        
        # 应用dropout
        input = self.dropout_layer(input)
        hidden = self.recurrent_dropout_layer(hidden)
        
        # KAN变换
        gi = self.input_kan(input)
        gh = self.hidden_kan(hidden)
        
        # 添加偏置
        if self.bias is not None:
            if self.bias.device != device:
                self.bias = self.bias.to(device)
            gi = gi + self.bias
        
        # 计算门控值
        gates = gi + gh
        forget_gate, input_gate, candidate_gate, output_gate = gates.chunk(4, dim=1)
        
        # 应用激活函数
        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        candidate_gate = torch.tanh(candidate_gate)
        output_gate = torch.sigmoid(output_gate)
        
        # 更新细胞状态和隐藏状态
        new_hidden = forget_gate * hidden + input_gate * candidate_gate
        output = output_gate * torch.tanh(new_hidden)
        
        return output

class TKANLayer(nn.Module):
    """TKAN 层实现"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        return_sequences: bool = False,
        stateful: bool = False,
        **kwargs
    ):
        """TKAN 层初始化"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.stateful = stateful
        
        self.cell = TKANCell(input_size, hidden_size, **kwargs)
        
        # 状态存储（用于stateful模式）
        self.register_buffer('state', None)
    
    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """序列前向传播"""
        batch_size, seq_len, input_size = x.size()
        device = x.device
        
        # 初始化状态
        if initial_state is not None:
            hidden = initial_state.to(device) if initial_state.device != device else initial_state
        elif self.stateful and self.state is not None:
            hidden = self.state.to(device) if self.state.device != device else self.state
        else:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            hidden = self.cell(x[:, t], hidden)
            if self.return_sequences:
                outputs.append(hidden)
        
        # 保存状态（用于stateful模式）
        if self.stateful:
            self.state = hidden.detach()
        
        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return hidden
    
    def reset_states(self):
        """重置状态（用于 stateful 模式）"""
        self.state = None

# =============================================================================
# 模型基类
# =============================================================================

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, config: TKANConfig):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 子类必须实现"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        except Exception:
            # 如果模型还未构建，返回默认值
            total_params = 0
            trainable_params = 0
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024) if total_params > 0 else 0.0  # 假设float32
        }

class TKANModel(BaseModel):
    """TKAN 主模型类"""
    
    def __init__(self, config: TKANConfig):
        super().__init__(config)
        
        # 输入维度（延迟初始化）
        self.input_dim = getattr(config, 'input_dim', None)
        
        # 模型组件（延迟初始化）
        self.tkan_layers = None
        self.output_layer = None
        self.dropout = nn.Dropout(config.dropout)
        self.batch_norm = None
        
        # 如果配置中没有input_dim，必须在训练时动态设置
        if self.input_dim is None:
            print("⚠️ 配置中未指定input_dim，将在数据加载时自动检测")
    
    def _build_layers(self, input_dim: int):
        """🔥 使用动态输入维度构建网络，绝对不能硬编码"""
        if input_dim <= 0:
            raise ValueError(f"❌ 输入维度必须大于0，当前值: {input_dim}")
        
        print(f"🔧 构建TKAN网络层，输入维度: {input_dim}")
        
        # 更新配置
        self.config.input_dim = input_dim
        self.input_dim = input_dim
        
        # 验证隐藏层维度配置
        if not self.config.hidden_dims or len(self.config.hidden_dims) == 0:
            raise ValueError("❌ hidden_dims配置不能为空")
        
        # 构建TKAN层
        self.tkan_layers = nn.ModuleList()
        layer_input_size = input_dim
        
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            if hidden_dim <= 0:
                raise ValueError(f"❌ 隐藏层维度必须大于0，第{i}层维度: {hidden_dim}")
            
            return_sequences = (i < len(self.config.hidden_dims) - 1) or self.config.return_sequences
            
            print(f"  🔧 构建第{i+1}层TKAN: {layer_input_size} -> {hidden_dim}, return_sequences={return_sequences}")
            
            tkan_layer = TKANLayer(
                input_size=layer_input_size,
                hidden_size=hidden_dim,
                return_sequences=return_sequences,
                stateful=self.config.stateful,
                kan_degree=self.config.spline_order,
                kan_grid_size=self.config.grid_size,
                dropout=self.config.dropout,
                recurrent_dropout=self.config.recurrent_dropout,
                use_bias=self.config.use_bias
            )
            self.tkan_layers.append(tkan_layer)
            layer_input_size = hidden_dim
        
        # 输出层
        final_hidden_size = self.config.hidden_dims[-1] if self.config.hidden_dims else self.config.units
        print(f"  🔧 构建输出层: {final_hidden_size} -> {self.config.output_dim}")
        self.output_layer = nn.Linear(final_hidden_size, self.config.output_dim)
        
        # 批量归一化
        self.batch_norm = nn.BatchNorm1d(final_hidden_size)
        
        print(f"✅ TKAN网络层构建完成，共{len(self.tkan_layers)}层TKAN + 1层输出")
        
        # 验证模型参数
        total_params = sum(p.numel() for p in self.parameters())
        if total_params == 0:
            raise RuntimeError("❌ 模型构建后参数数量为0，请检查网络结构")
        print(f"✅ 模型参数总数: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        device = x.device
        
        # 首次前向传播时自动检测并构建网络
        if self.tkan_layers is None:
            if self.input_dim is None:
                if x.dim() == 2:
                    self.input_dim = x.shape[-1]  # 从输入数据自动检测
                elif x.dim() == 3:
                    self.input_dim = x.shape[-1]
                else:
                    raise ValueError(f"不支持的输入维度: {x.shape}")
                print(f"🔍 自动检测输入维度: {self.input_dim}")
            self._build_layers(self.input_dim)
            # 确保新构建的层在正确的设备上
            self.to(device)
        
        # 确保所有模型组件在正确的设备上
        if self.tkan_layers is not None:
            for layer in self.tkan_layers:
                if next(layer.parameters()).device != device:
                    layer.to(device)
        
        if self.output_layer is not None and next(self.output_layer.parameters()).device != device:
            self.output_layer.to(device)
        
        if self.batch_norm is not None and next(self.batch_norm.parameters()).device != device:
            self.batch_norm.to(device)
        
        # 确保输入是3D (batch_size, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 通过TKAN层
        for tkan_layer in self.tkan_layers:
            x = tkan_layer(x)
            if x.dim() == 3:
                # 如果返回序列，只对最后一个时间步应用dropout
                x = self.dropout(x)
            else:
                x = self.dropout(x)
        
        # 如果是3D输出，取最后一个时间步
        if x.dim() == 3:
            x = x[:, -1, :]
        
        # 批量归一化
        x = self.batch_norm(x)
        
        # 输出层
        x = self.output_layer(x)
        
        # 输出激活 - 分类任务使用logits输出，推理时再应用sigmoid
        if self.config.model_type == "classification":
            # 训练时返回logits，推理时需要手动应用sigmoid
            return x
        elif self.config.output_activation == 'softmax':
            x = F.softmax(x, dim=-1)
            return x
        else:
            return x

# =============================================================================
# 数据处理类
# =============================================================================

class TKANDataLoaderManager:
    """TKAN数据加载器管理类 - 只支持真实数据"""
    
    def __init__(self, data_config_path: str, config: TKANConfig):
        """
        初始化数据加载器管理器
        
        Args:
            data_config_path: 数据配置文件路径，包含data_split信息
            config: TKAN配置对象
        """
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
        feature_identifier = self.data_config.get('data_format', {}).get('feature_identifier', '@')
        feature_cols = [col for col in df.columns if feature_identifier in col]
        
        if not feature_cols:
            # 如果没有@符号的列，假设除了'class'列外都是特征
            feature_cols = [col for col in df.columns if col != 'class']
        
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
    
    def _load_real_dataframes(self):
        """🚫 严格禁止：不得包含任何模拟数据生成代码"""
        try:
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
            print(f"📊 加载数据文件: {data_file}")
            
            # 读取Parquet文件
            df = pd.read_parquet(data_file)
            print(f"📊 数据形状: {df.shape}")
            
            return self._split_dataframes(df)
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def _split_dataframes(self, df):
        """时间序列数据分割"""
        # 识别特征列（包含@符号的列）
        feature_identifier = self.data_config.get('data_format', {}).get('feature_identifier', '@')
        feature_cols = [col for col in df.columns if feature_identifier in col]
        
        if not feature_cols:
            raise ValueError(f"未找到包含 '{feature_identifier}' 的特征列")
        
        print(f"🔍 识别到 {len(feature_cols)} 个特征列")
        
        # 获取特征和标签
        X = df[feature_cols].values.astype(np.float32)
        y = df['class'].values.astype(np.float32)
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 时间序列分割
        if 'date' in df.columns:
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
        
        print(f"📊 训练集大小: {len(X_train)}")
        print(f"📊 验证集大小: {len(X_val)}")
        print(f"📊 测试集大小: {len(X_test)}")
        
        # 创建DataFrame
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df['class'] = y_train
        
        val_df = pd.DataFrame(X_val, columns=feature_cols)
        val_df['class'] = y_val
        
        test_df = pd.DataFrame(X_test, columns=feature_cols)
        test_df['class'] = y_test
        
        return train_df, val_df, test_df
    
    def _create_dataloader(self, dataset, shuffle: bool = False, batch_size: int = 32) -> DataLoader:
        """创建优化的数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,        # 数据加载优化
            pin_memory=True,      # GPU内存优化
            persistent_workers=True,  # 数据加载优化
            prefetch_factor=2     # 数据加载优化
        )
    
    def load_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        🔥 必须从真实数据文件加载，严格禁止模拟数据
        """
        # 加载真实数据
        train_df, val_df, test_df = self._load_real_dataframes()
        
        # 🔥 关键：动态检测特征维度
        feature_cols = [col for col in train_df.columns if col != 'class']
        input_dim = len(feature_cols)
        print(f"🔍 检测到 {input_dim} 个特征: {feature_cols[:5]}...")
        
        # 更新配置
        self.config.input_dim = input_dim
        
        # 准备数据
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['class'].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df['class'].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df['class'].values.astype(np.float32)
        
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
        
        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 创建DataLoader
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
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: TKANConfig, model: TKANModel, data_loader_manager: TKANDataLoaderManager,
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
        
        # 确保模型的所有子模块都在正确的设备上
        for module in self.model.modules():
            if hasattr(module, 'to'):
                module.to(self.device)
        
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
    
    def setup_training_components(self, train_loader: DataLoader = None):
        """设置训练组件"""
        # 如果模型还没有构建层，先进行一次前向传播来构建
        if self.model.tkan_layers is None and train_loader is not None:
            print("🔧 模型层未构建，进行初始化前向传播...")
            self.model.train()
            with torch.no_grad():
                # 获取一个批次的数据来初始化模型
                for batch_data, _ in train_loader:
                    batch_data = batch_data.to(self.device)
                    _ = self.model(batch_data)  # 触发模型层构建
                    break
            print("✅ 模型层构建完成")
        
        # 验证模型参数是否存在
        model_params = list(self.model.parameters())
        if len(model_params) == 0:
            raise RuntimeError("❌ 模型参数为空，无法初始化优化器。请检查模型构建逻辑。")
        
        print(f"✅ 模型参数数量: {len(model_params)}")
        total_params = sum(p.numel() for p in model_params)
        print(f"✅ 总参数数量: {total_params:,}")
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        # 损失函数 - 使用BCEWithLogitsLoss避免混合精度训练问题
        if self.config.model_type == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target)
                
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
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                
                # 梯度裁剪
                if hasattr(self.config, 'gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.config.model_type == "classification":
                # 对logits应用sigmoid后再进行预测
                probs = torch.sigmoid(output.squeeze())
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        metrics = {"loss": total_loss / len(train_loader)}
        
        if self.config.model_type == "classification":
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
            for data, target in val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 混合精度推理
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output.squeeze(), target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                
                if self.config.model_type == "classification":
                    # 对logits应用sigmoid得到概率
                    probs = torch.sigmoid(output.squeeze())
                    preds = (probs > 0.5).float()
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        metrics = {"loss": total_loss / len(val_loader)}
        
        if self.config.model_type == "classification":
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
            metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
            
            # 计算AUC（如果有概率输出）
            if len(all_probs) > 0 and len(np.unique(all_targets)) == 2:
                try:
                    metrics["auc"] = roc_auc_score(all_targets, all_probs)
                    # 添加验证预测数据用于Optuna优化
                    metrics["y_true_val"] = all_targets
                    metrics["y_prob_val"] = all_probs
                except Exception:
                    pass  # 如果无法计算AUC，跳过
        
        return metrics
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """🔄 完整训练: 训练将运行完整的epochs数量，🚫 严格禁止快速测试模式"""
        # 获取数据加载器
        batch_size = batch_size or self.config.batch_size
        train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(
            batch_size=batch_size)
        
        # 传入训练数据加载器来设置训练组件
        self.setup_training_components(train_loader)
        
        self.logger.info("开始训练...")
        self.logger.info(f"模型信息: {self.model.get_model_info()}")
        
        start_time = time.time()
        epochs = self.config.epochs  # 🚫 严格禁止快速测试，使用完整epochs
        
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
            
            if self.config.model_type == "classification":
                epoch_info.update({
                    "train_acc": train_metrics.get("accuracy", 0),
                    "val_acc": val_metrics.get("accuracy", 0),
                    "val_f1": val_metrics.get("f1", 0),
                    "train_auc": train_metrics.get("auc", 0),
                    "val_auc": val_metrics.get("auc", 0)
                })
                current_score = val_metrics.get("accuracy", 0)
            else:
                current_score = -val_metrics["loss"]  # 回归任务使用负损失
            
            self.train_history.append(epoch_info)
            
            # 学习率调度
            self.scheduler.step(current_score)
            
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
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )

            if self.config.model_type == "classification":
                log_msg += f", Val Acc: {val_metrics.get('accuracy', 0):.4f}"
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
    
    def __init__(self, config: TKANConfig, model: TKANModel, data_loader_manager: TKANDataLoaderManager,
                 checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.model = model
        self.data_loader_manager = data_loader_manager
        self.checkpoint_dir = checkpoint_dir  # Checkpoint保存目录
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
                
                if self.config.model_type == "classification":
                    # 对logits应用sigmoid得到概率
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > 0.5).float()
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
                # 对logits应用sigmoid得到概率
                prob = torch.sigmoid(output.squeeze())
                pred = (prob > 0.5).float().item()
                return pred, np.array([prob.cpu().numpy()])
            else:
                pred = output.item()
                return pred, np.array([pred])
    
    def evaluate(self) -> Dict[str, float]:
        """📊 必须包含AUC: 在模型评估中必须包含AUC指标计算，特别是二分类任务"""
        # 获取测试数据加载器
        _, _, test_loader = self.data_loader_manager.load_data_loaders()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                if self.config.model_type == "classification":
                    # 对logits应用sigmoid得到概率
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > 0.5).float()
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
            
            # 📊 必须包含AUC指标计算
            if len(np.unique(all_targets)) == 2:  # 二分类
                metrics["auc"] = roc_auc_score(all_targets, all_probs)
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
    
    def __init__(self, config: TKANConfig, model: TKANModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个基于Time-series Kolmogorov-Arnold Networks的{self.config.model_type}模型。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB

## 模型架构

### 核心组件

TKAN模型结合了Kolmogorov-Arnold Networks和时间序列处理能力，具有以下特点：

- **KAN线性层**: 使用B-spline基函数进行非线性变换
- **TKAN单元格**: 基于KAN的循环神经网络单元
- **多层架构**: 支持多层TKAN层堆叠
- **动态维度适配**: 自动适应不同的输入特征维度

### 网络结构

```
输入层 -> TKAN层1 -> TKAN层2 -> ... -> 批量归一化 -> 输出层 -> 激活函数
```

## 技术原理

TKAN模型基于Kolmogorov-Arnold表示定理，使用可学习的单变量函数替代传统的线性变换，
能够更好地捕捉数据中的非线性关系和时间依赖性。

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}
- 混合精度训练: {self.config.use_mixed_precision}

### 模型配置
- 隐藏维度: {self.config.hidden_dims}
- Dropout: {self.config.dropout}
- 网格大小: {self.config.grid_size}
- 样条阶数: {self.config.spline_order}

## 使用方法

### 训练模型

```python
python TKAN_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```python
python TKAN_unified.py inference --config config.yaml --data-config data.yaml --checkpoint best_model.pth
```

### 模型评估

```python
python TKAN_unified.py inference --config config.yaml --data-config data.yaml --checkpoint best_model.pth
```

## 性能指标

模型支持以下评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- AUC-ROC (二分类任务)

## 注意事项

1. 模型会自动检测输入特征维度，无需手动配置
2. 支持GPU加速和混合精度训练
3. 建议使用时间序列数据进行训练
4. 模型输出经过sigmoid激活，适用于二分类任务

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}
- 支持动态维度适配
- 集成Optuna超参数优化
- 添加混合精度训练支持

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# 主要接口函数
# =============================================================================

def create_model_factory(config: TKANConfig) -> TKANModel:
    """模型工厂函数"""
    return TKANModel(config)

def create_data_loader_manager(data_config_path: str, config: TKANConfig) -> TKANDataLoaderManager:
    """数据加载器管理器工厂函数"""
    return TKANDataLoaderManager(data_config_path, config)

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
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建配置对象
    config = TKANConfig(**config_dict.get('model', {}))
    
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
    
    # 4. 创建训练器
    trainer = UnifiedTrainer(config, model, data_loader_manager, checkpoint_dir=checkpoint_dir)
    
    # 5. 执行训练
    results = trainer.train(no_save_model=no_save_model)
    
    # 6. 输出JSON格式的训练结果供Optuna解析
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
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = TKANConfig(**config_dict.get('model', {}))
    
    # 2. 创建数据加载器管理器
    data_loader_manager = create_data_loader_manager(data_config_path, config)
    
    # 3. 创建模型
    model = create_model_factory(config)
    
    # 4. 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 5. 加载检查点
    checkpoint = inferencer.load_checkpoint(checkpoint_path)
    
    # 6. 执行推理和评估
    metrics = inferencer.evaluate()
    
    # 7. 输出结果
    print("推理评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = TKANConfig(**config_dict.get('model', {}))
    
    # 2. 创建模型
    model = create_model_factory(config)
    
    # 3. 生成文档
    doc_generator = ModelDocumentationGenerator(config, model)
    doc_generator.generate_model_md("MODEL.md")
    
    print("文档生成完成")

# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="统一TKAN模型训练、推理和文档生成工具")
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
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/TKAN_9235352221"
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