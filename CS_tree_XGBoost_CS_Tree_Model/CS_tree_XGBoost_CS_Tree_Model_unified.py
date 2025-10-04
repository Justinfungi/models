#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS_tree_XGBoost_CS_Tree_Model_unified.py - 统一模型实现
整合训练、推理和文档生成功能的统一模板，支持XGBoost CS Tree模型架构
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

warnings.filterwarnings('ignore')

# =============================================================================
# 基础配置和工具函数
# =============================================================================

@dataclass
class CS_tree_XGBoost_CS_Tree_ModelConfig:
    """CS_tree_XGBoost_CS_Tree_Model配置类"""
    # 模型基础配置
    model_name: str = "CS_tree_XGBoost_CS_Tree_Model"
    model_type: str = "classification"  # classification, regression, time_series
    
    # 训练配置
    learning_rate: float = 0.1
    batch_size: int = 32  # 确保batch size足够大避免BatchNorm问题
    epochs: int = 100
    
    # XGBoost特定配置
    n_estimators: int = 100
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: int = 1
    random_state: int = 42
    n_jobs: int = -1
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "auto"
    verbosity: int = 1
    early_stopping_rounds: int = 10
    
    # 性能优化配置
    use_mixed_precision: bool = True  # 混合精度训练
    dataloader_num_workers: int = 4  # 数据加载并发数
    pin_memory: bool = True  # 固定内存
    gradient_clip_value: float = 1.0  # 梯度裁剪
    
    # 数据配置
    seq_len: int = 96
    pred_len: int = 24
    input_dim: Optional[int] = None  # 动态设置
    output_dim: int = 2  # 二分类
    hidden_dim: int = 128
    dropout: float = 0.2
    
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

def create_directories(config: CS_tree_XGBoost_CS_Tree_ModelConfig) -> None:
    """创建必要的目录结构"""
    dirs = [config.checkpoint_dir, config.results_dir, config.logs_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 数据处理基类
# =============================================================================

class CS_tree_XGBoost_CS_Tree_ModelDataLoaderManager:
    """CS_tree_XGBoost_CS_Tree_Model数据加载器管理类"""
    
    def __init__(self, data_config_path: str, config: CS_tree_XGBoost_CS_Tree_ModelConfig):
        """
        初始化数据加载器管理器
        
        Args:
            data_config_path: 数据配置文件路径，包含data_split信息
            config: 模型配置
        """
        self.data_config_path = data_config_path
        self.config = config
        self.data_config = self._load_data_config()
        self.data_split_info = self.data_config.get('data_split', {})
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
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
    
    def get_data_split_info(self) -> Dict[str, Any]:
        """获取数据分割信息"""
        return self.data_split_info
    
    def _create_dataloader(self, dataset, shuffle: bool = False, batch_size: int = 32) -> DataLoader:
        """创建优化的数据加载器，确保batch size不为1以避免BatchNorm问题"""
        # 确保batch size至少为2，避免BatchNorm在单样本时出错
        actual_batch_size = max(batch_size, 2)
        
        # 如果数据集大小小于batch_size，调整batch_size
        if len(dataset) < actual_batch_size:
            actual_batch_size = max(len(dataset), 1)
            print(f"⚠️ 数据集大小({len(dataset)})小于批次大小，调整为: {actual_batch_size}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=shuffle,
            num_workers=4,  # 数据加载优化
            pin_memory=True,  # GPU内存优化
            persistent_workers=True,  # 数据加载优化
            prefetch_factor=2,  # 数据加载优化
            drop_last=False  # 不丢弃最后一个不完整的batch
        )
        return dataloader

    def _load_parquet_data(self, file_path: str) -> pd.DataFrame:
        """加载parquet数据文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        df = pd.read_parquet(file_path)
        print(f"📊 加载数据文件: {file_path}")
        print(f"   数据形状: {df.shape}")
        print(f"   特征列数: {df.shape[1] - 1}")  # 减去标签列
        
        return df

    def get_input_dim_from_dataframe(self, df: pd.DataFrame) -> int:
        """从DataFrame获取特征维度（排除标签列和非数值列）"""
        # 排除非数值列和标签列
        exclude_cols = ['symbol', 'date', 'time', 'code', 'fut_code', 'exchange', 'industry_name', 'class']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = len(feature_cols)
        print(f"🔍 检测到输入特征维度: {features}")
        return features

    def validate_input_dimensions(self, config: CS_tree_XGBoost_CS_Tree_ModelConfig, actual_input_dim: int) -> int:
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
        """
        加载真实数据并创建数据加载器

        Args:
            batch_size: 批次大小

        Returns:
            train_loader, validation_loader, test_loader
        """
        # 获取数据文件路径
        data_split = self.get_data_split_info()
        
        # 构建数据文件的完整路径
        base_data_path = "/home/feng.hao.jie/deployment/model_explorer/b_model_reproduction_agent/data/feature_set"
        
        train_file = os.path.join(base_data_path, data_split['train']['file'])
        valid_file = os.path.join(base_data_path, data_split['validation']['file'])
        test_file = os.path.join(base_data_path, data_split['test']['file'])
        
        # 加载数据
        train_df = self._load_parquet_data(train_file)
        valid_df = self._load_parquet_data(valid_file)
        test_df = self._load_parquet_data(test_file)
        
        # 动态检测特征维度
        input_dim = self.get_input_dim_from_dataframe(train_df)
        self.validate_input_dimensions(self.config, input_dim)
        
        # 分离特征和标签 - 只使用数值列，排除标签列
        exclude_cols = ['symbol', 'date', 'time', 'code', 'fut_code', 'exchange', 'industry_name', 'class']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        print(f"🔍 特征列: {len(feature_cols)} 个")
        print(f"🔍 排除列: {exclude_cols}")
        
        X_train = train_df[feature_cols].values
        y_train = train_df['class'].values
        
        X_valid = valid_df[feature_cols].values
        y_valid = valid_df['class'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['class'].values
        
        # 数据预处理 - 添加NaN检查和处理
        print(f"🔍 数据预处理前检查:")
        print(f"   训练集NaN数量: {np.isnan(X_train).sum()}")
        print(f"   验证集NaN数量: {np.isnan(X_valid).sum()}")
        print(f"   测试集NaN数量: {np.isnan(X_test).sum()}")
        
        # 处理NaN值 - 用0填充
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 数据预处理后检查
        print(f"🔍 数据预处理后检查:")
        print(f"   训练集NaN数量: {np.isnan(X_train_scaled).sum()}")
        print(f"   验证集NaN数量: {np.isnan(X_valid_scaled).sum()}")
        print(f"   测试集NaN数量: {np.isnan(X_test_scaled).sum()}")
        
        # 再次处理可能的NaN值
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_valid_scaled = np.nan_to_num(X_valid_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        
        X_valid_tensor = torch.FloatTensor(X_valid_scaled)
        y_valid_tensor = torch.LongTensor(y_valid)
        
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = self._create_dataloader(train_dataset, shuffle=True, batch_size=batch_size)
        valid_loader = self._create_dataloader(valid_dataset, shuffle=False, batch_size=batch_size)
        test_loader = self._create_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        
        print(f"✅ 数据加载器创建完成")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   验证集: {len(valid_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
        
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
    
    def __init__(self, config: CS_tree_XGBoost_CS_Tree_ModelConfig):
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
# CS_tree_XGBoost_CS_Tree_Model模型实现
# =============================================================================

class CS_tree_XGBoost_CS_Tree_ModelModel(BaseModel):
    """CS_tree_XGBoost_CS_Tree_Model模型实现"""
    
    def __init__(self, config: CS_tree_XGBoost_CS_Tree_ModelConfig):
        super().__init__(config)
        
        # 输入维度必须从配置中获取，不能硬编码
        self.input_dim = getattr(config, 'input_dim', None)
        
        # 如果配置中没有input_dim，必须在训练时动态设置
        if self.input_dim is None:
            print("⚠️ 配置中未指定input_dim，将在数据加载时自动检测")
        
        # 延迟初始化网络层，等待input_dim确定
        self.layers = None
        
        # XGBoost模型（用于混合架构）
        self.xgb_model = None
        
    def _build_layers(self, input_dim: int):
        """使用动态输入维度构建网络，绝对不能硬编码"""
        print(f"🔧 构建网络层，输入维度: {input_dim}")
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),  # 动态input_dim
            nn.LayerNorm(self.config.hidden_dim),  # 使用LayerNorm替代BatchNorm避免batch size=1的问题
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.LayerNorm(self.config.hidden_dim // 2),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 首次前向传播时自动检测并构建网络
        if self.layers is None:
            if self.input_dim is None:
                self.input_dim = x.shape[-1]  # 从输入数据自动检测
                print(f"🔍 自动检测输入维度: {self.input_dim}")
            self.layers = self._build_layers(self.input_dim)
            # 确保层在正确的设备上
            self.layers = self.layers.to(x.device)
        
        # 添加输入数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ 检测到输入包含NaN或Inf值，进行修复")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查batch size，如果为1且模型在训练模式，切换到评估模式进行前向传播
        original_training = self.training
        if x.shape[0] == 1 and self.training:
            print(f"⚠️ 检测到batch size为1，临时切换到评估模式避免LayerNorm问题")
            self.eval()
        
        try:
            output = self.layers(x)
        finally:
            # 恢复原始训练模式
            if original_training and not self.training:
                self.train()
        
        # 添加输出数值稳定性检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"⚠️ 检测到模型输出包含NaN或Inf值，进行修复")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output
    
    def init_xgboost_model(self):
        """初始化XGBoost模型"""
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            colsample_bylevel=self.config.colsample_bylevel,
            colsample_bynode=self.config.colsample_bynode,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            gamma=self.config.gamma,
            min_child_weight=self.config.min_child_weight,
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            tree_method=self.config.tree_method,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=self.config.verbosity
        )

# =============================================================================
# 训练器类
# =============================================================================

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, config: CS_tree_XGBoost_CS_Tree_ModelConfig, model: BaseModel, 
                 data_loader_manager: CS_tree_XGBoost_CS_Tree_ModelDataLoaderManager, 
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
        # 确保模型已经初始化
        if hasattr(self.model, 'layers') and self.model.layers is None:
            # 如果模型还没有初始化层，先用一个dummy输入来初始化
            if self.config.input_dim:
                dummy_input = torch.randn(1, self.config.input_dim).to(self.device)
                _ = self.model(dummy_input)  # 触发层的初始化
        
        # 使用更保守的学习率防止梯度爆炸
        safe_lr = min(self.config.learning_rate, 0.001)  # 限制最大学习率
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=safe_lr,
            eps=1e-8,  # 增加数值稳定性
            weight_decay=1e-5  # 添加权重衰减防止过拟合
        )
        
        if safe_lr != self.config.learning_rate:
            self.logger.info(f"🔧 为数值稳定性调整学习率: {self.config.learning_rate} -> {safe_lr}")
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        if self.config.model_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
    
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
                
                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪 - 使用更保守的值
                gradient_clip = getattr(self.config, 'gradient_clip_value', 0.5)
                gradient_clip = min(gradient_clip, 0.5)  # 限制最大梯度裁剪值
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪 - 使用更保守的值
                gradient_clip = getattr(self.config, 'gradient_clip_value', 0.5)
                gradient_clip = min(gradient_clip, 0.5)  # 限制最大梯度裁剪值
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.config.model_type == "classification":
                # 添加数值稳定性检查
                if torch.isnan(output).any() or torch.isinf(output).any():
                    self.logger.warning(f"⚠️ 检测到模型输出包含NaN或Inf值，进行修复")
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                
                probs = torch.softmax(output, dim=1)
                
                # 检查概率是否包含NaN
                if torch.isnan(probs).any():
                    self.logger.warning(f"⚠️ 检测到softmax输出包含NaN值，进行修复")
                    probs = torch.nan_to_num(probs, nan=0.5)  # 用0.5填充NaN概率
                
                preds = torch.argmax(output, dim=1)
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
        
        metrics = {"loss": total_loss / len(train_loader)}
        
        if self.config.model_type == "classification":
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
            # 计算训练AUC - 添加NaN检查
            if len(np.unique(all_targets)) == 2:  # 二分类
                all_probs_array = np.array(all_probs)
                if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                    # 检查并处理NaN值
                    prob_positive = all_probs_array[:, 1]
                    targets_array = np.array(all_targets)
                    
                    # 过滤NaN值
                    valid_mask = ~(np.isnan(prob_positive) | np.isnan(targets_array))
                    if valid_mask.sum() > 0:
                        try:
                            metrics["auc"] = roc_auc_score(targets_array[valid_mask], prob_positive[valid_mask])
                        except ValueError as e:
                            self.logger.warning(f"⚠️ 训练AUC计算失败: {e}")
                            metrics["auc"] = 0.5  # 默认值
                    else:
                        self.logger.warning(f"⚠️ 所有概率值都是NaN，设置AUC为默认值")
                        metrics["auc"] = 0.5
        
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
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                if self.config.model_type == "classification":
                    # 添加数值稳定性检查
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        self.logger.warning(f"⚠️ 验证时检测到模型输出包含NaN或Inf值，进行修复")
                        output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    probs = torch.softmax(output, dim=1)
                    
                    # 检查概率是否包含NaN
                    if torch.isnan(probs).any():
                        self.logger.warning(f"⚠️ 验证时检测到softmax输出包含NaN值，进行修复")
                        probs = torch.nan_to_num(probs, nan=0.5)  # 用0.5填充NaN概率
                    
                    preds = torch.argmax(output, dim=1)
                    all_probs.extend(probs.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_targets.extend(target.detach().cpu().numpy())
        
        metrics = {"loss": total_loss / len(val_loader)}
        
        if self.config.model_type == "classification":
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
            metrics["f1"] = f1_score(all_targets, all_preds, average='weighted')
            
            # 计算AUC（如果有概率输出）- 添加强化的NaN检查
            if len(all_probs) > 0 and len(np.unique(all_targets)) == 2:
                try:
                    all_probs_array = np.array(all_probs)
                    if all_probs_array.ndim == 2 and all_probs_array.shape[1] >= 2:
                        # 检查并处理NaN值
                        prob_positive = all_probs_array[:, 1]
                        targets_array = np.array(all_targets)
                        
                        # 过滤NaN和Inf值
                        valid_mask = ~(np.isnan(prob_positive) | np.isnan(targets_array) | 
                                     np.isinf(prob_positive) | np.isinf(targets_array))
                        
                        if valid_mask.sum() > 0 and len(np.unique(targets_array[valid_mask])) == 2:
                            metrics["auc"] = roc_auc_score(targets_array[valid_mask], prob_positive[valid_mask])
                        else:
                            self.logger.warning(f"⚠️ 验证AUC计算失败：有效样本不足或类别不足")
                            metrics["auc"] = 0.5  # 默认值
                except Exception as e:
                    self.logger.warning(f"⚠️ 验证AUC计算异常: {e}")
                    metrics["auc"] = 0.5  # 默认值
            
            # 添加预测数据到metrics中，供后续使用
            metrics["y_true"] = all_targets
            metrics["y_prob"] = all_probs
        
        return metrics
    
    def save_checkpoint_if_best(self, epoch: int, val_loss: float):
        """保存最佳checkpoint的策略"""
        # 更新全局最佳
        if val_loss < self.checkpoint_tracker['global_best']['val_loss']:
            self._save_checkpoint(epoch, val_loss, 'global_best')
        
        # 更新区间最佳（实时更新best_model.pth）
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
        if epoch == 29:  # epoch从0开始，所以29是第30个epoch
            self._save_interval_best('early_best', '00-30')
        elif epoch == 59:
            self._save_interval_best('mid_best', '30-60')
        elif epoch == 99:
            self._save_interval_best('late_best', '60-100')

    def _save_checkpoint(self, epoch: int, val_loss: float, checkpoint_type: str):
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

    def _save_interval_best(self, checkpoint_type: str, epoch_range: str):
        """在区间结束时，复制区间最佳checkpoint为interval_best"""
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
    
    def train(self, batch_size: int = None, no_save_model: bool = False) -> Dict[str, Any]:
        """完整训练流程"""
        # 获取数据加载器，确保batch size至少为2
        batch_size = batch_size or self.config.batch_size
        batch_size = max(batch_size, 2)  # 确保batch size至少为2
        train_loader, val_loader, test_loader = self.data_loader_manager.load_data_loaders(
            batch_size=batch_size)
        
        self.setup_training_components()
        
        self.logger.info("开始训练...")
        self.logger.info(f"模型信息: {self.model.get_model_info()}")
        
        start_time = time.time()
        epochs = self.config.epochs
        
        # 用于保存最后一个epoch的验证预测数据
        final_val_predictions = None
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate_epoch(val_loader)
            
            # 如果是最后一个epoch，保存验证预测数据
            if epoch == epochs - 1 and self.config.model_type == "classification":
                final_val_predictions = {
                    "y_true_val": val_metrics.get("y_true", []),
                    "y_prob_val": val_metrics.get("y_prob", [])
                }
            
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
            
            # 保存checkpoint策略
            if not no_save_model:
                self.save_checkpoint_if_best(epoch, val_metrics["loss"])
            
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
                if 'auc' in train_metrics:
                    log_msg += f", Train AUC: {train_metrics['auc']:.4f}"

            log_msg += f", Time: {epoch_info['time']:.2f}s"
            self.logger.info(log_msg)
        
        total_time = time.time() - start_time
        
        # 保存训练历史
        self.save_training_history()
        
        result = {
            "best_val_score": self.best_val_score,
            "total_epochs": len(self.train_history),
            "total_time": total_time,
            "train_history": self.train_history
        }
        
        # 添加最后一个epoch的验证预测数据
        if final_val_predictions:
            result.update(final_val_predictions)
        
        return result
    
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
        checkpoint_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
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
    
    def __init__(self, config: CS_tree_XGBoost_CS_Tree_ModelConfig, model: BaseModel, 
                 data_loader_manager: CS_tree_XGBoost_CS_Tree_ModelDataLoaderManager):
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
                
                if self.config.model_type == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_probs.extend(probs.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                else:
                    all_preds.extend(outputs.detach().cpu().numpy())
                    all_probs.extend(outputs.detach().cpu().numpy())  # 回归任务概率即为预测值
        
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
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                if self.config.model_type == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_probs.extend(probs.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                else:
                    all_preds.extend(outputs.detach().cpu().numpy())
                
                all_targets.extend(target.detach().cpu().numpy())
        
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
    
    def generate_predictions(self, start_date: str = None, end_date: str = None, 
                           output_path: str = None, output_format: str = "parquet") -> pd.DataFrame:
        """生成预测数据文件用于回测"""
        # 1. 获取数据分割信息
        data_split = self.data_loader_manager.get_data_split_info()
        
        # 2. 确定使用的数据集（测试集）
        _, _, test_loader = self.data_loader_manager.load_data_loaders()
        
        # 3. 执行批量预测
        predictions, probabilities = self.predict_batch(test_loader)
        
        # 4. 生成日期序列并对齐
        test_start = data_split['test']['start_date']
        test_end = data_split['test']['end_date']
        
        # 生成日期范围
        date_range = pd.date_range(start=test_start, end=test_end, freq='D')
        
        # 确保预测数量与日期数量匹配
        if len(predictions) != len(date_range):
            # 如果数量不匹配，截取或填充
            min_len = min(len(predictions), len(date_range))
            predictions = predictions[:min_len]
            probabilities = probabilities[:min_len]
            date_range = date_range[:min_len]
        
        # 5. 创建标准格式DataFrame
        predictions_df = pd.DataFrame({
            'date': date_range,                    # 预测日期
            'prediction': predictions,             # 预测类别 (0或1)
            'confidence': probabilities[:, 1] if probabilities.ndim > 1 else probabilities,     # 预测置信度
            'probability_class_0': probabilities[:, 0] if probabilities.ndim > 1 else 1 - probabilities,  # 类别0的概率
            'probability_class_1': probabilities[:, 1] if probabilities.ndim > 1 else probabilities,  # 类别1的概率
            'model_name': self.config.model_name,  # 模型名称
            'timestamp': datetime.now().isoformat()  # 生成时间戳
        })
        
        # 6. 保存为指定格式文件
        if output_path:
            if output_format == "parquet":
                predictions_df.to_parquet(output_path, index=False)
            elif output_format == "csv":
                predictions_df.to_csv(output_path, index=False)
            
            self.logger.info(f"预测文件已保存: {output_path}")
        
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
    
    def __init__(self, config: CS_tree_XGBoost_CS_Tree_ModelConfig, model: BaseModel):
        self.config = config
        self.model = model
    
    def generate_model_md(self, output_path: str = "MODEL.md"):
        """生成MODEL.md文档"""
        model_info = self.model.get_model_info()
        
        content = f"""# {self.config.model_name} Model

## 模型概述

{self.config.model_name} 是一个基于XGBoost的{self.config.model_type}模型，结合了深度学习和梯度提升树的优势。

### 核心特点

- **任务类型**: {self.config.model_type}
- **模型参数**: {model_info['total_parameters']:,}
- **可训练参数**: {model_info['trainable_parameters']:,}
- **模型大小**: {model_info['model_size_mb']:.2f} MB
- **XGBoost集成**: 支持XGBoost梯度提升树算法

## 模型架构

### 核心组件

1. **神经网络组件**: 多层感知机用于特征学习
2. **XGBoost组件**: 梯度提升树用于最终预测
3. **混合架构**: 结合深度学习和传统机器学习的优势

### 网络结构

```
输入层 -> 隐藏层1({self.config.hidden_dim}) -> Dropout -> 隐藏层2({self.config.hidden_dim//2}) -> Dropout -> 输出层({self.config.output_dim})
```

## 技术原理

### XGBoost参数配置

- **n_estimators**: {self.config.n_estimators}
- **max_depth**: {self.config.max_depth}
- **learning_rate**: {self.config.learning_rate}
- **subsample**: {self.config.subsample}
- **colsample_bytree**: {self.config.colsample_bytree}

## 配置参数

### 训练配置
- 学习率: {self.config.learning_rate}
- 批次大小: {self.config.batch_size}
- 训练轮数: {self.config.epochs}

### 数据配置
- 序列长度: {self.config.seq_len}
- 预测长度: {self.config.pred_len}
- 输入维度: {self.config.input_dim or "动态检测"}
- 输出维度: {self.config.output_dim}

## 使用方法

### 训练模型

```bash
python CS_tree_XGBoost_CS_Tree_Model_unified.py train --config config.yaml --data-config data.yaml
```

### 模型推理

```bash
python CS_tree_XGBoost_CS_Tree_Model_unified.py inference --config config.yaml --data-config data.yaml --checkpoint best_model.pth
```

### 模型评估

```bash
python CS_tree_XGBoost_CS_Tree_Model_unified.py inference --config config.yaml --data-config data.yaml --checkpoint best_model.pth --inference-mode eval
```

### 生成预测文件

```bash
python CS_tree_XGBoost_CS_Tree_Model_unified.py inference --config config.yaml --data-config data.yaml --checkpoint best_model.pth --inference-mode test --format parquet
```

## 性能指标

模型支持以下评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- AUC-ROC

## 注意事项

1. **数据格式**: 支持parquet格式的时间序列数据
2. **特征维度**: 自动检测输入特征维度，无需手动配置
3. **GPU支持**: 自动检测GPU可用性，支持混合精度训练
4. **Checkpoint**: 支持多种checkpoint保存策略

## 更新日志

- 初始版本: {datetime.now().strftime('%Y-%m-%d')}
- 支持XGBoost集成
- 支持动态特征维度检测
- 支持Optuna超参数优化

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"模型文档已生成: {output_path}")

# =============================================================================
# 主要接口函数
# =============================================================================

def main_train(config_path: str = "config.yaml", data_config_path: str = "data.yaml",
               optuna_config_path: str = None, device: str = "auto",
               epochs_override: int = None, no_save_model: bool = False, 
               seed: int = 42, checkpoint_dir: str = "checkpoints"):
    """主训练函数"""
    # 设置所有随机种子以确保可复现性
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
    
    # 设置固定的日志路径
    log_dir = "/home/feng.hao.jie/deployment/model_explorer/c_training_evaluation_agent/training/logs/CS_tree_XGBoost_CS_Tree_Model_3982978951"
    logger = setup_logging(log_dir=log_dir, log_filename="log.txt")
    
    logger.info(f"🎲 设置随机种子: {seed}")
    logger.info(f"💾 Checkpoint保存目录: {checkpoint_dir}")
    
    # 确保checkpoint目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建配置对象
    config = CS_tree_XGBoost_CS_Tree_ModelConfig()
    
    # 从配置文件更新配置
    if 'hyperparameters' in config_dict:
        hp = config_dict['hyperparameters']
        config.n_estimators = hp.get('n_estimators', config.n_estimators)
        config.max_depth = hp.get('max_depth', config.max_depth)
        config.learning_rate = hp.get('learning_rate', config.learning_rate)
        config.subsample = hp.get('subsample', config.subsample)
        config.colsample_bytree = hp.get('colsample_bytree', config.colsample_bytree)
        config.colsample_bylevel = hp.get('colsample_bylevel', config.colsample_bylevel)
        config.colsample_bynode = hp.get('colsample_bynode', config.colsample_bynode)
        config.reg_alpha = hp.get('reg_alpha', config.reg_alpha)
        config.reg_lambda = hp.get('reg_lambda', config.reg_lambda)
        config.gamma = hp.get('gamma', config.gamma)
        config.min_child_weight = hp.get('min_child_weight', config.min_child_weight)
        config.objective = hp.get('objective', config.objective)
        config.eval_metric = hp.get('eval_metric', config.eval_metric)
        config.tree_method = hp.get('tree_method', config.tree_method)
        config.random_state = hp.get('random_state', config.random_state)
        config.n_jobs = hp.get('n_jobs', config.n_jobs)
        config.verbosity = hp.get('verbosity', config.verbosity)
        config.early_stopping_rounds = hp.get('early_stopping_rounds', config.early_stopping_rounds)
    
    if 'training' in config_dict:
        training = config_dict['training']
        config.batch_size = training.get('batch_size', config.batch_size)
        config.epochs = training.get('epochs', config.epochs)
    
    # 应用Optuna配置覆盖
    if optuna_config_path and os.path.exists(optuna_config_path):
        with open(optuna_config_path, 'r', encoding='utf-8') as f:
            optuna_config = yaml.safe_load(f)
        
        # 应用超参数覆盖
        for key, value in optuna_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"🔧 Optuna覆盖参数: {key} = {value}")
    
    # 覆盖配置
    if device != "auto":
        config.device = device
    if epochs_override:
        config.epochs = epochs_override
    
    config.checkpoint_dir = checkpoint_dir
    
    # 2. 创建数据加载器管理器
    data_loader_manager = CS_tree_XGBoost_CS_Tree_ModelDataLoaderManager(data_config_path, config)
    
    # 3. 创建模型
    model = CS_tree_XGBoost_CS_Tree_ModelModel(config)
    
    # 4. 创建训练器
    trainer = UnifiedTrainer(config, model, data_loader_manager, checkpoint_dir=checkpoint_dir)
    
    # 5. 执行训练
    results = trainer.train(no_save_model=no_save_model)
    
    # 6. 输出JSON格式结果
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
    """主推理函数"""
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建配置对象
    config = CS_tree_XGBoost_CS_Tree_ModelConfig()
    
    # 从配置文件更新配置
    if 'hyperparameters' in config_dict:
        hp = config_dict['hyperparameters']
        config.n_estimators = hp.get('n_estimators', config.n_estimators)
        config.max_depth = hp.get('max_depth', config.max_depth)
        config.learning_rate = hp.get('learning_rate', config.learning_rate)
        config.subsample = hp.get('subsample', config.subsample)
        config.colsample_bytree = hp.get('colsample_bytree', config.colsample_bytree)
    
    # 2. 创建数据加载器管理器
    data_loader_manager = CS_tree_XGBoost_CS_Tree_ModelDataLoaderManager(data_config_path, config)
    
    # 3. 创建数据加载器以获取输入维度
    _, _, _ = data_loader_manager.load_data_loaders()
    
    # 4. 创建模型并初始化
    model = CS_tree_XGBoost_CS_Tree_ModelModel(config)
    
    # 5. 初始化模型层（使用dummy输入）
    if config.input_dim:
        dummy_input = torch.randn(1, config.input_dim)
        _ = model(dummy_input)  # 触发层的初始化
    
    # 6. 创建推理器
    inferencer = UnifiedInferencer(config, model, data_loader_manager)
    
    # 7. 加载检查点
    checkpoint = inferencer.load_checkpoint(checkpoint_path)
    
    # 8. 根据mode执行不同操作
    if mode == "eval":
        # 传统模型评估
        metrics = inferencer.evaluate()
        print("评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        return metrics
    elif mode == "test":
        # 生成预测数据文件用于回测
        if not output_path:
            output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        
        predictions_df = inferencer.generate_predictions(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            output_format=output_format
        )
        
        print(f"预测文件已生成: {output_path}")
        print(f"预测数据形状: {predictions_df.shape}")
        return predictions_df

def main_generate_docs(config_path: str = "config.yaml", data_config_path: str = "data.yaml"):
    """主文档生成函数"""
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建配置对象
    config = CS_tree_XGBoost_CS_Tree_ModelConfig()
    
    # 2. 创建模型
    model = CS_tree_XGBoost_CS_Tree_ModelModel(config)
    
    # 3. 生成文档
    doc_generator = ModelDocumentationGenerator(config, model)
    doc_generator.generate_model_md("MODEL.md")
    
    print("文档生成完成")

# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CS_tree_XGBoost_CS_Tree_Model统一模型训练、推理和文档生成工具")
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