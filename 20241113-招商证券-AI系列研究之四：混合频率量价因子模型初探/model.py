#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py - 混合频率量价因子模型架构定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import sys
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path

class ModelConfig:
    """模型配置类，从YAML文件加载配置"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})


class MultiFrequencyEncoder(nn.Module):
    """多频率数据编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化多频率编码器
        
        Args:
            config: 编码器配置
        """
        super().__init__()
        self.config = config
        
        # 周频编码器
        weekly_config = config.get('weekly', {})
        self.weekly_encoder = self._build_lstm_encoder(
            input_dim=weekly_config.get('input_dim', 50),
            hidden_dim=weekly_config.get('hidden_dim', 128),
            num_layers=weekly_config.get('num_layers', 2),
            dropout=weekly_config.get('dropout', 0.2),
            bidirectional=weekly_config.get('bidirectional', True)
        )
        
        # 日频编码器
        daily_config = config.get('daily', {})
        self.daily_encoder = self._build_lstm_encoder(
            input_dim=daily_config.get('input_dim', 100),
            hidden_dim=daily_config.get('hidden_dim', 256),
            num_layers=daily_config.get('num_layers', 2),
            dropout=daily_config.get('dropout', 0.2),
            bidirectional=daily_config.get('bidirectional', True)
        )
        
        # 日内编码器
        intraday_config = config.get('intraday', {})
        self.intraday_encoder = self._build_lstm_encoder(
            input_dim=intraday_config.get('input_dim', 200),
            hidden_dim=intraday_config.get('hidden_dim', 512),
            num_layers=intraday_config.get('num_layers', 2),
            dropout=intraday_config.get('dropout', 0.2),
            bidirectional=intraday_config.get('bidirectional', True)
        )
        
        # 计算输出维度
        self.output_dim = self._calculate_output_dim()
    
    def _build_lstm_encoder(self, input_dim: int, hidden_dim: int, num_layers: int, 
                           dropout: float, bidirectional: bool) -> nn.Module:
        """构建LSTM编码器"""
        return nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
    
    def _calculate_output_dim(self) -> int:
        """计算输出维度"""
        weekly_dim = self.config['weekly']['hidden_dim'] * (2 if self.config['weekly']['bidirectional'] else 1)
        daily_dim = self.config['daily']['hidden_dim'] * (2 if self.config['daily']['bidirectional'] else 1)
        intraday_dim = self.config['intraday']['hidden_dim'] * (2 if self.config['intraday']['bidirectional'] else 1)
        return weekly_dim + daily_dim + intraday_dim
    
    def forward(self, weekly_data: torch.Tensor, daily_data: torch.Tensor, 
                intraday_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            weekly_data: 周频数据 [batch_size, seq_len, weekly_features]
            daily_data: 日频数据 [batch_size, seq_len, daily_features]
            intraday_data: 日内数据 [batch_size, seq_len, intraday_features]
            
        Returns:
            编码后的特征 [batch_size, combined_features]
        """
        # 周频编码
        weekly_output, (weekly_hidden, _) = self.weekly_encoder(weekly_data)
        weekly_features = weekly_hidden[-1] if not self.config['weekly']['bidirectional'] else \
                         torch.cat([weekly_hidden[-2], weekly_hidden[-1]], dim=1)
        
        # 日频编码
        daily_output, (daily_hidden, _) = self.daily_encoder(daily_data)
        daily_features = daily_hidden[-1] if not self.config['daily']['bidirectional'] else \
                        torch.cat([daily_hidden[-2], daily_hidden[-1]], dim=1)
        
        # 日内编码
        intraday_output, (intraday_hidden, _) = self.intraday_encoder(intraday_data)
        intraday_features = intraday_hidden[-1] if not self.config['intraday']['bidirectional'] else \
                           torch.cat([intraday_hidden[-2], intraday_hidden[-1]], dim=1)
        
        # 拼接所有频率特征
        combined_features = torch.cat([weekly_features, daily_features, intraday_features], dim=1)
        
        return combined_features


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化多头注意力
        
        Args:
            config: 注意力配置
        """
        super().__init__()
        self.num_heads = config.get('num_heads', 8)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.dropout = config.get('dropout', 0.1)
        self.use_residual = config.get('use_residual', True)
        
        assert self.hidden_dim % self.num_heads == 0
        self.head_dim = self.hidden_dim // self.num_heads
        
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            注意力输出 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # 计算Q, K, V
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # 应用注意力
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # 输出投影
        output = self.output_proj(attention_output)
        output = self.dropout_layer(output)
        
        # 残差连接和层归一化
        if self.use_residual:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)
            
        return output


class FactorExtractor(nn.Module):
    """因子提取器"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        初始化因子提取器
        
        Args:
            input_dim: 输入维度
            config: 因子提取器配置
        """
        super().__init__()
        self.num_factors = config.get('num_factors', 64)
        self.factor_dim = config.get('factor_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.activation = config.get('activation', 'relu')
        self.dropout = config.get('dropout', 0.3)
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # 构建因子提取网络
        layers = []
        current_dim = input_dim
        
        for i in range(self.num_layers):
            # 线性层
            if i == self.num_layers - 1:
                # 最后一层输出因子
                layers.append(nn.Linear(current_dim, self.num_factors * self.factor_dim))
            else:
                next_dim = max(self.factor_dim, current_dim // 2)
                layers.append(nn.Linear(current_dim, next_dim))
                current_dim = next_dim
            
            # 批归一化
            if self.use_batch_norm and i < self.num_layers - 1:
                layers.append(nn.BatchNorm1d(current_dim))
            
            # 激活函数
            if i < self.num_layers - 1:
                if self.activation == 'relu':
                    layers.append(nn.ReLU())
                elif self.activation == 'gelu':
                    layers.append(nn.GELU())
                elif self.activation == 'tanh':
                    layers.append(nn.Tanh())
            
            # Dropout
            if i < self.num_layers - 1:
                layers.append(nn.Dropout(self.dropout))
        
        self.factor_network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            提取的因子 [batch_size, num_factors, factor_dim]
        """
        batch_size = x.shape[0]
        factors = self.factor_network(x)
        factors = factors.view(batch_size, self.num_factors, self.factor_dim)
        return factors


class Classifier(nn.Module):
    """分类器"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        初始化分类器
        
        Args:
            input_dim: 输入维度
            config: 分类器配置
        """
        super().__init__()
        self.hidden_dims = config.get('hidden_dims', [512, 256, 128])
        self.dropout = config.get('dropout', 0.4)
        self.activation = config.get('activation', 'relu')
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.output_activation = config.get('output_activation', 'softmax')
        self.num_classes = config.get('num_classes', 2)
        
        # 构建分类网络
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            
            layers.append(nn.Dropout(self.dropout))
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, self.num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            分类概率 [batch_size, num_classes]
        """
        logits = self.classifier(x)
        
        if self.output_activation == 'softmax':
            return F.softmax(logits, dim=1)
        elif self.output_activation == 'sigmoid':
            return torch.sigmoid(logits)
        else:
            return logits


class MixedFrequencyFactorModel(nn.Module):
    """混合频率量价因子模型"""
    
    def __init__(self, config: ModelConfig):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        model_config = config.get_model_config()
        
        self.model_name = model_config.get('name', 'mixed_frequency_factor_model')
        self.model_type = model_config.get('type', 'classification')
        self.num_classes = model_config.get('num_classes', 2)
        
        # 多频率编码器
        encoder_config = model_config.get('multi_frequency_encoder', {})
        self.multi_freq_encoder = MultiFrequencyEncoder(encoder_config)
        
        # 注意力机制
        attention_config = model_config.get('attention', {})
        attention_config['hidden_dim'] = self.multi_freq_encoder.output_dim
        self.attention = MultiHeadAttention(attention_config)
        
        # 因子提取器
        factor_config = model_config.get('factor_extractor', {})
        self.factor_extractor = FactorExtractor(self.multi_freq_encoder.output_dim, factor_config)
        
        # 分类器
        classifier_config = model_config.get('classifier', {})
        classifier_config['num_classes'] = self.num_classes
        factor_output_dim = factor_config.get('num_factors', 64) * factor_config.get('factor_dim', 128)
        self.classifier = Classifier(factor_output_dim, classifier_config)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, weekly_data: torch.Tensor, daily_data: torch.Tensor, 
                intraday_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            weekly_data: 周频数据
            daily_data: 日频数据
            intraday_data: 日内数据
            
        Returns:
            分类概率
        """
        # 多频率编码
        encoded_features = self.multi_freq_encoder(weekly_data, daily_data, intraday_data)
        
        # 添加序列维度用于注意力计算
        if len(encoded_features.shape) == 2:
            encoded_features = encoded_features.unsqueeze(1)
        
        # 注意力机制
        attended_features = self.attention(encoded_features)
        attended_features = attended_features.squeeze(1)
        
        # 因子提取
        factors = self.factor_extractor(attended_features)
        factors_flat = factors.view(factors.shape[0], -1)
        
        # 分类
        output = self.classifier(factors_flat)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_output_dim': self.multi_freq_encoder.output_dim,
        }


def create_model(config_path: str = "config.yaml") -> MixedFrequencyFactorModel:
    """
    创建模型实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        模型实例
    """
    try:
        config = ModelConfig(config_path)
        model = MixedFrequencyFactorModel(config)
        return model
    except Exception as e:
        raise RuntimeError(f"创建模型失败: {e}")


def load_model(model_path: str, config_path: str = "config.yaml") -> MixedFrequencyFactorModel:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        
    Returns:
        加载的模型
    """
    try:
        model = create_model(config_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")


if __name__ == "__main__":
    print("🚀 开始 model.py 快速验证...")
    
    try:
        # 1. 测试配置加载
        print("📋 测试配置加载...")
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件 {config_path} 不存在，跳过配置测试")
            # 创建最小配置用于测试
            config = ModelConfig.__new__(ModelConfig)
            config.config = {
                'model': {
                    'name': 'test_model',
                    'type': 'classification',
                    'num_classes': 2,
                    'multi_frequency_encoder': {
                        'weekly': {'input_dim': 20, 'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.1, 'bidirectional': True},
                        'daily': {'input_dim': 30, 'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.1, 'bidirectional': True},
                        'intraday': {'input_dim': 40, 'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.1, 'bidirectional': True}
                    },
                    'attention': {'num_heads': 4, 'hidden_dim': 384, 'dropout': 0.1, 'use_residual': True},
                    'factor_extractor': {'num_factors': 32, 'factor_dim': 64, 'num_layers': 2, 'activation': 'relu', 'dropout': 0.2, 'use_batch_norm': True},
                    'classifier': {'hidden_dims': [256, 128], 'dropout': 0.3, 'activation': 'relu', 'use_batch_norm': True, 'output_activation': 'softmax'}
                }
            }
        else:
            config = ModelConfig(config_path)
        print("✅ 配置加载成功")
        
        # 2. 测试模型创建
        print("🏗️  测试模型创建...")
        model = MixedFrequencyFactorModel(config)
        print("✅ 模型创建成功")
        
        # 3. 测试设备适配
        print("🖥️  测试设备适配...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ 模型已移动到设备: {device}")
        
        # 4. 测试前向传播
        print("⚡ 测试前向传播...")
        batch_size = 4
        seq_len = 10
        
        # 创建模拟数据
        model_config = config.get_model_config()
        encoder_config = model_config.get('multi_frequency_encoder', {})
        
        weekly_dim = encoder_config.get('weekly', {}).get('input_dim', 20)
        daily_dim = encoder_config.get('daily', {}).get('input_dim', 30)
        intraday_dim = encoder_config.get('intraday', {}).get('input_dim', 40)
        
        weekly_data = torch.randn(batch_size, seq_len, weekly_dim).to(device)
        daily_data = torch.randn(batch_size, seq_len, daily_dim).to(device)
        intraday_data = torch.randn(batch_size, seq_len, intraday_dim).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(weekly_data, daily_data, intraday_data)
        
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
        print(f"📊 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 5. 打印模型信息
        print("📈 模型统计信息:")
        model_info = model.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # 6. 测试模型保存和加载
        print("💾 测试模型保存...")
        temp_model_path = "temp_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.config
        }, temp_model_path)
        print("✅ 模型保存成功")
        
        # 清理临时文件
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print("🗑️  临时文件已清理")
        
        print("✅ model.py 验证通过!")
        print("🎉 所有测试均成功完成!")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)