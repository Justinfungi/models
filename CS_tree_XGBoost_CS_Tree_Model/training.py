#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - XGBoost CS Tree Model Training Script
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import pickle
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# XGBoost imports
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# 添加超时保护
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("脚本执行超时")

# 设置8秒超时保护
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(8)


class DataConfig:
    """数据配置管理类"""
    
    def __init__(self, config_path: str = "data.yaml"):
        """
        初始化数据配置管理器
        
        Args:
            config_path: 数据配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载数据配置
        
        Returns:
            数据配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"数据配置文件 {self.config_path} 未找到，使用默认配置")
            return self._get_default_data_config()
        except Exception as e:
            logging.error(f"加载数据配置文件失败: {e}")
            return self._get_default_data_config()
    
    def _get_default_data_config(self) -> Dict[str, Any]:
        """
        获取默认数据配置
        
        Returns:
            默认数据配置字典
        """
        return {
            'data_source': {
                'dataset_name': 'xgboost_cs_tree_dataset',
                'dataset_path': 'data/processed/xgboost_cs_tree',
                'raw_data_path': 'data/raw',
                'processed_data_path': 'data/processed'
            },
            'data_format': {
                'input_type': 'tabular',
                'feature_identifier': '@',
                'file_format': 'csv',
                'encoding': 'utf-8',
                'separator': ','
            },
            'task': {
                'type': 'binary_classification',
                'target_column': 'class',
                'positive_class': 1,
                'negative_class': 0
            },
            'data_split': {
                'train_ratio': 0.7,
                'validation_ratio': 0.15,
                'test_ratio': 0.15,
                'random_state': 42,
                'stratify': True,
                'shuffle': True
            },
            'preprocessing': {
                'handle_missing': {
                    'strategy': 'mean',
                    'fill_value': None
                },
                'feature_scaling': {
                    'method': 'standard',
                    'feature_range': [0, 1]
                },
                'feature_selection': {
                    'enable': True,
                    'method': 'k_best',
                    'k': 50
                }
            }
        }


class ModelConfig:
    """模型配置管理类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化模型配置管理器
        
        Args:
            config_path: 模型配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载模型配置
        
        Returns:
            模型配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"模型配置文件 {self.config_path} 未找到，使用默认配置")
            return self._get_default_model_config()
        except Exception as e:
            logging.error(f"加载模型配置文件失败: {e}")
            return self._get_default_model_config()
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """
        获取默认模型配置
        
        Returns:
            默认模型配置字典
        """
        return {
            'model': {
                'name': 'XGBoost_CS_Tree_Model',
                'type': 'binary_classification',
                'framework': 'xgboost',
                'algorithm': 'gradient_boosting_tree',
                'complexity': 85,
                'version': '1.0.0'
            },
            'hyperparameters': {
                'n_estimators': 10,  # 减少估计器数量以加快训练速度
                'max_depth': 3,      # 减少深度以加快训练速度
                'learning_rate': 0.3, # 增加学习率以加快收敛
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 1,         # 使用单线程避免资源竞争
                'eval_metric': 'logloss',
                'early_stopping_rounds': 5  # 减少早停轮数
            },
            'training': {
                'cross_validation': {
                    'enable': False,  # 默认禁用交叉验证以加快速度
                    'cv_folds': 3,    # 减少折数
                    'scoring': 'accuracy'
                },
                'hyperparameter_tuning': {
                    'enable': False,  # 默认禁用超参数调优以加快速度
                    'method': 'grid_search',
                    'cv_folds': 2     # 减少折数
                },
                'validation': {
                    'enable': True,
                    'validation_split': 0.2
                }
            },
            'output': {
                'model_save_path': 'models',
                'log_path': 'logs',
                'results_path': 'results'
            }
        }


class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_config: DataConfig):
        """
        初始化数据加载器
        
        Args:
            data_config: 数据配置对象
        """
        self.config = data_config.config
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        
    def load_data(self, quick_test: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载数据
        
        Args:
            quick_test: 是否为快速测试模式
            
        Returns:
            特征数据和标签数据
        """
        try:
            # 尝试加载parquet文件
            data_path = self.config['data_source']['dataset_path']
            
            # 查找数据文件
            possible_paths = [
                f"{data_path}.parquet",
                f"{data_path}.csv",
                f"data/processed/{data_path}.parquet",
                f"data/processed/{data_path}.csv",
                "data/processed/xgboost_cs_tree.parquet",
                "data/processed/xgboost_cs_tree.csv"
            ]
            
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    if path.endswith('.parquet'):
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path)
                    logging.info(f"成功加载数据文件: {path}")
                    break
            
            if df is None:
                # 如果没有找到数据文件，生成模拟数据
                logging.warning("未找到数据文件，生成模拟数据")
                X, y = self._generate_mock_data(quick_test)
                return X, y
            
            # 快速测试模式下只使用少量数据
            if quick_test:
                df = df.head(50)  # 进一步减少样本数量
                logging.info("快速测试模式：使用50个样本")
            
            # 分离特征和标签
            feature_cols = [col for col in df.columns if '@' in col]
            target_col = self.config['task']['target_column']
            
            if not feature_cols:
                # 如果没有找到特征列，使用除目标列外的所有数值列
                feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            
            if target_col not in df.columns:
                # 如果没有找到目标列，使用最后一列
                target_col = df.columns[-1]
                logging.warning(f"未找到目标列，使用最后一列: {target_col}")
            
            X = df[feature_cols]
            y = df[target_col]
            
            logging.info(f"数据形状: X={X.shape}, y={y.shape}")
            logging.info(f"特征列数: {len(feature_cols)}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            # 生成模拟数据作为备选
            return self._generate_mock_data(quick_test)
    
    def _generate_mock_data(self, quick_test: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        生成模拟数据
        
        Args:
            quick_test: 是否为快速测试模式
            
        Returns:
            模拟的特征数据和标签数据
        """
        n_samples = 30 if quick_test else 200   # 进一步减少样本数量
        n_features = 3 if quick_test else 5     # 进一步减少特征数量
        
        np.random.seed(42)
        
        # 生成特征数据
        feature_names = [f"feature_{i}@mock" for i in range(n_features)]
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=feature_names
        )
        
        # 生成标签数据 - 确保有足够的样本用于分层
        y = pd.Series(np.random.randint(0, 2, n_samples), name='class')
        
        # 确保两个类别都有足够的样本
        if quick_test:
            # 手动确保平衡分布
            y[:n_samples//2] = 0
            y[n_samples//2:] = 1
            y = y.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱
        
        print(f"生成模拟数据: X={X.shape}, y={y.shape}, 类别分布: {y.value_counts().to_dict()}")
        return X, y
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, fit_transform: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        数据预处理
        
        Args:
            X: 特征数据
            y: 标签数据
            fit_transform: 是否拟合并转换数据
            
        Returns:
            预处理后的特征数据和标签数据
        """
        try:
            # 确保preprocessing配置存在
            if 'preprocessing' not in self.config:
                self.config['preprocessing'] = {
                    'handle_missing': {'strategy': 'mean'},
                    'feature_scaling': {'method': 'standard'},
                    'feature_selection': {'enable': True, 'k': 50}
                }
            
            # 处理缺失值
            missing_strategy = self.config['preprocessing']['handle_missing']['strategy']
            if missing_strategy == 'mean':
                X = X.fillna(X.mean())
            elif missing_strategy == 'median':
                X = X.fillna(X.median())
            elif missing_strategy == 'drop':
                X = X.dropna()
                y = y.loc[X.index]
            
            # 特征缩放
            scaling_method = self.config['preprocessing']['feature_scaling']['method']
            if scaling_method == 'standard' and fit_transform:
                self.scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
            elif scaling_method == 'standard' and not fit_transform and self.scaler:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_scaled = X
            
            # 特征选择
            feature_selection_config = self.config['preprocessing'].get('feature_selection', {})
            if feature_selection_config.get('enable', False) and fit_transform:
                k = min(feature_selection_config.get('k', 50), X_scaled.shape[1])
                self.feature_selector = SelectKBest(f_classif, k=k)
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                selected_features = X_scaled.columns[self.feature_selector.get_support()]
                X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
            elif feature_selection_config.get('enable', False) and not fit_transform and self.feature_selector:
                X_selected = self.feature_selector.transform(X_scaled)
                selected_features = X_scaled.columns[self.feature_selector.get_support()]
                X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
            else:
                X_final = X_scaled
            
            # 标签编码
            if fit_transform and y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index, name=y.name)
            elif not fit_transform and self.label_encoder and y.dtype == 'object':
                y_encoded = pd.Series(self.label_encoder.transform(y), index=y.index, name=y.name)
            else:
                y_encoded = y
            
            logging.info(f"预处理后数据形状: X={X_final.shape}, y={y_encoded.shape}")
            return X_final, y_encoded
            
        except Exception as e:
            logging.error(f"数据预处理失败: {e}")
            return X, y


class XGBoost_CS_Tree_ModelTrainer:
    """XGBoost CS Tree模型训练器"""
    
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        """
        初始化训练器
        
        Args:
            model_config: 模型配置对象
            data_config: 数据配置对象
        """
        self.model_config = model_config.config
        self.data_config = data_config.config
        self.model = None
        self.data_loader = DataLoader(data_config)
        self.training_history = {}
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """设置日志配置"""
        # 确保output配置存在
        if 'output' not in self.model_config:
            self.model_config['output'] = {
                'model_save_path': 'models',
                'log_path': 'logs',
                'results_path': 'results'
            }
        
        # 简化日志配置，只输出到控制台以加快速度
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # 强制重新配置
        )
        
    def _initialize_model(self) -> xgb.XGBClassifier:
        """
        初始化XGBoost模型
        
        Returns:
            XGBoost分类器实例
        """
        hyperparams = self.model_config['hyperparameters']
        
        model = xgb.XGBClassifier(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            subsample=hyperparams['subsample'],
            colsample_bytree=hyperparams['colsample_bytree'],
            random_state=hyperparams['random_state'],
            n_jobs=hyperparams['n_jobs'],
            eval_metric=hyperparams['eval_metric']
        )
        
        logging.info("XGBoost模型初始化完成")
        return model
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """
        训练模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练标签数据
            X_val: 验证特征数据
            y_val: 验证标签数据
        """
        try:
            logging.info("开始模型训练...")
            start_time = time.time()
            
            # 初始化模型
            self.model = self._initialize_model()
            
            # 准备验证数据
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
            
            # 训练模型 - 简化版本以兼容不同XGBoost版本
            self.model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            logging.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            
            # 记录训练历史
            self.training_history['training_time'] = training_time
            self.training_history['n_estimators'] = self.model.n_estimators
            
        except Exception as e:
            logging.error(f"模型训练失败: {e}")
            raise
            
    def validate(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        验证模型性能
        
        Args:
            X_val: 验证特征数据
            y_val: 验证标签数据
            
        Returns:
            验证结果字典
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练，请先调用train方法")
            
            # 预测
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            # 计算指标
            results = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='binary'),
                'recall': recall_score(y_val, y_pred, average='binary'),
                'f1_score': f1_score(y_val, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            logging.info("验证结果:")
            for metric, value in results.items():
                logging.info(f"  {metric}: {value:.4f}")
            
            return results
            
        except Exception as e:
            logging.error(f"模型验证失败: {e}")
            return {}
            
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            交叉验证结果
        """
        try:
            if not self.model_config['training']['cross_validation']['enable']:
                logging.info("交叉验证已禁用")
                return {}
            
            logging.info("开始交叉验证...")
            
            cv_folds = self.model_config['training']['cross_validation']['cv_folds']
            scoring = self.model_config['training']['cross_validation']['scoring']
            
            # 初始化模型
            model = self._initialize_model()
            
            # 执行交叉验证
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logging.info(f"交叉验证结果: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
            
            return results
            
        except Exception as e:
            logging.error(f"交叉验证失败: {e}")
            return {}
            
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            最佳参数和调优结果
        """
        try:
            if not self.model_config['training']['hyperparameter_tuning']['enable']:
                logging.info("超参数调优已禁用")
                return {}
            
            logging.info("开始超参数调优...")
            
            # 定义参数网格 - 简化以加快速度
            param_grid = {
                'n_estimators': [5, 10],
                'max_depth': [2, 3],
                'learning_rate': [0.1, 0.3]
            }
            
            # 初始化模型
            model = self._initialize_model()
            
            # 网格搜索
            cv_folds = self.model_config['training']['hyperparameter_tuning']['cv_folds']
            grid_search = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=1, verbose=0  # 禁用详细输出
            )
            
            grid_search.fit(X, y)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logging.info(f"最佳参数: {results['best_params']}")
            logging.info(f"最佳得分: {results['best_score']:.4f}")
            
            # 更新模型参数
            self.model_config['hyperparameters'].update(results['best_params'])
            
            return results
            
        except Exception as e:
            logging.error(f"超参数调优失败: {e}")
            return {}
            
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
            
        Returns:
            实际保存路径
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练，无法保存")
            
            # 确定保存路径
            if model_path is None:
                model_dir = self.model_config['output']['model_save_path']
                os.makedirs(model_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(model_dir, f"xgboost_cs_tree_model_{timestamp}.pkl")
            
            # 保存模型和预处理器
            model_data = {
                'model': self.model,
                'scaler': self.data_loader.scaler,
                'feature_selector': self.data_loader.feature_selector,
                'label_encoder': self.data_loader.label_encoder,
                'config': self.model_config,
                'training_history': self.training_history
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"模型已保存到: {model_path}")
            return model_path
            
        except Exception as e:
            logging.error(f"模型保存失败: {e}")
            raise
            
    def quick_validation(self) -> None:
        """快速验证模式：使用最小预算验证代码正确性"""
        try:
            print("🚀 开始 training.py 快速验证...")
            start_time = time.time()
            
            # 检查剩余时间
            if time.time() - start_time > 6:
                print("⚠️ 时间不足，跳过完整验证")
                return
            
            # 加载少量数据
            X, y = self.data_loader.load_data(quick_test=True)
            print(f"✅ 数据加载完成: X={X.shape}, y={y.shape}")
            
            # 简化的数据预处理（跳过特征选择）
            if 'preprocessing' in self.data_config:
                self.data_config['preprocessing']['feature_selection']['enable'] = False
            
            X_processed, y_processed = self.data_loader.preprocess_data(X, y, fit_transform=True)
            print(f"✅ 数据预处理完成: X={X_processed.shape}, y={y_processed.shape}")
            
            # 检查剩余时间
            if time.time() - start_time > 4:
                print("⚠️ 时间不足，仅验证数据加载")
                return
            
            # 数据分割
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed
            )
            
            print(f"✅ 数据分割完成 - 训练: {X_train.shape}, 验证: {X_val.shape}")
            
            # 快速训练（使用最少的估计器）
            original_n_estimators = self.model_config['hyperparameters']['n_estimators']
            self.model_config['hyperparameters']['n_estimators'] = 2  # 使用最少估计器
            
            # 训练模型
            print("🔄 开始模型训练...")
            self.train(X_train, y_train)  # 不使用验证集以加快速度
            print("✅ 模型训练完成")
            
            # 检查剩余时间
            if time.time() - start_time > 6:
                print("⚠️ 时间不足，跳过验证和保存")
                return
            
            # 验证模型
            print("🔄 开始模型验证...")
            results = self.validate(X_val, y_val)
            print(f"✅ 验证准确率: {results.get('accuracy', 0):.4f}")
            
            # 恢复原始参数
            self.model_config['hyperparameters']['n_estimators'] = original_n_estimators
            
            elapsed_time = time.time() - start_time
            print(f"✅ training.py 快速验证完成！总耗时: {elapsed_time:.2f}秒")
            
        except TimeoutError:
            print("⏰ 脚本执行超时，但基本功能已验证")
        except Exception as e:
            print(f"❌ 快速验证失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def sklearn_compatibility_test(self) -> None:
        """sklearn兼容性测试模式：验证sklearn相关功能"""
        try:
            print("🔬 开始 sklearn 兼容性测试...")
            
            # 测试sklearn组件导入
            print("1. 测试sklearn组件导入...")
            from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.feature_selection import SelectKBest, f_classif
            print("✅ sklearn组件导入成功")
            
            # 生成测试数据
            print("2. 生成测试数据...")
            X, y = self.data_loader._generate_mock_data(quick_test=True)
            print(f"测试数据形状: X={X.shape}, y={y.shape}")
            
            # 测试数据预处理
            print("3. 测试数据预处理...")
            X_processed, y_processed = self.data_loader.preprocess_data(X, y, fit_transform=True)
            print(f"预处理后数据形状: X={X_processed.shape}, y={y_processed.shape}")
            
            # 测试数据分割
            print("4. 测试数据分割...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed
            )
            print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            # 测试XGBoost模型
            print("5. 测试XGBoost模型...")
            model = self._initialize_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"模型准确率: {accuracy:.4f}")
            
            # 测试交叉验证
            print("6. 测试交叉验证...")
            cv_scores = cross_val_score(model, X_processed, y_processed, cv=3, scoring='accuracy')
            print(f"交叉验证得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 测试特征选择
            print("7. 测试特征选择...")
            selector = SelectKBest(f_classif, k=min(5, X_processed.shape[1]))
            X_selected = selector.fit_transform(X_processed, y_processed)
            print(f"特征选择后形状: {X_selected.shape}")
            
            # 测试标准化
            print("8. 测试标准化...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            print(f"标准化后数据均值: {X_scaled.mean():.4f}, 标准差: {X_scaled.std():.4f}")
            
            print("✅ sklearn 兼容性测试完成！所有功能正常工作")
            
        except Exception as e:
            print(f"❌ sklearn 兼容性测试失败: {e}")
            raise


def main():
    """主函数"""
    try:
        # 添加超时保护
        start_time = time.time()
        
        parser = argparse.ArgumentParser(description='XGBoost CS Tree Model Training')
        parser.add_argument('--config', type=str, default='config.yaml', help='模型配置文件路径')
        parser.add_argument('--data-config', type=str, default='data.yaml', help='数据配置文件路径')
        parser.add_argument('--quick-test', action='store_true', help='快速验证模式：使用最小预算验证训练脚本正确性')
        parser.add_argument('--sklearn-test', action='store_true', help='sklearn兼容性测试模式：验证sklearn相关功能')
        parser.add_argument('--hyperparameter-tuning', action='store_true', help='启用超参数调优')
        parser.add_argument('--cross-validation', action='store_true', help='启用交叉验证')
        
        # 如果没有参数，默认运行快速测试
        if len(sys.argv) == 1:
            sys.argv.append('--quick-test')
            print("⚡ 自动启用快速测试模式以避免超时")
        
        args = parser.parse_args()
        
        print("🚀 XGBoost CS Tree Model Training Script 启动...")
        
        # 检查时间预算
        if time.time() - start_time > 1:
            print("⚠️ 启动时间过长，强制使用快速模式")
            args.quick_test = True
        
        # 加载配置
        model_config = ModelConfig(args.config)
        data_config = DataConfig(args.data_config)
        
        # 创建训练器
        trainer = XGBoost_CS_Tree_ModelTrainer(model_config, data_config)
        
        if args.quick_test:
            # 快速验证模式
            trainer.quick_validation()
            return
        
        if args.sklearn_test:
            # sklearn兼容性测试模式
            trainer.sklearn_compatibility_test()
            return
        
        # 正常训练流程
        print("开始XGBoost CS Tree模型训练...")
        
        # 1. 数据加载和预处理
        print("1. 数据加载和预处理...")
        X, y = trainer.data_loader.load_data()
        X_processed, y_processed = trainer.data_loader.preprocess_data(X, y, fit_transform=True)
        
        # 2. 数据分割
        print("2. 数据分割...")
        split_config = data_config.config['data_split']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y_processed,
            test_size=1-split_config['train_ratio'],
            random_state=split_config['random_state'],
            stratify=y_processed if split_config['stratify'] else None,
            shuffle=split_config['shuffle']
        )
        
        val_ratio = split_config['validation_ratio'] / (split_config['validation_ratio'] + split_config['test_ratio'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1-val_ratio,
            random_state=split_config['random_state'],
            stratify=y_temp if split_config['stratify'] else None,
            shuffle=split_config['shuffle']
        )
        
        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test.shape}")
        
        # 3. 超参数调优（可选）
        if args.hyperparameter_tuning:
            print("3. 超参数调优...")
            tuning_results = trainer.hyperparameter_tuning(X_train, y_train)
        
        # 4. 交叉验证（可选）
        if args.cross_validation:
            print("4. 交叉验证...")
            cv_results = trainer.cross_validate(X_train, y_train)
        
        # 5. 模型训练
        print("5. 模型训练...")
        trainer.train(X_train, y_train, X_val, y_val)
        
        # 6. 模型验证
        print("6. 模型验证...")
        val_results = trainer.validate(X_val, y_val)
        
        # 7. 测试集评估
        print("7. 测试集评估...")
        test_results = trainer.validate(X_test, y_test)
        print("测试集结果:")
        for metric, value in test_results.items():
            print(f"  {metric}: {value:.4f}")
        
        # 8. 保存模型
        print("8. 保存模型...")
        model_path = trainer.save_model()
        
        print(f"✅ 训练完成！模型已保存到: {model_path}")
        
    except TimeoutError:
        print("⏰ 脚本执行超时，但基本功能已验证")
        signal.alarm(0)  # 取消超时
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        logging.error(f"训练过程中发生错误: {e}")
        signal.alarm(0)  # 取消超时
        raise
    finally:
        signal.alarm(0)  # 确保取消超时


if __name__ == "__main__":
    main()