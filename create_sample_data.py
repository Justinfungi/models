#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_sample_data.py - 创建示例数据集供所有模型测试使用
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class SampleDataGenerator:
    """示例数据生成器"""

    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)

    def generate_base_features(self, n_samples: int = 10000) -> pd.DataFrame:
        """生成基础特征数据"""
        features = {}

        # 技术指标特征
        for i in range(20):
            features[f'tech_indicator_{i}'] = np.random.randn(n_samples)

        # 价格相关特征
        features['close_price'] = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01))
        features['open_price'] = features['close_price'] * (1 + np.random.randn(n_samples) * 0.02)
        features['high_price'] = np.maximum(features['open_price'], features['close_price']) * (1 + np.abs(np.random.randn(n_samples)) * 0.03)
        features['low_price'] = np.minimum(features['open_price'], features['close_price']) * (1 - np.abs(np.random.randn(n_samples)) * 0.03)
        features['volume'] = np.abs(np.random.randn(n_samples)) * 1000000 + 500000

        # 技术指标计算
        close_series = pd.Series(features['close_price'])
        features['returns'] = np.log(close_series / close_series.shift(1)).fillna(0)
        features['ma5'] = close_series.rolling(5).mean().fillna(close_series)
        features['ma20'] = close_series.rolling(20).mean().fillna(close_series)
        features['ma60'] = close_series.rolling(60).mean().fillna(close_series)
        features['rsi'] = self._calculate_rsi(close_series)
        features['macd'], features['macd_signal'] = self._calculate_macd(close_series)
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = self._calculate_bollinger_bands(close_series)

        # 基本面特征
        for i in range(15):
            features[f'fundamental_{i}'] = np.random.randn(n_samples)

        # 估值指标
        features['pe_ratio'] = np.abs(np.random.randn(n_samples)) * 50 + 10
        features['pb_ratio'] = np.abs(np.random.randn(n_samples)) * 5 + 0.5
        features['roe'] = np.random.randn(n_samples) * 0.1 + 0.15
        features['net_profit_growth'] = np.random.randn(n_samples) * 0.2

        # 市场微观结构特征
        features['bid_ask_spread'] = np.abs(np.random.randn(n_samples)) * 0.001 + 0.0005
        features['order_book_depth'] = np.abs(np.random.randn(n_samples)) * 1000 + 500
        features['trade_frequency'] = np.abs(np.random.randn(n_samples)) * 100 + 50
        features['market_impact'] = np.random.randn(n_samples) * 0.01

        return pd.DataFrame(features)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.fillna(prices), middle.fillna(prices), lower.fillna(prices)

    def generate_labels(self, features: pd.DataFrame) -> np.ndarray:
        """生成三分类标签 (-1, 0, 1)"""
        n_samples = len(features)

        # 基于多个因子生成更真实的标签
        returns = features['returns'].values

        # 添加一些技术指标的影响
        ma_signal = (features['close_price'] > features['ma20']).astype(int) - 0.5
        rsi_signal = ((features['rsi'] > 70).astype(int) - (features['rsi'] < 30).astype(int))

        # 综合信号
        combined_signal = returns * 0.5 + ma_signal * 0.3 + rsi_signal * 0.2

        # 添加噪声
        noise = np.random.randn(n_samples) * 0.1
        final_signal = combined_signal + noise

        # 生成标签
        labels = np.zeros(n_samples, dtype=int)
        labels[final_signal > 0.05] = 1   # 上涨
        labels[final_signal < -0.05] = -1  # 下跌
        # 中间保持为0（震荡）

        return labels

    def generate_multifrequency_data(self, base_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """生成多频率数据（用于混合频率模型）"""
        # 这里简化处理，实际应该从原始高频数据聚合

        # 日频数据就是基础数据
        daily_data = base_features.copy()

        # 模拟周频数据（按周聚合）
        if 'date' in base_features.columns:
            base_features = base_features.set_index('date')
            weekly_data = base_features.resample('W').agg({
                col: 'last' if col in ['close_price', 'volume'] else 'mean'
                for col in base_features.columns
                if col != 'instrument'
            }).reset_index()
        else:
            weekly_data = base_features.copy()  # 如果没有日期列，使用原数据

        # 模拟日内数据（简单复制日频数据，添加时间信息）
        intraday_data = base_features.copy()
        if 'date' in intraday_data.columns:
            intraday_data = intraday_data.reset_index(drop=True)
            # 为每行添加随机时间戳（模拟日内数据）
            base_times = pd.date_range('09:30:00', '15:00:00', freq='1H').time
            intraday_data['time'] = np.random.choice(base_times, len(intraday_data))
            intraday_data['datetime'] = intraday_data.apply(
                lambda row: pd.Timestamp.combine(row['date'].date(), row['time']), axis=1
            )

        return {
            'weekly': weekly_data,
            'daily': daily_data,
            'intraday': intraday_data
        }

    def create_complete_dataset(self, n_samples: int = 10000) -> Dict[str, Any]:
        """创建完整的数据集"""
        print(f"🔧 生成 {n_samples} 条示例数据...")

        # 生成基础特征
        features = self.generate_base_features(n_samples)

        # 添加时间和股票信息
        features['date'] = pd.date_range('2015-01-01', periods=n_samples, freq='D')
        features['instrument'] = np.random.choice(['000001.SZ', '000002.SZ', '600000.SH'], n_samples)
        features['weight'] = np.ones(n_samples)  # 样本权重

        # 生成标签
        labels = self.generate_labels(features)
        features['label'] = labels

        # 生成多频率数据
        multifreq_data = self.generate_multifrequency_data(features[['date'] + [col for col in features.columns if col not in ['date', 'instrument', 'weight']]])

        dataset = {
            'features': features,
            'labels': labels,
            'multifrequency': multifreq_data
        }

        print("✅ 数据生成完成")
        print(f"   - 样本数量: {n_samples}")
        print(f"   - 特征数量: {len(features.columns)}")
        print(f"   - 标签分布: {np.bincount(labels + 1)} (-1, 0, 1)")

        return dataset

    def save_dataset(self, dataset: Dict[str, Any], output_dir: str = "sample_data"):
        """保存数据集"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存主数据集
        dataset['features'].to_csv(output_path / "features.csv", index=False)
        np.save(output_path / "labels.npy", dataset['labels'])

        # 保存多频率数据
        for freq_name, freq_data in dataset['multifrequency'].items():
            freq_data.to_csv(output_path / f"{freq_name}_features.csv", index=False)

        # 保存数据信息
        info = {
            'n_samples': len(dataset['features']),
            'n_features': len(dataset['features'].columns),
            'label_distribution': np.bincount(dataset['labels'] + 1).tolist(),
            'date_range': [
                dataset['features']['date'].min().strftime('%Y-%m-%d'),
                dataset['features']['date'].max().strftime('%Y-%m-%d')
            ],
            'instruments': dataset['features']['instrument'].unique().tolist()
        }

        with open(output_path / "dataset_info.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(info, f, default_flow_style=False)

        print(f"💾 数据集已保存到: {output_path}")

    def create_model_specific_data(self, model_name: str) -> Dict[str, Any]:
        """为特定模型创建定制数据"""
        base_dataset = self.create_complete_dataset()

        if '混合频率' in model_name:
            # 为混合频率模型准备特定格式
            # 这里可以添加特定预处理
            pass
        elif 'BayesianCNN' in model_name:
            # 为BayesianCNN准备数据
            pass
        elif 'XGBoost' in model_name:
            # 为XGBoost准备表格数据
            pass

        return base_dataset

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建量化交易模型示例数据")
    parser.add_argument('--n_samples', type=int, default=10000, help='样本数量')
    parser.add_argument('--output_dir', type=str, default='sample_data', help='输出目录')
    parser.add_argument('--model', type=str, default=None, help='特定模型名称')

    args = parser.parse_args()

    generator = SampleDataGenerator()

    if args.model:
        print(f"🎯 为模型 {args.model} 创建定制数据")
        dataset = generator.create_model_specific_data(args.model)
    else:
        print("📊 创建通用示例数据")
        dataset = generator.create_complete_dataset(args.n_samples)

    generator.save_dataset(dataset, args.output_dir)

    print("🎉 数据创建完成！")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"📄 包含文件: features.csv, labels.npy, 多频率数据等")

if __name__ == "__main__":
    main()
