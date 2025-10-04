# 20241113-招商证券-AI系列研究之四：混合频率量价因子模型初探 Model

## 模型概述

混合频率量价因子模型是基于招商证券AI系列研究的深度学习模型，专门用于金融时间序列的二分类预测任务。该模型通过融合多频率数据（周频、日频、日内频率）和量价因子特征，实现对金融市场趋势的精准预测。

### 核心特点

- **多频率数据融合**：集成周频、日频和日内三种时间频率的数据，捕获不同时间尺度的市场信号
- **深度特征提取**：采用双向GRU网络提取时序特征，结合多头注意力机制增强特征表达能力
- **因子化建模**：通过因子提取器将原始特征转换为高级因子表示，提升模型的金融解释性
- **端到端训练**：支持完整的训练、验证和推理流程，具备完善的性能监控和模型导出功能

## 模型架构

### 核心组件

1. **多频率编码器 (MultiFrequencyEncoder)**
   - 三个独立的GRU网络分别处理周频、日频、日内数据
   - 多头注意力机制实现跨频率特征融合
   - 残差连接保证梯度流动

2. **因子提取器 (FactorExtractor)**
   - 多个并行的因子提取网络
   - 动态权重分配机制
   - 因子级别的特征表示

3. **分类器 (Classifier)**
   - 多层全连接网络
   - Dropout正则化防止过拟合
   - 支持二分类和多分类任务

### 网络结构

```
输入数据 → 多频率编码器 → 因子提取器 → 分类器 → 输出概率
    ↓           ↓           ↓         ↓
周频数据    双向GRU      因子网络    全连接层
日频数据  + 注意力机制  + 权重分配  + Dropout
日内数据    残差连接     加权融合    Softmax
```

## 使用方法

### 安装依赖

```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install pyyaml
```

### 基本使用

#### 1. 模型训练

```python
from model import create_model, ModelConfig
from training import ModelTrainer, DataLoader

# 创建配置和模型
config = ModelConfig("config.yaml")
model, device = create_model("config.yaml")

# 准备数据
data_config = DataConfig("data.yaml")
data_loader = DataLoader(data_config)
features, labels = data_loader.load_data()

# 训练模型
trainer = ModelTrainer(model, config, data_config)
results = trainer.train(features, labels)
```

#### 2. 模型推理

```python
from inference import ModelInferencer
import numpy as np

# 初始化推理器
inferencer = ModelInferencer(
    config_path="config.yaml",
    model_path="checkpoints/best_model.pth"
)

# 单样本推理
sample_data = np.random.randn(69)  # 69个特征
result = inferencer.predict_single(sample_data)
print(f"预测类别: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.4f}")

# 批量推理
batch_data = [np.random.randn(69) for _ in range(10)]
batch_results = inferencer.predict_batch(batch_data)
```

#### 3. 快速验证

```python
# 验证模型组件
python model.py

# 验证训练流程
python training.py --quick-test

# 验证推理功能
python inference.py --quick-test
```

## 配置说明

### 模型配置 (config.yaml)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.input_dim` | 69 | 输入特征维度 |
| `model.hidden_dim` | 128 | 隐藏层维度 |
| `model.factor_dim` | 64 | 因子维度 |
| `model.num_factors` | 10 | 因子数量 |
| `model.num_classes` | 2 | 分类类别数 |
| `model.dropout_rate` | 0.3 | Dropout比率 |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.batch_size` | 64 | 批次大小 |
| `training.num_epochs` | 100 | 训练轮数 |
| `training.learning_rate` | 0.001 | 学习率 |
| `training.weight_decay` | 1e-5 | 权重衰减 |

### 数据配置 (data.yaml)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.feature_prefix` | "@" | 特征列前缀 |
| `data.sequence_length` | 60 | 序列长度 |
| `data.train_ratio` | 0.7 | 训练集比例 |
| `data.preprocessing.scaling_method` | "standard" | 标准化方法 |

## 代码使用示例

### 完整训练示例

```python
#!/usr/bin/env python3
import torch
import numpy as np
from model import create_model, ModelConfig
from training import ModelTrainer, DataLoader, DataConfig

def main():
    # 1. 配置初始化
    config = ModelConfig("config.yaml")
    data_config = DataConfig("data.yaml")
    
    # 2. 创建模型
    model, device = create_model("config.yaml")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 数据准备
    data_loader = DataLoader(data_config)
    features, labels = data_loader.load_data()
    print(f"数据形状: 特征{features.shape}, 标签{labels.shape}")
    
    # 4. 模型训练
    trainer = ModelTrainer(model, config, data_config)
    results = trainer.train(features, labels)
    
    # 5. 结果输出
    print(f"最佳验证损失: {results['best_val_loss']:.4f}")
    print(f"最终准确率: {results['final_metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

### 推理部署示例

```python
#!/usr/bin/env python3
import pandas as pd
from inference import ModelInferencer

def deploy_model():
    # 1. 初始化推理器
    inferencer = ModelInferencer(
        config_path="config.yaml",
        model_path="checkpoints/best_model.pth"
    )
    
    # 2. 加载测试数据
    test_data = pd.read_parquet("data/test_data.parquet")
    
    # 3. 批量推理
    results = []
    batch_size = 32
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data.iloc[i:i+batch_size]
        batch_results = inferencer.predict_batch([
            row.values for _, row in batch.iterrows()
        ])
        results.extend(batch_results)
    
    # 4. 保存结果
    predictions_df = pd.DataFrame(results)
    predictions_df.to_csv("predictions.csv", index=False)
    
    # 5. 性能统计
    stats = inferencer.get_inference_stats()
    print(f"推理统计: {stats}")

if __name__ == "__main__":
    deploy_model()
```

### 性能基准测试

```python
#!/usr/bin/env python3
from inference import ModelInferencer

def benchmark_performance():
    # 初始化推理器
    inferencer = ModelInferencer("config.yaml", "checkpoints/best_model.pth")
    
    # 运行基准测试
    benchmark_results = inferencer.benchmark_performance(
        num_samples=1000,
        batch_size=32
    )
    
    # 输出结果
    print("性能基准测试结果:")
    for metric, value in benchmark_results.items():
        print(f"  {metric}: {value:.6f}")
    
    # 模型导出
    inferencer.export_model("model.onnx", format="onnx")
    print("模型已导出为 ONNX 格式")

if __name__ == "__main__":
    benchmark_performance()
```