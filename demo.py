#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py - 量化交易模型统一演示
展示如何使用测试工具和示例数据
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    """主演示函数"""
    print("🚀 量化交易模型统一演示")
    print("=" * 50)

    # 1. 检查示例数据
    print("\n📊 1. 检查示例数据")
    sample_data_dir = Path("sample_data")
    if sample_data_dir.exists():
        print(f"✅ 示例数据目录存在: {sample_data_dir}")

        # 检查主要文件
        files_to_check = ["features.csv", "labels.npy", "dataset_info.yaml"]
        for file in files_to_check:
            if (sample_data_dir / file).exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ 缺失: {file}")

        # 读取并显示数据信息
        try:
            with open(sample_data_dir / "dataset_info.yaml", 'r', encoding='utf-8') as f:
                import yaml
                info = yaml.safe_load(f)
            print(f"   📈 数据集信息:")
            print(f"      - 样本数量: {info['n_samples']}")
            print(f"      - 特征数量: {info['n_features']}")
            print(f"      - 标签分布: {info['label_distribution']}")
            print(f"      - 时间范围: {info['date_range']}")
        except Exception as e:
            print(f"   ❌ 读取数据集信息失败: {e}")

    else:
        print("❌ 示例数据目录不存在，请先运行: python create_sample_data.py")

    # 2. 检查模型目录结构
    print("\n🏗️  2. 检查模型目录结构")
    model_dirs = [
        '20241113-招商证券-AI系列研究之四：混合频率量价因子模型初探',
        'BayesianCNN',
        'CS_tree_XGBoost_CS_Tree_Model',
        'HRM',
        'mamba',
        'TKAN',
        'wanglang_20250916_Conv_Trans',
        'wanglang_20250916_ConvM_Lstm'
    ]

    for model_dir in model_dirs:
        dir_path = Path(model_dir)
        if dir_path.exists():
            config_exists = (dir_path / "config.yaml").exists()
            model_file_exists = any((dir_path / f).exists() for f in ["model.py", "*unified.py"])
            print(f"   ✅ {model_dir}: 配置={config_exists}, 模型文件={model_file_exists}")
        else:
            print(f"   ❌ 缺失模型目录: {model_dir}")

    # 3. 运行快速架构检查
    print("\n🔍 3. 运行快速架构检查")
    print("   运行完整测试可能需要较长时间...")
    print("   如需完整测试，请运行: python test_all_models.py")

    # 简单检查一个模型的配置
    test_model = "20241113-招商证券-AI系列研究之四：混合频率量价因子模型初探"
    config_path = Path(test_model) / "config.yaml"

    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            num_classes = config.get('model', {}).get('num_classes', 'unknown')
            print(f"   ✅ {test_model} 配置检查:")
            print(f"      - 分类数量: {num_classes}")
            print(f"      - 模型类型: {config.get('model', {}).get('type', 'unknown')}")
        except Exception as e:
            print(f"   ❌ 配置检查失败: {e}")
    else:
        print(f"   ❌ 配置文件不存在: {config_path}")

    # 4. 使用说明
    print("\n📚 4. 使用说明")
    print("   📄 生成示例数据: python create_sample_data.py --n_samples 5000")
    print("   🧪 测试所有模型: python test_all_models.py")
    print("   📊 查看测试报告: cat model_test_report.md")

    print("\n   模型训练示例:")
    print("   cd 20241113-招商证券-AI系列研究之四：混合频率量价因子模型初探")
    print("   python training.py")

    print("\n   模型推理示例:")
    print("   cd 20241113-招商证券-AI系列研究之四：混合频率量价因子模型初探")
    print("   python inference.py --model_path checkpoints/best_model.pth")

    print("\n🎉 演示完成!")
    print("   所有模型已配置为三分类任务（-1, 0, 1）")
    print("   示例数据包含完整的量化交易特征")

if __name__ == "__main__":
    main()
