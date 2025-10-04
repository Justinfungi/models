#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_model.py - 模型调试工具
提供详细的模型诊断和修复建议
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelDebugger:
    """模型调试器"""

    def __init__(self):
        self.models_dir = Path(__file__).parent
        self.model_names = [
            '20241113-招商证券-AI系列研究之四：混合频率量价因子模型初探',
            'BayesianCNN',
            'CS_tree_XGBoost_CS_Tree_Model',
            'HRM',
            'mamba',
            'TKAN',
            'wanglang_20250916_Conv_Trans',
            'wanglang_20250916_ConvM_Lstm'
        ]
        self.sample_data_path = self.models_dir / "sample_data"

    def load_sample_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """加载示例数据"""
        try:
            features_path = self.sample_data_path / "features.csv"
            labels_path = self.sample_data_path / "labels.npy"

            if not features_path.exists() or not labels_path.exists():
                print("❌ 示例数据文件不存在，请先运行: python create_sample_data.py")
                return None, None

            features = pd.read_csv(features_path)
            labels = np.load(labels_path)

            print(f"✅ 加载示例数据: {len(features)} 样本, {len(features.columns)} 特征")
            return features, labels

        except Exception as e:
            print(f"❌ 加载示例数据失败: {e}")
            return None, None

    def diagnose_model_config(self, model_name: str) -> Dict[str, Any]:
        """诊断模型配置"""
        result = {
            'config_valid': False,
            'num_classes_correct': False,
            'output_dim_correct': False,
            'loss_function_correct': False,
            'issues': [],
            'suggestions': []
        }

        model_path = self.models_dir / model_name
        config_path = model_path / "config.yaml"

        if not config_path.exists():
            result['issues'].append("配置文件不存在")
            result['suggestions'].append("创建config.yaml文件")
            return result

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            result['config_valid'] = True

            # 检查num_classes
            num_classes = self._extract_num_classes(config)
            if num_classes == 3:
                result['num_classes_correct'] = True
            else:
                result['issues'].append(f"num_classes = {num_classes}, 应为3")
                result['suggestions'].append("设置 num_classes: 3")

            # 检查输出维度
            output_dim = self._extract_output_dim(config)
            if output_dim in [3, None]:  # None表示自动推断
                result['output_dim_correct'] = True
            else:
                result['issues'].append(f"output_dim = {output_dim}, 应为3")
                result['suggestions'].append("设置 output_dim: 3")

            # 检查损失函数
            loss_function = self._extract_loss_function(config)
            if loss_function in ['categorical_crossentropy', 'cross_entropy', None]:
                result['loss_function_correct'] = True
            elif loss_function == 'binary_crossentropy':
                result['issues'].append("使用二分类损失函数，应改为多分类")
                result['suggestions'].append("改为 categorical_crossentropy 或 cross_entropy")

        except Exception as e:
            result['issues'].append(f"配置解析错误: {e}")
            result['suggestions'].append("检查YAML格式")

        return result

    def _extract_num_classes(self, config: Dict[str, Any]) -> Optional[int]:
        """从配置中提取num_classes"""
        locations = [
            ('model', 'num_classes'),
            ('task', 'num_classes'),
            ('architecture', 'num_classes'),
            ('architecture', 'classifier', 'num_classes')
        ]

        for location in locations:
            value = config
            try:
                for key in location:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                continue
        return None

    def _extract_output_dim(self, config: Dict[str, Any]) -> Optional[int]:
        """从配置中提取output_dim"""
        locations = [
            ('architecture', 'output_dim'),
            ('model', 'output_dim'),
            ('architecture', 'network', 'output_dim')
        ]

        for location in locations:
            value = config
            try:
                for key in location:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                continue
        return None

    def _extract_loss_function(self, config: Dict[str, Any]) -> Optional[str]:
        """从配置中提取损失函数"""
        locations = [
            ('training', 'loss', 'type'),
            ('training', 'loss_function'),
            ('loss', 'type'),
            ('loss_function',)
        ]

        for location in locations:
            value = config
            try:
                for key in location:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                continue
        return None

    def diagnose_model_code(self, model_name: str) -> Dict[str, Any]:
        """诊断模型代码"""
        result = {
            'code_exists': False,
            'imports_work': False,
            'class_structure': False,
            'forward_method': False,
            'training_method': False,
            'issues': [],
            'suggestions': []
        }

        model_path = self.models_dir / model_name

        # 查找Python文件
        python_files = list(model_path.glob('*.py'))
        if not python_files:
            result['issues'].append("没有找到Python文件")
            result['suggestions'].append("创建模型实现文件")
            return result

        result['code_exists'] = True

        # 检查每个Python文件
        for py_file in python_files:
            if py_file.name.startswith('test_') or py_file.name.startswith('debug_'):
                continue

            try:
                # 添加到Python路径
                if str(model_path) not in sys.path:
                    sys.path.insert(0, str(model_path))

                # 动态导入
                module_name = py_file.stem
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, py_file)

                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    result['imports_work'] = True

                    # 检查类结构
                    model_classes = []
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if hasattr(attr, '__bases__') and hasattr(attr, '__dict__'):
                            # 找到模型类
                            if 'Model' in attr_name and hasattr(attr, 'forward'):
                                model_classes.append(attr_name)
                                result['class_structure'] = True

                                # 检查forward方法
                                if hasattr(attr, 'forward'):
                                    result['forward_method'] = True

                                # 检查训练相关方法
                                training_methods = ['fit', 'train_epoch', 'train']
                                if any(hasattr(attr, method) for method in training_methods):
                                    result['training_method'] = True

                    if not model_classes:
                        result['issues'].append(f"{py_file.name}: 未找到模型类")
                        result['suggestions'].append(f"在{py_file.name}中定义继承nn.Module的模型类")

            except Exception as e:
                error_msg = f"{py_file.name}: {str(e)}"
                result['issues'].append(error_msg)
                result['suggestions'].append(f"修复{py_file.name}中的导入或语法错误")

        return result

    def test_model_training(self, model_name: str) -> Dict[str, Any]:
        """测试模型训练"""
        result = {
            'data_loading': False,
            'model_initialization': False,
            'forward_pass': False,
            'training_loop': False,
            'issues': [],
            'suggestions': []
        }

        # 加载示例数据
        features, labels = self.load_sample_data()
        if features is None or labels is None:
            result['issues'].append("无法加载示例数据")
            return result

        result['data_loading'] = True

        try:
            model_path = self.models_dir / model_name
            if str(model_path) not in sys.path:
                sys.path.insert(0, str(model_path))

            # 尝试导入主要模块
            main_module = None
            for py_file in model_path.glob('*unified.py'):
                try:
                    module_name = py_file.stem
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        main_module = module
                        break
                except:
                    continue

            if not main_module:
                result['issues'].append("无法导入模型模块")
                return result

            # 查找模型类
            model_class = None
            for attr_name in dir(main_module):
                attr = getattr(main_module, attr_name)
                if hasattr(attr, '__bases__') and hasattr(attr, 'forward'):
                    if 'Model' in attr_name and 'BaseModel' not in attr_name:
                        model_class = attr
                        break

            if not model_class:
                result['issues'].append("未找到模型类")
                return result

            # 初始化模型
            try:
                # 简化模型初始化（实际可能需要更多参数）
                if 'BayesianCNN' in model_name:
                    model = model_class(input_dim=features.shape[1])
                elif 'XGBoost' in model_name:
                    model = model_class()
                else:
                    model = model_class()

                result['model_initialization'] = True
            except Exception as e:
                result['issues'].append(f"模型初始化失败: {e}")
                result['suggestions'].append("检查模型构造函数参数")
                return result

            # 测试前向传播
            try:
                # 准备输入数据
                if hasattr(model, 'forward'):
                    # 转换为tensor
                    if isinstance(features, pd.DataFrame):
                        X_sample = torch.tensor(features.iloc[:32].values, dtype=torch.float32)
                    else:
                        X_sample = torch.tensor(features[:32], dtype=torch.float32)

                    with torch.no_grad():
                        output = model(X_sample)
                        result['forward_pass'] = True
                else:
                    result['issues'].append("模型没有forward方法")
            except Exception as e:
                result['issues'].append(f"前向传播失败: {e}")
                result['suggestions'].append("检查输入数据格式和模型forward方法")

            # 尝试简单的训练循环
            try:
                if hasattr(model, 'parameters'):
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    criterion = torch.nn.CrossEntropyLoss()

                    model.train()
                    for _ in range(1):  # 只训练1步
                        optimizer.zero_grad()
                        output = model(X_sample)
                        y_sample = torch.tensor(labels[:32], dtype=torch.long)
                        loss = criterion(output, y_sample)
                        loss.backward()
                        optimizer.step()

                    result['training_loop'] = True
            except Exception as e:
                result['issues'].append(f"训练循环失败: {e}")

        except Exception as e:
            result['issues'].append(f"测试过程中出错: {e}")

        return result

    def generate_debug_report(self, model_name: str) -> str:
        """生成调试报告"""
        print(f"\n🔍 调试模型: {model_name}")
        print("=" * 60)

        report_lines = [f"# {model_name} 调试报告", ""]

        # 配置诊断
        config_result = self.diagnose_model_config(model_name)
        report_lines.append("## 📋 配置诊断")
        report_lines.append(f"- 配置有效: {'✅' if config_result['config_valid'] else '❌'}")
        report_lines.append(f"- 分类数量正确: {'✅' if config_result['num_classes_correct'] else '❌'}")
        report_lines.append(f"- 输出维度正确: {'✅' if config_result['output_dim_correct'] else '❌'}")
        report_lines.append(f"- 损失函数正确: {'✅' if config_result['loss_function_correct'] else '❌'}")

        if config_result['issues']:
            report_lines.append("\n**问题:**")
            for issue in config_result['issues']:
                report_lines.append(f"- {issue}")

        if config_result['suggestions']:
            report_lines.append("\n**建议:**")
            for suggestion in config_result['suggestions']:
                report_lines.append(f"- {suggestion}")

        # 代码诊断
        code_result = self.diagnose_model_code(model_name)
        report_lines.append("\n## 💻 代码诊断")
        report_lines.append(f"- 代码存在: {'✅' if code_result['code_exists'] else '❌'}")
        report_lines.append(f"- 导入成功: {'✅' if code_result['imports_work'] else '❌'}")
        report_lines.append(f"- 类结构正确: {'✅' if code_result['class_structure'] else '❌'}")
        report_lines.append(f"- 前向方法存在: {'✅' if code_result['forward_method'] else '❌'}")
        report_lines.append(f"- 训练方法存在: {'✅' if code_result['training_method'] else '❌'}")

        if code_result['issues']:
            report_lines.append("\n**问题:**")
            for issue in code_result['issues']:
                report_lines.append(f"- {issue}")

        if code_result['suggestions']:
            report_lines.append("\n**建议:**")
            for suggestion in code_result['suggestions']:
                report_lines.append(f"- {suggestion}")

        # 训练测试
        train_result = self.test_model_training(model_name)
        report_lines.append("\n## 🏃 训练测试")
        report_lines.append(f"- 数据加载: {'✅' if train_result['data_loading'] else '❌'}")
        report_lines.append(f"- 模型初始化: {'✅' if train_result['model_initialization'] else '❌'}")
        report_lines.append(f"- 前向传播: {'✅' if train_result['forward_pass'] else '❌'}")
        report_lines.append(f"- 训练循环: {'✅' if train_result['training_loop'] else '❌'}")

        if train_result['issues']:
            report_lines.append("\n**问题:**")
            for issue in train_result['issues']:
                report_lines.append(f"- {issue}")

        if train_result['suggestions']:
            report_lines.append("\n**建议:**")
            for suggestion in train_result['suggestions']:
                report_lines.append(f"- {suggestion}")

        return "\n".join(report_lines)

    def debug_all_models(self):
        """调试所有模型"""
        print("🚀 开始全面模型调试...")

        for model_name in self.model_names:
            try:
                report = self.generate_debug_report(model_name)
                print(report)

                # 保存单个报告
                report_path = self.models_dir / f"{model_name}_debug_report.md"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                print(f"📄 详细报告已保存: {report_path}")

            except Exception as e:
                print(f"❌ 调试{model_name}时出错: {e}")
                traceback.print_exc()

        print("\n🎉 调试完成!")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="量化交易模型调试工具")
    parser.add_argument('--model', type=str, help='指定调试的模型名称')
    parser.add_argument('--all', action='store_true', help='调试所有模型')

    args = parser.parse_args()

    debugger = ModelDebugger()

    if args.model:
        report = debugger.generate_debug_report(args.model)
        print(report)

        # 保存报告
        report_path = debugger.models_dir / f"{args.model}_debug_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 报告已保存: {report_path}")

    elif args.all:
        debugger.debug_all_models()
    else:
        print("使用方法:")
        print("  python debug_model.py --model <model_name>    # 调试单个模型")
        print("  python debug_model.py --all                  # 调试所有模型")
        print("\n可用模型:")
        for model in debugger.model_names:
            print(f"  - {model}")

if __name__ == "__main__":
    main()
