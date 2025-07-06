import json
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 超参数配置
HYPERPARAMS = {
    'input_file': 'training_logIsam.json',  # 输入JSON文件路径
    'output_dir': '',     # 输出目录（留空则与输入文件同目录）
    'auto_name_output': True,  # 是否自动使用输入文件名作为输出文件名
    'max_epochs': 100,    # 最大轮次数
    'show_grid': True,    # 是否显示网格线
    'fig_width': 12,      # 图表宽度
    'fig_height': 6,      # 图表高度
    'train_color': 'blue',  # 训练集准确率曲线颜色
    'test_color': 'red',    # 测试集准确率曲线颜色
    'line_style': '-'       # 线样式（'-'表示实线）
}

def load_json_data(file_path: str) -> Dict:
    """
    从指定路径加载JSON数据
    
    参数:
        file_path: JSON文件路径
    
    返回:
        解析后的JSON数据字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        raise

def extract_accuracy_data(data: Dict, max_epochs: int) -> Tuple[List[float], List[float]]:
    """
    从JSON数据中提取训练集和测试集的准确率数据
    
    参数:
        data: 包含准确率数据的字典
        max_epochs: 最大轮次数
    
    返回:
        训练集准确率列表和测试集准确率列表
    """
    # 获取训练集和测试集准确率数据
    train_acc = data.get('train_acc', [])[:max_epochs]
    test_acc = data.get('test_acc', [])[:max_epochs]
    
    # 验证数据完整性
    if not train_acc or not test_acc:
        raise ValueError("数据中缺少训练集或测试集准确率信息")
    
    # 如果数据点少于max_epochs个，用最后一个值填充剩余位置
    if len(train_acc) < max_epochs:
        train_acc.extend([train_acc[-1]] * (max_epochs - len(train_acc)))
    if len(test_acc) < max_epochs:
        test_acc.extend([test_acc[-1]] * (max_epochs - len(test_acc)))
    
    return train_acc, test_acc

def get_output_path(input_path: str) -> Optional[str]:
    """
    根据输入文件路径生成输出文件路径
    
    参数:
        input_path: 输入文件路径
    
    返回:
        输出文件路径，如果不自动命名则返回None
    """
    if not HYPERPARAMS['auto_name_output']:
        return HYPERPARAMS['output_file']
    
    # 获取输入文件的基本名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 确定输出目录
    output_dir = HYPERPARAMS['output_dir']
    if not output_dir:
        # 如果输出目录为空，则使用输入文件所在目录
        output_dir = os.path.dirname(input_path)
    
    # 构建输出路径
    output_path = os.path.join(output_dir, f"{base_name}_accuracy.png")
    return output_path

def plot_accuracy(train_acc: List[float], test_acc: List[float], 
                  optimizer: Optional[str] = None, output_path: Optional[str] = None) -> None:
    """
    绘制训练集和测试集准确率对比图
    
    参数:
        train_acc: 训练集准确率列表
        test_acc: 测试集准确率列表
        optimizer: 优化器名称，用于图表标题
        output_path: 图表保存路径，若为None则显示图表
    """
    epochs = list(range(1, HYPERPARAMS['max_epochs'] + 1))
    
    plt.figure(figsize=(HYPERPARAMS['fig_width'], HYPERPARAMS['fig_height']))
    
    # 修正：分开指定颜色和线型参数
    plt.plot(epochs, train_acc, color=HYPERPARAMS["train_color"], linestyle=HYPERPARAMS["line_style"], label='训练集准确率')
    plt.plot(epochs, test_acc, color=HYPERPARAMS["test_color"], linestyle=HYPERPARAMS["line_style"], label='测试集准确率')
    
    plt.title(f'模型准确率对比（{"优化器: " + optimizer if optimizer else ""}）')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.grid(HYPERPARAMS['show_grid'])
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path)
        print(f"图表已保存至 {output_path}")
    else:
        plt.show()

def main():
    """
    主函数：执行可视化流程
    """
    try:
        input_file = HYPERPARAMS['input_file']
        
        # 加载数据
        data = load_json_data(input_file)
        
        # 提取优化器名称（如果有）
        optimizer = data.get('optimizer')
        
        # 提取准确率数据
        train_acc, test_acc = extract_accuracy_data(data, HYPERPARAMS['max_epochs'])
        
        # 确定输出路径
        output_path = get_output_path(input_file) if HYPERPARAMS['auto_name_output'] else HYPERPARAMS['output_file']
        
        # 绘制并保存/显示图表
        plot_accuracy(train_acc, test_acc, optimizer, output_path)
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()