import json
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def read_json_file(file_path):
    """读取JSON文件并返回数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        sys.exit(1)

def plot_test_accuracy(data_list, file_names, output_path=None):
    """绘制四种模型的测试准确率对比图"""
    plt.figure(figsize=(12, 8))
    
    for i, data in enumerate(data_list):
        optimizer = data.get('optimizer', f'模型{i+1}')
        test_acc = data.get('test_acc', [])
        
        # 确保有足够的测试准确率数据点
        if len(test_acc) < 100:
            print(f"警告：{optimizer} 的测试准确率数据不足100轮")
            test_acc_to_plot = test_acc
        else:
            test_acc_to_plot = test_acc[:100]
        
        # 绘制折线图
        plt.plot(range(1, len(test_acc_to_plot) + 1), test_acc_to_plot, 
                 label=optimizer, linewidth=2)
    
    plt.title('四种模型前100轮测试集准确率对比', fontsize=16)
    plt.xlabel('训练轮次', fontsize=14)
    plt.ylabel('测试准确率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"图表已保存至 {output_path}")
    
    plt.show()

def main():
    # 定义JSON文件路径
    json_files = [
        'training_logsgd.json',
        'training_logadam.json',
        'training_logsam.json',
        'training_logIsam.json'
    ]
    # json_files = [
    #     'SGD_training_log.json',
    #     'Adam_training_log.json',
    #     'SAM_training_log.json',
    #     #'ImprovedSAM_training_log.json'
    # ]
    
    # 读取所有JSON文件的数据
    all_data = []
    file_names = []
    
    for file_path in json_files:
        data = read_json_file(file_path)
        all_data.append(data)
        file_names.append(os.path.basename(file_path))
    
    # 绘制测试准确率对比图
    plot_test_accuracy(all_data, file_names, output_path='test_accuracy_comparison.png')

if __name__ == "__main__":
    main()    