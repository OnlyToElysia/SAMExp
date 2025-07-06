import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights   # 导入EfficientNet
import numpy as np
import os
import json

# 定义配置类
class Config:
    
    DATASET = 'cifar10'  
    NUM_CLASSES = 10 if DATASET == 'cifar10' else 100

    
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    SAM_RHO = 0.05
    SAM_ADAPTIVE = False
    IMPROVED_SAM_SMOOTHING = 0.1

    
    DATA_DIR = './data'
    OUTPUT_DIR = './results'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_FILE = os.path.join(OUTPUT_DIR, 'training_log.json')

    
    SELECTED_OPTIMIZER = 'SGD'

    
    RESUME_FROM_CHECKPOINT = True
    RESUME_OPTIMIZER = SELECTED_OPTIMIZER  # 可选: SGD, Adam, SAM

    


cfg = Config()

# 定义SAM优化器
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"无效的rho值: {rho}，应是非负的"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # 应用扰动
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # 恢复原始参数
                p.sub_(self.state[p]["e_w"])  # 应用负扰动

        self.base_optimizer.step()  # 执行优化步骤

        if zero_grad: self.zero_grad()
        for p in self.state.keys():
            del self.state[p]["old_p"]
            del self.state[p]["e_w"]

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization需要传入闭包函数来重新计算损失"
        closure = torch.enable_grad()(closure)  # 确保在计算损失时启用梯度计算

        self.first_step(zero_grad=True)
        closure()  # 计算扰动后的损失
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # 所有参数应该在同一设备上
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# 改进的SAM优化器
class ImprovedSAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, smoothing=0.1, **kwargs):
        super(ImprovedSAM, self).__init__(params, base_optimizer, rho, adaptive, **kwargs)
        self.smoothing = smoothing
        self.iteration = 0
        self.initial_smoothing = smoothing

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        decay_rate = 0.99
        if self.iteration % 10 == 0:
            self.smoothing = max(self.smoothing * decay_rate, 0.01)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if 'ema_grad' not in self.state[p]:
                    self.state[p]['ema_grad'] = torch.zeros_like(p.grad)
                self.state[p]['ema_grad'] = self.smoothing * p.grad + (1 - self.smoothing) * self.state[p]['ema_grad']
                p.grad.data = self.state[p]['ema_grad']
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale_factor = torch.clamp(grad_norm / 10.0, min=0.1, max=10.0)
            scale = group["rho"] / (grad_norm + 1e-12) * scale_factor
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()
        self.iteration += 1

# 训练函数
def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, log_data):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if isinstance(optimizer, SAM) or isinstance(optimizer, ImprovedSAM):
            loss.backward()
            optimizer.first_step(zero_grad=True)  # 第一次更新：应用扰动

            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)  # 第二次更新：在扰动点计算梯度后更新参数
        else:
            # 标准优化器步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler:
        scheduler.step()

    # 记录训练日志
    train_loss /= len(train_loader)
    train_acc = 100. * correct / total
    log_data['train_loss'].append(train_loss)
    log_data['train_acc'].append(train_acc)

    return train_loss, train_acc

# 测试函数
def test(model, test_loader, criterion, device, epoch, log_data):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 记录测试日志
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    log_data['test_loss'].append(test_loss)
    log_data['test_acc'].append(test_acc)

    return test_loss, test_acc

# 保存检查点
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, log_data, optimizer_name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'log_data': log_data
    }

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f'{optimizer_name}_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)

# 加载检查点
def load_checkpoint(model, optimizer, scheduler, optimizer_name):
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f'{optimizer_name}_checkpoint.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    log_data = checkpoint['log_data']

    return model, optimizer, scheduler, start_epoch, best_acc, log_data

# 主函数
def main():
    # 创建保存目录
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}", end='')

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if cfg.DATASET == 'cifar10' else
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if cfg.DATASET == 'cifar10' else
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 加载数据集
    if cfg.DATASET == 'cifar10':
        train_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=False, download=True, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root=cfg.DATA_DIR, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=cfg.DATA_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    # 手动指定优化器
    optimizer_name = cfg.SELECTED_OPTIMIZER
    print(f"\n开始使用{optimizer_name}优化器训练...")

    # 初始化模型
    model = efficientnet_b0(weights=None)  # 初始化EfficientNet
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, cfg.NUM_CLASSES)  # 修改最后一层的输出维度
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 创建优化器
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE/10, weight_decay=cfg.WEIGHT_DECAY)
    elif optimizer_name == 'SAM':
        optimizer = SAM(model.parameters(), optim.SGD, rho=cfg.SAM_RHO, adaptive=cfg.SAM_ADAPTIVE,
                        lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    elif optimizer_name == 'ImprovedSAM':
        optimizer = ImprovedSAM(model.parameters(), optim.SGD, rho=cfg.SAM_RHO, adaptive=cfg.SAM_ADAPTIVE,
                                smoothing=cfg.IMPROVED_SAM_SMOOTHING, lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    log_data = {
        'optimizer': optimizer_name,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    start_epoch = 1
    best_acc = 0.0

    if cfg.RESUME_FROM_CHECKPOINT and optimizer_name == cfg.RESUME_OPTIMIZER:
        try:
            model, optimizer, scheduler, start_epoch, best_acc, log_data = load_checkpoint(
                model, optimizer, scheduler, optimizer_name)
        except FileNotFoundError as e:
            print(f"错误: {e}. 将从头开始训练.", end='')

    # 训练和测试循环
    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, epoch, log_data)
        test_loss, test_acc = test(model, test_loader, criterion, device, epoch, log_data)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f'{optimizer_name}_best_model.pth'))

        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, log_data, optimizer_name)

        with open(cfg.LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=4)

        print(f"第 {epoch} 轮 | 训练集 - 损失: {train_loss:.3f} | 准确率: {train_acc:.2f}% | 测试集 - 损失: {test_loss:.3f} | 准确率: {test_acc:.2f}%")

    print(f"{optimizer_name}优化器训练完成！最佳测试准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main()