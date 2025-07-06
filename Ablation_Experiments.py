import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
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
    IMPROVED_SAM_RHO_DECAY = 0.99
    IMPROVED_SAM_MIN_RHO = 0.01
    IMPROVED_SAM_WARMUP_EPOCHS = 5

    
    DATA_DIR = './data'
    OUTPUT_DIR = './results'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_FILE = os.path.join(OUTPUT_DIR, 'training_log.json')

    
    SELECTED_OPTIMIZER = 'ImprovedSAM'

    
    RESUME_FROM_CHECKPOINT = True
    RESUME_OPTIMIZER = SELECTED_OPTIMIZER


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
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        for p in self.state.keys():
            if "old_p" in self.state[p]:
                del self.state[p]["old_p"]
            if "e_w" in self.state[p]:
                del self.state[p]["e_w"]

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization需要传入闭包函数来重新计算损失"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
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
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, 
                 smoothing=0.1, rho_decay=0.99, min_rho=0.01, warmup_epochs=5, 
                 disable_adaptive=False, disable_smoothing=False, disable_layered=False, **kwargs):
        super(ImprovedSAM, self).__init__(params, base_optimizer, rho, adaptive, **kwargs)
        self.smoothing = smoothing
        self.initial_rho = rho
        self.rho_decay = rho_decay
        self.min_rho = min_rho
        self.warmup_epochs = warmup_epochs
        self.iteration = 0
        self.epoch = 0
        self.disable_adaptive = disable_adaptive
        self.disable_smoothing = disable_smoothing
        self.disable_layered = disable_layered

    def set_epoch(self, epoch):
        """设置当前epoch，用于预热调度"""
        self.epoch = epoch

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # 1. 自适应扰动强度
        if not self.disable_adaptive:
            # 根据迭代次数指数衰减
            rho_decay_factor = self.rho_decay ** self.iteration
            current_rho = max(self.initial_rho * rho_decay_factor, self.min_rho)
            
            # 预热期线性增加rho
            if self.epoch < self.warmup_epochs:
                warmup_factor = (self.epoch + 1) / self.warmup_epochs
                current_rho = min(current_rho, self.initial_rho * warmup_factor)
            
            # 更新所有参数组的rho
            for group in self.param_groups:
                group['rho'] = current_rho
        else:
            for group in self.param_groups:
                group['rho'] = self.initial_rho

        # 2. 智能梯度平滑
        if not self.disable_smoothing:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: 
                        continue
                        
                    # 初始化EMA梯度
                    if 'ema_grad' not in self.state[p]:
                        self.state[p]['ema_grad'] = p.grad.clone().detach()
                    
                    # 应用EMA平滑
                    self.state[p]['ema_grad'] = self.smoothing * p.grad + (1 - self.smoothing) * self.state[p]['ema_grad']
                    p.grad.data = self.state[p]['ema_grad'].clone()

        # 3. 分层扰动策略
        if not self.disable_layered:
            # 计算每层的梯度范数
            layer_grad_norms = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: 
                        continue
                    grad_norm = p.grad.data.norm(p=2).item()
                    layer_grad_norms.append(grad_norm)
            
            # 计算梯度范数的中位数
            if layer_grad_norms:
                median_grad_norm = np.median(layer_grad_norms)
            else:
                median_grad_norm = 1e-6
        else:
            median_grad_norm = 1.0

        # 4. 计算全局梯度范数
        grad_norm = self._grad_norm()
        
        # 应用扰动
        param_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                
                if not self.disable_layered:
                    # 分层扰动缩放因子
                    layer_scale = median_grad_norm / (layer_grad_norms[param_idx] + 1e-12)
                    layer_scale = torch.clamp(torch.tensor(layer_scale), 0.1, 10.0).to(p.device)
                else:
                    layer_scale = 1.0
                
                # 全局扰动缩放
                scale = group["rho"] / (grad_norm + 1e-12) * layer_scale
                
                # 保存原始参数并应用扰动
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
                
                param_idx += 1

        if zero_grad: 
            self.zero_grad()
        
        self.iteration += 1

# 训练函数
def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, log_data):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 为ImprovedSAM设置当前epoch
    if isinstance(optimizer, ImprovedSAM):
        optimizer.set_epoch(epoch)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if isinstance(optimizer, SAM) or isinstance(optimizer, ImprovedSAM):
            loss.backward()
            optimizer.first_step(zero_grad=True)

            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler:
        scheduler.step()

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
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    log_data = checkpoint['log_data']

    return model, optimizer, scheduler, start_epoch, best_acc, log_data

def main():
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

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

    if cfg.DATASET == 'cifar10':
        train_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=False, download=True, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root=cfg.DATA_DIR, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=cfg.DATA_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    ablation_settings = [
        {"disable_adaptive": True, "disable_smoothing": False, "disable_layered": False, "name": "ImprovedSAM_no_adaptive"},
        {"disable_adaptive": False, "disable_smoothing": True, "disable_layered": False, "name": "ImprovedSAM_no_smoothing"},
        {"disable_adaptive": False, "disable_smoothing": False, "disable_layered": True, "name": "ImprovedSAM_no_layered"}
    ]

    for setting in ablation_settings:
        optimizer_name = setting["name"]
        print(f"\n开始使用{optimizer_name}优化器训练...")

        model = resnet18(num_classes=cfg.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = ImprovedSAM(
            model.parameters(), 
            optim.SGD, 
            rho=cfg.SAM_RHO, 
            adaptive=cfg.SAM_ADAPTIVE,
            smoothing=cfg.IMPROVED_SAM_SMOOTHING,
            rho_decay=cfg.IMPROVED_SAM_RHO_DECAY,
            min_rho=cfg.IMPROVED_SAM_MIN_RHO,
            warmup_epochs=cfg.IMPROVED_SAM_WARMUP_EPOCHS,
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM, 
            weight_decay=cfg.WEIGHT_DECAY,
            disable_adaptive=setting["disable_adaptive"],
            disable_smoothing=setting["disable_smoothing"],
            disable_layered=setting["disable_layered"]
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        log_data = {
            'optimizer': optimizer_name,
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'best_acc': 0.0
        }

        start_epoch = 1
        best_acc = 0.0

        if cfg.RESUME_FROM_CHECKPOINT and optimizer_name == cfg.RESUME_OPTIMIZER:
            try:
                model, optimizer, scheduler, start_epoch, best_acc, log_data = load_checkpoint(
                    model, optimizer, scheduler, optimizer_name)
                print(f"从检查点恢复训练: 起始轮次 {start_epoch}, 最佳准确率 {best_acc:.2f}%")
            except FileNotFoundError as e:
                print(f"错误: {e}. 将从头开始训练.")

        for epoch in range(start_epoch, cfg.EPOCHS + 1):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, epoch, log_data)
            test_loss, test_acc = test(model, test_loader, criterion, device, epoch, log_data)

            if test_acc > best_acc:
                best_acc = test_acc
                log_data['best_acc'] = best_acc
                torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f'{optimizer_name}_best_model.pth'))
                print(f"新的最佳模型! 准确率: {best_acc:.2f}%")

            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, log_data, optimizer_name)

            with open(cfg.LOG_FILE.replace('.json', f'_{optimizer_name}.json'), 'w') as f:
                json.dump(log_data, f, indent=4)

            print(f"第 {epoch} 轮 | 训练集 - 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}% | 测试集 - 损失: {test_loss:.4f} | 准确率: {test_acc:.2f}%")

        print(f"{optimizer_name}优化器训练完成！最佳测试准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main()