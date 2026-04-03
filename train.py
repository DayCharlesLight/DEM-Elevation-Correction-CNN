import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np

from models.dem_cnn_model import DEMCorrectionNet
from data.sample_data import DEMDataset, generate_sample_dem_data, get_data_loaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for dem, ref in tqdm(train_loader, desc="Training"):
        dem = dem.to(device)
        ref = ref.to(device)
        
        # 前向传播
        outputs = model(dem)
        loss = criterion(outputs, ref)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for dem, ref in tqdm(val_loader, desc="Validating"):
            dem = dem.to(device)
            ref = ref.to(device)
            
            outputs = model(dem)
            loss = criterion(outputs, ref)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 创建TensorBoard写入器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'logs/dem_correction_{timestamp}'
    writer = SummaryWriter(log_dir)
    
    # 生成或加载数据
    print("生成示例数据...")
    dem_data, ref_data = generate_sample_dem_data(num_samples=20)
    
    # 保存为临时路径（实际应用中使用真实路径）
    dem_paths = [f'temp_dem_{i}.npy' for i in range(len(dem_data))]
    ref_paths = [f'temp_ref_{i}.npy' for i in range(len(ref_data))]
    
    for i, (dem, ref) in enumerate(zip(dem_data, ref_data)):
        np.save(dem_paths[i], dem)
        np.save(ref_paths[i], ref)
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataset = DEMDataset(dem_paths, ref_paths, patch_size=256)
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # 初始化模型
    print("初始化模型...")
    model = DEMCorrectionNet(in_channels=1, out_channels=1)
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = f'checkpoints/best_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ 模型已保存: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停
        if patience_counter >= args.patience:
            print(f"早停：验证损失在{args.patience}个epoch内未改进")
            break
    
    # 清理临时文件
    import os
    for path in dem_paths + ref_paths:
        if os.path.exists(path):
            os.remove(path)
    
    writer.close()
    print("\n训练完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEM高程校正CNN训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    args = parser.parse_args()
    
    main(args)
