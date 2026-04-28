import os
import numpy as np
import rasterio
import pandas as pd
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset, DataLoader
# ======== Part 1. 设置参数与准备插值 ========
# 输入文件
DSM_FILE = "dsm.tif"
SLOPE_FILE = "slope.tif"
ASPECT_FILE = "aspect.tif"
CONTROL_FILE = "control_points.csv"  # 列包含 x, y, z (或 lon, lat, z)
TRUTH_TIF = "truth.tif"
PATCH_SIZE = 128
STRIDE = 64
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ======== Part 2. 控制点插值为truth影像 ========
def interpolate_control_points_to_raster(dsm_path, ctrl_path, out_path):
    with rasterio.open(dsm_path) as src:
        dsm_data = src.read(1)
        profile = src.profile
        width, height = src.width, src.height
        x0, y0 = src.transform * (0, 0)
        cellsize = src.transform[0]
        xs = np.arange(width) * cellsize + x0
        ys = np.arange(height) * cellsize + y0
        grid_x, grid_y = np.meshgrid(xs, ys)
    df = pd.read_csv(ctrl_path)
    ctrl_x = df.iloc[:, 0].values
    ctrl_y = df.iloc[:, 1].values
    ctrl_z = df.iloc[:, 2].values
    grid_z = griddata(
        (ctrl_x, ctrl_y), ctrl_z, 
        (grid_x, grid_y), method='linear', fill_value=np.nan
    )
    grid_z[np.isnan(grid_z)] = np.nanmean(ctrl_z)
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(grid_z.astype(np.float32), 1)
if not os.path.exists(TRUTH_TIF):
    print("进行控制点插值...")
    interpolate_control_points_to_raster(DSM_FILE, CONTROL_FILE, TRUTH_TIF)
# ======== Part 3. Dataset 定义 =========
class PatchDataset(Dataset):
    def __init__(self, dsm, slope, aspect, truth, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride
        self.inputs = np.stack([dsm, slope, aspect], axis=0)
        self.truth = truth
        self.patch_indices = []
        h, w = dsm.shape
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                self.patch_indices.append((y, x))
    def __len__(self):
        return len(self.patch_indices)
    def __getitem__(self, idx):
        y, x = self.patch_indices[idx]
        inp = self.inputs[:, y:y+self.patch_size, x:x+self.patch_size]
        label = self.truth[y:y+self.patch_size, x:x+self.patch_size]
        inp = (inp - np.mean(inp, axis=(1,2), keepdims=True)) / (np.std(inp, axis=(1,2), keepdims=True) + 1e-6)
        label = (label - np.mean(label)) / (np.std(label) + 1e-6)
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)
def load_image(fn):
    with rasterio.open(fn) as src:
        return src.read(1)
dsm = load_image(DSM_FILE)
slope = load_image(SLOPE_FILE)
aspect = load_image(ASPECT_FILE)
truth = load_image(TRUTH_TIF)
dataset = PatchDataset(dsm, slope, aspect, truth, PATCH_SIZE, STRIDE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# ======== Part 5. 定义网络和训练循环 ========
from model import UNet  # 假设仓库提供 model.py 和 UNet 定义
model = UNet(n_channels=3, n_classes=1).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)
crit = torch.nn.MSELoss()
print("训练样本总数：", len(dataset))
print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, Y in dataloader:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        pred = model(X)
        loss = crit(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * X.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss / len(dataset):.4f}")
    torch.save(model.state_dict(), f"unet_epoch{epoch+1}.pth")
print("训练完成，模型已保存。")
# ============ Part 6. 大影像预测 ========
def predict_on_large_image(model, dsm_path, slope_path, aspect_path, out_path,
                           patch_size=128, stride=64, device='cpu'):
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
        profile = src.profile
    slope = load_image(slope_path)
    aspect = load_image(aspect_path)
    h, w = dsm.shape
    out = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = np.stack([
                dsm[y:y+patch_size, x:x+patch_size],
                slope[y:y+patch_size, x:x+patch_size],
                aspect[y:y+patch_size, x:x+patch_size]
            ], axis=0)
            patch_norm = (patch - np.mean(patch, axis=(1,2), keepdims=True)) / (np.std(patch, axis=(1,2), keepdims=True) + 1e-6)
            patch_tensor = torch.tensor(patch_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(patch_tensor)
                pred = pred.squeeze().cpu().numpy()
            out[y:y+patch_size, x:x+patch_size] += pred
            count[y:y+patch_size, x:x+patch_size] += 1
    mask = count > 0
    out[mask] /= count[mask]
    out[~mask] = np.nan
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(out.astype(np.float32), 1)
    print("输出已保存到：", out_path)
# ============ 使用方法 ==============
# 测试数据（改成你自己的测试文件即可）
TEST_DSM = "test_dsm.tif"
TEST_SLOPE = "test_slope.tif"
TEST_ASPECT = "test_aspect.tif"
OUT_TIF = "test_corrected.tif"
MODEL_PATH = "unet_epoch20.pth" # 或你具体要用的权重
if os.path.exists(TEST_DSM) and os.path.exists(TEST_SLOPE) and os.path.exists(TEST_ASPECT) and os.path.exists(MODEL_PATH):
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    predict_on_large_image(
        model, 
        TEST_DSM, 
        TEST_SLOPE, 
        TEST_ASPECT, 
        OUT_TIF, 
        patch_size=PATCH_SIZE, 
        stride=STRIDE, 
        device=DEVICE
    )
else:
    print("如需测试推理，请放置模型权重以及test_dsm.tif等测试输入文件。")
