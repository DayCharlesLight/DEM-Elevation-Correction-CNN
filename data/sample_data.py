import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DEMDataset(Dataset):
    def __init__(self, dem_paths, reference_paths, patch_size=256, transform=None):
        self.dem_paths = dem_paths
        self.reference_paths = reference_paths
        self.patch_size = patch_size
        self.transform = transform
    
    def __len__(self):
        return len(self.dem_paths)
    
    def __getitem__(self, idx):
        if self.dem_paths[idx].endswith('.npy'):
            dem_data = np.load(self.dem_paths[idx]).astype(np.float32)
            ref_data = np.load(self.reference_paths[idx]).astype(np.float32)
        else:
            import rasterio
            with rasterio.open(self.dem_paths[idx]) as src:
                dem_data = src.read(1).astype(np.float32)
            with rasterio.open(self.reference_paths[idx]) as src:
                ref_data = src.read(1).astype(np.float32)
        
        dem_normalized = self._normalize(dem_data)
        ref_normalized = self._normalize(ref_data)
        dem_patch, ref_patch = self._random_crop(dem_normalized, ref_normalized)
        
        dem_tensor = torch.from_numpy(dem_patch).unsqueeze(0)
        ref_tensor = torch.from_numpy(ref_patch).unsqueeze(0)
        
        return dem_tensor, ref_tensor
    
    def _normalize(self, data):
        valid_mask = ~np.isnan(data)
        if valid_mask.sum() == 0:
            return np.zeros_like(data)
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        normalized = (data - min_val) / (max_val - min_val)
        normalized[~valid_mask] = 0
        return normalized
    
    def _random_crop(self, dem, ref):
        h, w = dem.shape
        if h < self.patch_size or w < self.patch_size:
            dem = np.pad(dem, ((0, max(0, self.patch_size - h)), (0, max(0, self.patch_size - w))), mode='constant')
            ref = np.pad(ref, ((0, max(0, self.patch_size - h)), (0, max(0, self.patch_size - w))), mode='constant')
        h, w = dem.shape
        y = np.random.randint(0, h - self.patch_size + 1)
        x = np.random.randint(0, w - self.patch_size + 1)
        return dem[y:y+self.patch_size, x:x+self.patch_size], ref[y:y+self.patch_size, x:x+self.patch_size]

def generate_sample_dem_data(num_samples=10, size=256):
    dem_data = []
    ref_data = []
    for i in range(num_samples):
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)
        dem = 100 + 50 * np.sin(X/2) * np.cos(Y/2) + np.random.normal(0, 5, (size, size))
        ref = 100 + 50 * np.sin(X/2) * np.cos(Y/2)
        dem_data.append(dem.astype(np.float32))
        ref_data.append(ref.astype(np.float32))
    return dem_data, ref_data
