import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class DIV2KDataset(Dataset):
    def __init__(self, folder, start_idx=None, end_idx=None, augmentation=True):
        self.folder = folder
        self.augmentation = augmentation
        all_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
        
        # 인덱스 범위 지정
        if start_idx is not None or end_idx is not None:
            start_idx = start_idx if start_idx is not None else 0
            end_idx = end_idx if end_idx is not None else len(all_files)
            self.files = all_files[start_idx:end_idx]
        else:
            self.files = all_files
    
    def __len__(self):
        return len(self.files)
    
    def apply_augmentation(self, patch_tensor):
        """데이터 증강 적용 (clean 이미지만)"""
        # Shape: [3, 64, 64]
        
        # 1. 좌우 반전 (50% 확률)
        if random.random() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[2])
        
        # 2. 상하 반전 (50% 확률)
        if random.random() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[1])
        
        # 3. 90도 회전 (0, 90, 180, 270도 중 랜덤)
        k = random.randint(0, 3)
        if k > 0:
            patch_tensor = torch.rot90(patch_tensor, k, dims=[1, 2])
        
        return patch_tensor
    
    def __getitem__(self, idx):
        # 이미지 열기
        img = Image.open(f"{self.folder}/{self.files[idx]}")
        
        # 랜덤 위치에서 40x40 패치 자르기 (DnCNN 표준)
        x = random.randint(0, img.width - 40)
        y = random.randint(0, img.height - 40)
        patch = img.crop((x, y, x + 40, y + 40))
        
        # 텐서로 변환 (0~1 범위)
        patch_array = np.array(patch)
        clean_tensor = torch.from_numpy(patch_array).permute(2, 0, 1).float() / 255.0
        
        # 랜덤 노이즈 레벨 (5~60)
        noise_level = random.uniform(5, 60)
        
        # 데이터 증강 먼저 적용 (clean 이미지만!)
        if self.augmentation:
            clean_tensor = self.apply_augmentation(clean_tensor)
        
        # 그 다음 가우시안 노이즈 추가 (augmentation 후!)
        noise = torch.randn_like(clean_tensor) * (noise_level / 255.0)
        noisy_tensor = clean_tensor + noise
        
        # Noise level map 생성 및 추가 (4채널)
        noise_map = torch.ones(1, 40, 40) * (noise_level / 255.0)
        noisy_with_map = torch.cat([noisy_tensor, noise_map], dim=0)
        
        return noisy_with_map, clean_tensor


# # 테스트 코드
# if __name__ == "__main__":
#     dataset = DIV2KDataset("./dncnn-denoising/DIV2K_train_HR")
#     print(f"이미지 개수: {len(dataset)}")
    
#     if len(dataset) > 0:
#         patch = dataset[0]
#         print(f"첫 번째 패치 shape: {patch.shape}")
        
#         # 이미지 보기
#         import matplotlib.pyplot as plt
#         plt.imshow(patch.permute(1, 2, 0))
#         plt.title("First 64x64 Patch")
#         plt.show()
#     else:
#         print("이미지 없음!")