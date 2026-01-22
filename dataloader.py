import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class DIV2KDataset(Dataset):
    def __init__(self, folder, start_idx=None, end_idx=None):
        self.folder = folder
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
    
    def __getitem__(self, idx):
        # 이미지 열기
        img = Image.open(f"{self.folder}/{self.files[idx]}")
        
        # 랜덤 위치에서 64x64 패치 자르기
        x = random.randint(0, img.width - 64)
        y = random.randint(0, img.height - 64)
        patch = img.crop((x, y, x + 64, y + 64))
        
        # 텐서로 변환 (0~1 범위)
        patch_array = np.array(patch)
        patch_tensor = torch.from_numpy(patch_array).permute(2, 0, 1).float() / 255.0
        
        # 랜덤 노이즈 레벨 (5~60)
        noise_level = random.uniform(5, 60)
        
        # 가우시안 노이즈 추가
        noise = torch.randn_like(patch_tensor) * (noise_level / 255.0)
        noisy = torch.clamp(patch_tensor + noise, 0, 1)
        
        # Noise level map 생성 및 추가 (4채널)
        noise_map = torch.ones(1, 64, 64) * (noise_level / 255.0)
        noisy_with_map = torch.cat([noisy, noise_map], dim=0) # cat = concatenation
        
        return noisy_with_map, patch_tensor


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