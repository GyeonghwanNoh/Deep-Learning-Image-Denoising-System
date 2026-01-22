import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import DIV2KDataset
from model import DenoisingNet


def train():
    # ========== 설정 ==========
    num_epochs = 50
    batch_size = 16
    learning_rate = 1e-4
    save_every = 5  # 5 에폭마다 저장
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========== 데이터로더 ==========
    print("\n1. 데이터셋 로딩 중...")
    dataset = DIV2KDataset("./DIV2K_train_HR", start_idx=0, end_idx=750)  # 1~750번 이미지
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"   총 이미지: {len(dataset)}")
    print(f"   배치 개수: {len(train_loader)}")
    
    # ========== 모델 ==========
    print("\n2. 모델 생성 중...")
    model = DenoisingNet().to(device)
    print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== Optimizer & Loss ==========
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    
    # ========== 학습 ==========
    print("\n3. 학습 시작!\n")
    os.makedirs('./checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)  # [B, 4, 64, 64]
            clean = clean.to(device)  # [B, 3, 64, 64]
            
            # Forward
            output = model(noisy)  # [B, 3, 64, 64]
            loss = criterion(output, clean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 진행상황 출력
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")
        
        # 에폭 평균 loss
        avg_loss = epoch_loss / len(train_loader)
        print(f"\n>>> Epoch [{epoch+1}/{num_epochs}] 완료 - Avg Loss: {avg_loss:.6f}\n")
        
        # 모델 저장
        if (epoch + 1) % save_every == 0:
            save_path = f'./checkpoints/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"✓ 모델 저장: {save_path}\n")
    
    # 최종 모델 저장
    final_path = './checkpoints/model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ 최종 모델 저장: {final_path}")
    print("\n=== 학습 완료! ===")


if __name__ == "__main__":
    train()
