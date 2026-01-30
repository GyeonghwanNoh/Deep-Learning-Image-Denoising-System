import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import DIV2KDataset
from model import DenoisingNet
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_psnr(img1, img2):
    """PSNR 계산"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def validate(model, val_folder, noise_level, device, num_samples=20):
    """Validation 세트로 PSNR 측정"""
    model.eval()
    total_psnr = 0
    
    import random
    files = sorted([f for f in os.listdir(val_folder) if f.endswith('.png')])
    val_files = random.sample(files, min(num_samples, len(files)))
    
    with torch.no_grad():
        for img_file in val_files:
            # 이미지 로드
            img = Image.open(os.path.join(val_folder, img_file)).convert('RGB')
            clean = np.array(img).astype(np.float32) / 255.0
            
            # 노이즈 추가
            noise = np.random.randn(*clean.shape) * (noise_level / 255.0)
            noisy = clean + noise
            
            # 텐서 변환
            noisy_t = torch.from_numpy(noisy).permute(2, 0, 1).float()
            noise_map = torch.ones(1, *noisy_t.shape[1:]) * (noise_level / 255.0)
            input_t = torch.cat([noisy_t, noise_map], dim=0).unsqueeze(0).to(device)
            
            # 추론
            output = model(input_t)[0].cpu().permute(1, 2, 0).numpy()
            
            # PSNR 계산
            total_psnr += calculate_psnr(output, clean)
    model.train()
    return total_psnr / len(val_files)


def train():
    # ========== 설정 ==========
    num_epochs = 500  # 200 → 500 (더 많은 iteration)
    batch_size = 64   # 16 → 64 (DnCNN 표준)
    lr = 1e-4  # KAIR 설정
    save_every = 20   # 저장 주기 조정
    noise_level = 25
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 데이터 로드 (Augmentation ON)
    dataset = DIV2KDataset("./DIV2K_train_HR", start_idx=0, end_idx=800, augmentation=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=4, pin_memory=True)
    print(f"Images: {len(dataset)}, Batches: {len(loader)}\n")
    
    # 모델
    model = DenoisingNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4, 
                                   betas=(0.9, 0.999))
    criterion = nn.MSELoss()  # MSE Loss
    
    # MultiStepLR Scheduler (500 epoch 기준)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 350, 450], gamma=0.5)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 학습
    os.makedirs('./checkpoints', exist_ok=True)
    losses, psnrs = [], []
    best_psnr, best_epoch = 0, 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(loader):
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
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(loader)}] "
                      f"Loss: {loss.item():.6f} "
                      f"LR: {current_lr:.2e}")
        
        # 에폭 평균 loss
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        
        # Validation
        val_psnr = validate(model, './DIV2K_valid_HR', noise_level, device)
        psnrs.append(val_psnr)
        
        # Scheduler step (MultiStep)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n>>> Epoch [{epoch+1}/{num_epochs}] 완료")
        print(f"    Train Loss: {avg_loss:.6f}")
        print(f"    Val PSNR: {val_psnr:.2f} dB")
        print(f"    LR: {current_lr:.2e}\n")
        
        # Best 모델 저장
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), './checkpoints/model_best.pth')
            print(f"✓ Best 모델 저장! (Epoch {best_epoch}, PSNR: {best_psnr:.2f} dB)\n")
        
        # 주기적 모델 저장 및 그래프 업데이트
        if (epoch + 1) % save_every == 0:
            save_path = f'./checkpoints/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"✓ 체크포인트 저장: {save_path}")
            
            # 중간 그래프 저장
            plt.figure(figsize=(14, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(range(1, epoch + 2), losses, 'b-', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Training Loss', fontsize=12)
            plt.title('Training Loss Curve', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, epoch + 2), psnrs, 'g-', linewidth=2)
            if best_epoch > 0:
                plt.axvline(best_epoch, color='r', linestyle='--', linewidth=2, label=f'Best: Epoch {best_epoch}')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Validation PSNR (dB)', fontsize=12)
            plt.title('Validation PSNR Curve', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./training_curve.png', dpi=150)
            plt.close()
            print(f"✓ 그래프 업데이트: ./training_curve.png\n")
    
    # 최종 저장
    torch.save(model.state_dict(), './checkpoints/model_final.pth')
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best: Epoch {best_epoch} ({best_psnr:.2f} dB)")
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    train()

