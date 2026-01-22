import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import DenoisingNet


def calculate_psnr(img1, img2):
    """PSNR 계산 (0-1 범위)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def test_image(model, img_path, noise_level=25, device='cpu'):
    """단일 이미지 테스트"""
    # 이미지 로드
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 노이즈 추가
    np.random.seed(0)
    noise = np.random.randn(*img_np.shape) * (noise_level / 255.0)
    noisy_np = np.clip(img_np + noise, 0, 1)
    
    # 텐서로 변환
    clean_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1).float()
    
    # Noise level map 추가
    noise_map = torch.ones(1, *noisy_tensor.shape[1:]) * (noise_level / 255.0)
    noisy_with_map = torch.cat([noisy_tensor, noise_map], dim=0).unsqueeze(0).to(device)
    
    # Denoising
    with torch.no_grad():
        output = model(noisy_with_map)
    
    # 결과 변환
    output_np = output[0].cpu().permute(1, 2, 0).numpy()
    output_np = np.clip(output_np, 0, 1)
    
    # PSNR 계산
    psnr_noisy = calculate_psnr(noisy_np, img_np)
    psnr_output = calculate_psnr(output_np, img_np)
    
    return img_np, noisy_np, output_np, psnr_noisy, psnr_output


def test():
    # ========== 설정 ==========
    model_path = './checkpoints/model_final.pth'
    test_img_path = './DIV2K_train_HR'  # 테스트 이미지 폴더
    noise_level = 25
    save_dir = './test_results'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ========== 모델 로드 ==========
    print("1. 모델 로딩 중...")
    model = DenoisingNet().to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("   먼저 train.py를 실행하세요!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ 모델 로드 완료: {model_path}\n")
    
    # ========== 테스트 ==========
    os.makedirs(save_dir, exist_ok=True)
    
    # 테스트 이미지 목록 (751~800번)
    all_files = sorted([f for f in os.listdir(test_img_path) if f.endswith('.png')])
    img_files = all_files[750:800]  # 751~800번 이미지
    
    if len(img_files) == 0:
        print(f"❌ 테스트 이미지가 없습니다: {test_img_path}")
        return
    
    print(f"2. 테스트 시작 (총 {len(img_files)}장)\n")
    
    total_psnr_noisy = 0
    total_psnr_output = 0
    
    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(test_img_path, img_file)
        print(f"[{idx+1}/{len(img_files)}] {img_file}")
        
        # 테스트
        clean, noisy, output, psnr_noisy, psnr_output = test_image(
            model, img_path, noise_level, device
        )
        
        total_psnr_noisy += psnr_noisy
        total_psnr_output += psnr_output
        
        print(f"   Noisy PSNR: {psnr_noisy:.2f} dB")
        print(f"   Output PSNR: {psnr_output:.2f} dB")
        print(f"   개선: {psnr_output - psnr_noisy:.2f} dB\n")
        
        # 결과 저장
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(noisy)
        axes[0].set_title(f"Noisy ({psnr_noisy:.2f} dB)")
        axes[0].axis('off')
        
        axes[1].imshow(output)
        axes[1].set_title(f"Denoised ({psnr_output:.2f} dB)")
        axes[1].axis('off')
        
        axes[2].imshow(clean)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{idx+1}_{img_file}'))
        plt.close()
    
    # ========== 결과 요약 ==========
    avg_psnr_noisy = total_psnr_noisy / len(img_files)
    avg_psnr_output = total_psnr_output / len(img_files)
    
    print("=" * 50)
    print(f"평균 Noisy PSNR: {avg_psnr_noisy:.2f} dB")
    print(f"평균 Output PSNR: {avg_psnr_output:.2f} dB")
    print(f"평균 개선: {avg_psnr_output - avg_psnr_noisy:.2f} dB")
    print("=" * 50)
    print(f"\n✓ 결과 저장: {save_dir}/")


if __name__ == "__main__":
    test()
