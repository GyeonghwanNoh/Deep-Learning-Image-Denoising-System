import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DenoisingNet


class DnCNN_Pretrained(nn.Module):
    """KAIR Pretrained 모델용 (단순 Sequential 구조)"""
    def __init__(self, in_nc=4, out_nc=3, nc=64, nb=17):
        super(DnCNN_Pretrained, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(nb - 2):
            layers.append(nn.Conv2d(nc, nc, 3, 1, 1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def calculate_psnr(img1, img2):
    """PSNR 계산"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def test_model_on_images(model, img_files, test_img_path, noise_level, device, use_noise_map=True, residual=False):
    """모델로 이미지들 테스트하고 PSNR 리스트 반환"""
    psnr_list = []
    
    for img_file in img_files:
        img_path = os.path.join(test_img_path, img_file)
        
        # 이미지 로드
        img = Image.open(img_path).convert('RGB')
        clean = np.array(img).astype(np.float32) / 255.0
        
        # 노이즈 추가
        noise = np.random.randn(*clean.shape) * (noise_level / 255.0)
        noisy = clean + noise
        
        # 텐서 변환
        noisy_t = torch.from_numpy(noisy).permute(2, 0, 1).float()
        
        if use_noise_map:
            # Noise map 추가 (Your model)
            noise_map = torch.ones(1, *noisy_t.shape[1:]) * (noise_level / 255.0)
            input_t = torch.cat([noisy_t, noise_map], dim=0).unsqueeze(0).to(device)
        else:
            # Noise map 없이 (Pretrained model)
            input_t = noisy_t.unsqueeze(0).to(device)
        
        # 추론
        with torch.no_grad():
            model_output = model(input_t)[0].cpu().permute(1, 2, 0).numpy()
        
        # Residual learning: output이 noise이면 빼기
        if residual:
            output = noisy - model_output  # clean = noisy - noise
        else:
            output = model_output  # 직접 clean 출력
        
        # PSNR 계산
        psnr = calculate_psnr(output, clean)
        psnr_list.append(psnr)
    
    return psnr_list


def compare_models():
    # ========== 설정 ==========
    model1_path = './checkpoints/dncnn_color_blind.pth'  # Pretrained
    model2_path = './checkpoints/model_best.pth'         # Your model
    test_img_path = './DIV2K_valid_HR'
    noise_level = 25
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # ========== 이미지 파일 로드 ==========
    all_files = sorted([f for f in os.listdir(test_img_path) if f.endswith('.png')])
    
    if len(all_files) == 0:
        print(f"❌ 테스트 이미지가 없습니다: {test_img_path}")
        return
    
    print(f"테스트 이미지: {len(all_files)}개\n")
    
    # ========== 모델 1 (Pretrained) ==========
    print("[1] Pretrained Model (dncnn_color_blind.pth)")
    if not os.path.exists(model1_path):
        print(f"   ✗ 파일이 없습니다: {model1_path}\n")
        psnr1_list = None
    else:
        model1 = DnCNN_Pretrained(in_nc=3, out_nc=3, nc=64, nb=20).to(device)
        model1.load_state_dict(torch.load(model1_path, map_location=device))
        model1.eval()
        print("   모델 로드 완료")
        print("   테스트 중...")
        psnr1_list = test_model_on_images(model1, all_files, test_img_path, noise_level, device, use_noise_map=False, residual=True)
        avg1 = np.mean(psnr1_list)
        print(f"   평균 PSNR: {avg1:.2f} dB\n")
    
    # ========== 모델 2 (Your model) ==========
    print("[2] Your Trained Model (model_best.pth)")
    if not os.path.exists(model2_path):
        print(f"   ✗ 파일이 없습니다: {model2_path}\n")
        psnr2_list = None
    else:
        model2 = DenoisingNet().to(device)
        model2.load_state_dict(torch.load(model2_path, map_location=device))
        model2.eval()
        print("   모델 로드 완료")
        print("   테스트 중...")
        psnr2_list = test_model_on_images(model2, all_files, test_img_path, noise_level, device)
        avg2 = np.mean(psnr2_list)
        print(f"   평균 PSNR: {avg2:.2f} dB\n")
    
    # ========== 비교 그래프 ==========
    if psnr1_list and psnr2_list:
        print("그래프 생성 중...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 막대 그래프 (평균 비교)
        axes[0].bar(['Pretrained', 'Your Model'], [avg1, avg2], 
                   color=['#3498db', '#e74c3c'], width=0.6)
        axes[0].set_ylabel('Average PSNR (dB)', fontsize=12)
        axes[0].set_title('Model Performance Comparison', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        axes[0].text(0, avg1 + 0.3, f'{avg1:.2f} dB', ha='center', fontsize=11)
        axes[0].text(1, avg2 + 0.3, f'{avg2:.2f} dB', ha='center', fontsize=11)
        
        # 2. 선 그래프 (이미지별 비교)
        x = range(1, len(psnr1_list) + 1)
        axes[1].plot(x, psnr1_list, 'b-', linewidth=1.5, label='Pretrained', alpha=0.7)
        axes[1].plot(x, psnr2_list, 'r-', linewidth=1.5, label='Your Model', alpha=0.7)
        axes[1].set_xlabel('Image Index', fontsize=12)
        axes[1].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1].set_title('Per-Image PSNR Comparison', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./model_comparison.png', dpi=150)
        plt.close()
        
        print(f"✓ 그래프 저장: ./model_comparison.png\n")
        
        # ========== 결과 요약 ==========
        print("=" * 60)
        print("비교 결과")
        print("=" * 60)
        print(f"Pretrained:  {avg1:.2f} dB")
        print(f"Your Model:  {avg2:.2f} dB")
        print(f"차이:        {avg2 - avg1:+.2f} dB")
        
        if avg2 > avg1:
            print(f"\n✓ Your model이 {avg2 - avg1:.2f} dB 더 좋습니다!")
        elif avg2 < avg1:
            print(f"\n✗ Pretrained가 {avg1 - avg2:.2f} dB 더 좋습니다.")
        else:
            print("\n= 두 모델 성능이 동일합니다.")
        print("=" * 60)
        print("=" * 60)
    
    # # ========== 비교 그래프 (Pretrained 비교용) ==========
    # if psnr1_list and psnr2_list:
    #     print("그래프 생성 중...")
    #     
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    #     
    #     # 1. 막대 그래프 (평균 비교)
    #     axes[0].bar(['Pretrained', 'Your Model'], [avg1, avg2], 
    #                color=['#3498db', '#e74c3c'], width=0.6)
    #     axes[0].set_ylabel('Average PSNR (dB)', fontsize=12)
    #     axes[0].set_title('Model Performance Comparison', fontsize=14)
    #     axes[0].grid(True, alpha=0.3, axis='y')
    #     
    #     # 값 표시
    #     axes[0].text(0, avg1 + 0.3, f'{avg1:.2f} dB', ha='center', fontsize=11)
    #     axes[0].text(1, avg2 + 0.3, f'{avg2:.2f} dB', ha='center', fontsize=11)
    #     
    #     # 2. 선 그래프 (이미지별 비교)
    #     x = range(1, len(psnr1_list) + 1)
    #     axes[1].plot(x, psnr1_list, 'b-', linewidth=1.5, label='Pretrained', alpha=0.7)
    #     axes[1].plot(x, psnr2_list, 'r-', linewidth=1.5, label='Your Model', alpha=0.7)
    #     axes[1].set_xlabel('Image Index', fontsize=12)
    #     axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    #     axes[1].set_title('Per-Image PSNR Comparison', fontsize=14)
    #     axes[1].legend(fontsize=10)
    #     axes[1].grid(True, alpha=0.3)
    #     
    #     plt.tight_layout()
    #     plt.savefig('./model_comparison.png', dpi=150)
    #     plt.close()
    #     
    #     print(f"✓ 그래프 저장: ./model_comparison.png\n")
    #     
    #     # ========== 결과 요약 ==========
    #     print("=" * 60)
    #     print("비교 결과")
    #     print("=" * 60)
    #     print(f"Pretrained:  {avg1:.2f} dB")
    #     print(f"Your Model:  {avg2:.2f} dB")
    #     print(f"차이:        {avg2 - avg1:+.2f} dB")
    #     
    #     if avg2 > avg1:
    #         print(f"\n✓ Your model이 {avg2 - avg1:.2f} dB 더 좋습니다!")
    #     elif avg2 < avg1:
    #         print(f"\n✗ Pretrained가 {avg1 - avg2:.2f} dB 더 좋습니다.")
    #     else:
    #         print("\n= 두 모델 성능이 동일합니다.")
    #     print("=" * 60)


if __name__ == "__main__":
    compare_models()
