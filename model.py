import common
import torch.nn as nn


class DenoisingNet(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(DenoisingNet, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        res_scale = 0.1

        # Head: 4채널 → 64채널
        m_head = [conv(4, n_feats, kernel_size)]

        # Body: 16개 ResBlock
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]

        # Tail: 64채널 → 3채널 (noise 예측)
        m_tail = [conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x: [B, 4, H, W] (noisy RGB + noise_map)
        x_rgb = x[:, :3, :, :]  # noisy RGB만 추출
        
        # Feature extraction
        feat = self.head(x)
        feat = self.body(feat)
        
        # Noise 예측
        noise = self.tail(feat)
        
        # Denoising: y = x - noise
        denoised = x_rgb - noise
        
        return denoised


if __name__ == "__main__":
    import torch
    
    # 테스트
    model = DenoisingNet()
    x = torch.randn(1, 4, 64, 64)  # [B, 4, 64, 64]
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # [1, 3, 64, 64]
    
    # 파라미터 개수
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
