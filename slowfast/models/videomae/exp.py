from modeling_pretrain import pretrain_videomae_base_patch16_224
from masking_generator import TubeMaskingGenerator
import torch

def get_model():
    kwargs = {
        'drop_path_rate': 0.0,
        'decoder_depth': 4
    }
    model = pretrain_videomae_base_patch16_224(
        pretrained=False,
        pretrained_cfg={},
        **kwargs
    )
    return model

model = get_model()
patch_size = model.encoder.patch_embed.patch_size
window_size = (8, 14, 14)
model = model.cuda()
model.eval()
state_dict = torch.load('../../FROSTER/vit_b_hybrid_pt_800e.pth')['model']
model.load_state_dict(state_dict)

tubemask = TubeMaskingGenerator(window_size, mask_ratio=0.9)

# normalize input image
x = torch.randn(4, 3, 16, 224, 224).cuda()
mask = tubemask()
mask = torch.from_numpy(mask).unsqueeze(0).flatten(1).expand(x.size(0), -1).bool().cuda()

output = model.encoder(x, mask)
print(output.size())