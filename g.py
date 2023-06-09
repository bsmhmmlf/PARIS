import torch
from tqdm import tqdm
from torch import nn
from torchvision import utils
from sagan_models_test import Generator


g_ema = Generator(1,64, 256, 64).cuda()
checkpoint = torch.load('./1000298_G.pth')
g_ema.load_state_dict(checkpoint)




#optim = torch.optim.AdamW(extractor.parameters(), lr=0.0001)

iter_num = 10000


pbar = tqdm(range(iter_num))
g_ema.eval()
for i in pbar:
    with torch.no_grad():
        g_ema.eval()
        #z = torch.load(f'./z/{i}.pth')
        z = torch.randn(1, 256).cuda()
        #torch.save(z,f"/data-x/g13/yangzijin/Self-Attention-GAN-master/data/z/{str(i)}.pth")
        # torch.save(z,'z.pth')
        sample, _, _ = g_ema(z)
        utils.save_image(
            sample,
            f"./img/{str(i)}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

