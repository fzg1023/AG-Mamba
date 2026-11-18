
import os, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, '/home/fzg/Fault/AG-Mamba')
import torch, torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import models.agmamba_backbone as B

DEV = 'cuda'
torch.manual_seed(0)
m = B.LiquidMambaDGNet(num_classes=14, front_mode='gaborwide', front_init='gabor',
                       front_chan=512, kernel_size=128, gabor_f_lo=0.0025,
                       gabor_dc_frac=0.3, gabor_alias=2.0, use_ssm=True,
                       tau_mode='zero', n_blocks=3, d_model=128, d_inner=128,
                       n_state=8).to(DEV)
m.eval()   # 关 dropout / 用 running stats，隔离扫描后端本身的数值差异
x = torch.randn(16, 1, 1024, device=DEV)
y = torch.randint(0, 14, (16,), device=DEV)

def step():
    lg, _, dn, _ = m(x)
    loss = F.cross_entropy(lg, y) + dn
    m.zero_grad(); loss.backward()
    return (float(loss),
            {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None},
            lg.detach().clone())

l1, g1, o1 = step(); l2, g2, o2 = step()
print('OFFICIAL self         max|dlogits|=%.3e  max|dgrad|=%.3e'
      % (float((o1-o2).abs().max()), max(float((g1[k]-g2[k]).abs().max()) for k in g1)))
B._OFFICIAL_SCAN = None
l3, g3, o3 = step(); l4, g4, o4 = step()
print('PYSCAN   self         max|dlogits|=%.3e  max|dgrad|=%.3e'
      % (float((o3-o4).abs().max()), max(float((g3[k]-g4[k]).abs().max()) for k in g3)))
print('PY vs OFFICIAL        max|dlogits|=%.3e  max|dgrad|=%.3e  loss diff=%.3e'
      % (float((o3-o1).abs().max()), max(float((g3[k]-g1[k]).abs().max()) for k in g1), abs(l3-l1)))
