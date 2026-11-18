# -*- coding: utf-8 -*-
"""Reference-style, code-faithful AG-Mamba architecture figure."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(OUT, exist_ok=True)

INK, BLUE, GREEN, PURPLE, ORANGE, RED, GRAY = '#202020', '#2166ad', '#4d9561', '#8054a6', '#d97816', '#b5443b', '#777777'
BLUE_BG, GREEN_BG, PURPLE_BG, ORANGE_BG, RED_BG = '#eaf4fc', '#edf8ef', '#f5effa', '#fff4e5', '#fff4f2'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 7.2,
                     'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(14, 8.5), facecolor='white')
ax.set_xlim(0, 140); ax.set_ylim(0, 85); ax.axis('off')

def box(x, y, w, h, text='', fc='white', ec=INK, fs=7, lw=.8, r=.8, weight='normal'):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=.16,rounding_size={r}',
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    if text:
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
                color=INK, fontweight=weight, linespacing=1.15)

def panel(x, y, w, h, title, ec, fc='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.15,rounding_size=1',
                                facecolor=fc, edgecolor=ec, linewidth=1.0))
    title_size = 7.7 if w < 42 and len(title) > 35 else 9.2
    ax.text(x+1.2, y+h-1.7, title, ha='left', va='top', fontsize=title_size,
            fontweight='bold', color=INK)

def arrow(x1, y1, x2, y2, color=INK, lw=.9):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='-|>', mutation_scale=9,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))

def waveform(x, y, w, h, color=BLUE, seed=2):
    rng=np.random.default_rng(seed); t=np.linspace(0,1,220)
    s=.35*np.sin(2*np.pi*8*t)+.18*np.sin(2*np.pi*19*t)+.12*rng.normal(size=t.size)
    ax.plot(x+w*t, y+h*(.5+.34*s/np.max(np.abs(s))), color=color, lw=.65)

# ---------------- top workflow ----------------
Y, H = 59, 22
widths = [15, 37, 31, 18, 18]
xs = [1, 17, 55, 87, 106]
for x,w,t,e,fc in zip(xs,widths,
    ['Input','Dilation-robust aliased Gabor filter bank\n(Front-end)','Selective state-space backbone\n(SSM)','Readout','Classification'],
    [INK,BLUE,GREEN,PURPLE,ORANGE], ['white',BLUE_BG,GREEN_BG,PURPLE_BG,ORANGE_BG]):
    panel(x,Y,w,H,t,e,fc)

waveform(3, 70, 11, 7); ax.text(8.5, 67.2, r'$x\in\mathbb{R}^{1\times1024}$', ha='center', fontsize=7.5)
ax.text(8.5,61.2,'vibration signal',ha='center',fontsize=6.6)
arrow(16.0,70,16.8,70)

# frontend top: stacked filter maps and actual blocks
for i in range(5):
    ax.imshow(np.outer(np.linspace(0,1,22), np.linspace(0,1,18)), extent=(20+i*.75,27+i*.75,68+i*.25,75+i*.25),
              cmap='turbo', alpha=.9, aspect='auto', zorder=1)
box(29,67.2,8,5,'Conv/Pool\n1→64→128→256→512',BLUE_BG,BLUE,6.3)
arrow(27.8,70,28.7,70,BLUE)
ax.text(35,62.2,r'$F\in\mathbb{R}^{512\times128}$',ha='center',fontsize=7)
ax.text(35,64.2,'Conv1 k=128,s=2 → Pool2 → Conv2 k=9 → Conv3 k=7 → Pool2 → Conv4 k=5',ha='center',fontsize=5.3,color=BLUE)
arrow(54,70,54.8,70)

# SSM top
box(58,67.2,8.5,5,'Linear +\nposition',GREEN_BG,GREEN,6.4)
for i,x in enumerate([68,77]):
    box(x,67.2,7.3,5,'SSM\nblock',GREEN_BG,GREEN,6.6)
    if i==0: arrow(66.6,69.7,67.8,69.7,GREEN)
arrow(85.3,70,86.7,70)
ax.text(70.5,62.3,r'$U,H\in\mathbb{R}^{128\times128}$',ha='center',fontsize=6.8)
ax.text(70.5,64.2,'3 identical blocks · d=128 · N=8 · causal conv k=4',ha='center',fontsize=5.9,color=GREEN)

# readout
box(89,70.8,8.8,4.5,'mean pooling',PURPLE_BG,PURPLE,6.5); box(89,64.5,8.8,4.5,'max pooling',PURPLE_BG,PURPLE,6.5)
box(99,66.3,4.8,6,'concat\n$4d=512$',PURPLE_BG,PURPLE,6.5)
arrow(97.8,72,98.8,70.2,PURPLE); arrow(97.8,66.7,98.8,68.3,PURPLE); arrow(103.8,69.3,105,69.3,PURPLE)
ax.text(95,61.7,'direct projection and SSM output are pooled separately',ha='center',fontsize=5.9,color=PURPLE)
# classification
box(108,67,8,5.2,'512 → 256 → C\nReLU · dropout 0.3',ORANGE_BG,ORANGE,6.5,weight='bold'); arrow(106,69.5,107.8,69.5,ORANGE)
for i,c in enumerate(['#f1ca54','#cce6a3','#b8d5ef','#e58ed0']):
    ax.scatter(120+i%2*2,71-i//2*2.4,s=36,color=c,edgecolor=INK,lw=.45)
ax.text(124,71,'Fault 1',fontsize=6); ax.text(124,68.6,'Fault 2',fontsize=6); ax.text(124,66.2,'… Fault C',fontsize=6)

# ---------------- lower panels, matching uploaded reference ----------------
panel(1,31,79,25,'(a) Dilation-robust aliased Gabor filter bank',BLUE,BLUE_BG)
for i,(x,title) in enumerate(zip([3,22,41,60],['1. Gabor atom','2. Controlled phase modulation','3. Bessel sidebands (aliasing)','4. Wideband coverage'])):
    box(x,34,17,18,title,BLUE_BG,'#9ac5eb',6.7, .65)
    if i==0:
        xx=np.linspace(-1,1,35); yy=np.linspace(-1,1,35); X,Yy=np.meshgrid(xx,yy); Z=np.exp(-4*(X**2+Yy**2))*np.cos(8*X)
        ax.contourf(x+2+15*(X+1)/2,37+10*(Yy+1)/2,Z,levels=12,cmap='turbo',alpha=.9)
        ax.text(x+8.5,35.4,r'$g_k(t)=\exp(-t^2/2\sigma^2)\cos(2\pi f_kt)$',ha='center',fontsize=5.2)
    elif i==1:
        tt=np.linspace(0,1,100); ax.plot(x+2+13*tt,40+5*np.sin(8*np.pi*tt),color=BLUE,lw=.8); ax.plot(x+2+13*tt,40+5*np.sin(8*np.pi*tt+2*tt),color=ORANGE,lw=.8,ls='--')
        ax.text(x+8.5,35.4,r'$g_k(t)\,e^{j\phi_k(t)}$',ha='center',fontsize=6)
    elif i==2:
        xx=np.linspace(-1,1,100); ax.plot(x+2+13*(xx+1)/2,39+7*np.exp(-8*xx**2),color=BLUE); ax.plot(x+2+13*(xx+1)/2,39+5*np.exp(-25*(xx-.32)**2),color=ORANGE,ls='--')
        ax.text(x+8.5,35.4,r'$f_c\pm f_k$',ha='center',fontsize=6)
    else:
        arr=np.outer(np.exp(-np.linspace(0,2,18)),np.ones(28)); ax.imshow(arr,extent=(x+2,x+15,36,48),cmap='turbo',aspect='auto'); ax.text(x+8.5,35.4,'broadband response',ha='center',fontsize=5.7)

panel(82,31,57,25,'(b) Selective state-space block',GREEN,GREEN_BG)
box(85,43,14,7,'continuous-time\nselective SSM',GREEN_BG,'#9bd1a9',6.4)
box(101,43,13,7,'discretization\n' + r'$\bar A=e^{\Delta A}$',GREEN_BG,'#9bd1a9',6.4)
box(116,43,13,7,'input-dependent\nstep size $\Delta$',PURPLE_BG,PURPLE,6.4)
arrow(99.2,46.5,100.8,46.5,GREEN); arrow(114.2,46.5,115.8,46.5,GREEN)
box(91,34.5,21,5.5,'selection mechanism\n$\Delta,B,C=f(u)$',GREEN_BG,'#9bd1a9',6.2)
box(114,34.5,18,5.5,'residual +\nbranch LayerNorm',ORANGE_BG,ORANGE,6.5)
arrow(122.5,43,122.5,40.2,GREEN); arrow(112.2,37.2,113.8,37.2,GREEN)
ax.text(84,46.5,r'$X\in\mathbb{R}^{128\times128}$',ha='right',fontsize=6); arrow(84.3,46.5,85,46.5)
ax.text(133,46.5,r'$Z\in\mathbb{R}^{128\times128}$',ha='left',fontsize=6); arrow(131.8,46.5,132.7,46.5)

panel(1,3,63,25,'(c) Training · source domains only',PURPLE,PURPLE_BG)
box(3,16,13,8,'labeled source\n$x_s,y_s$',PURPLE_BG,'#bca1d5',6.4); box(3,6,13,8,'target data\nnot used',RED_BG,RED,6.4)
box(20,10,17,9,'AG-Mamba\nfeature extractor',BLUE_BG,BLUE,6.8,weight='bold')
box(41,16,10,6,'classifier',PURPLE_BG,PURPLE,6.4); box(41,7,10,6,'regularization',PURPLE_BG,PURPLE,6.2)
box(54,9,8,10,r'$\mathcal{L}=\mathcal{L}_{CE}+0.01\overline{\Delta^2}$',ORANGE_BG,ORANGE,6.1,weight='bold')
arrow(16.2,20,19.8,16.5,PURPLE); arrow(37.2,16.5,40.8,19,PURPLE); arrow(37.2,13,40.8,10,PURPLE); arrow(51.2,19,53.8,16,PURPLE); arrow(51.2,10,53.8,12,PURPLE)
ax.text(31,5.1,'source-only supervised optimization; no MMD, adversarial discriminator, GRL, or $\tau_0$',ha='center',fontsize=6,color=RED)

panel(66,3,73,25,'(d) Inference',ORANGE,ORANGE_BG)
box(68,13,13,8,'input\n$x\in\mathbb{R}^{1\times1024}$',ORANGE_BG,ORANGE,6.5); box(86,13,15,8,'AG-Mamba\nencoder',BLUE_BG,BLUE,6.8,weight='bold'); box(106,13,13,8,'classifier',ORANGE_BG,ORANGE,6.8)
arrow(81.2,17,85.8,17,ORANGE); arrow(101.2,17,105.8,17,ORANGE)
for i,c in enumerate(['#f1ca54','#cce6a3','#b8d5ef','#e58ed0']): ax.scatter(126,20-i*2.1,s=35,color=c,edgecolor=INK,lw=.45)
ax.text(129,20,'Fault 1',fontsize=6); ax.text(129,17.9,'Fault 2',fontsize=6); ax.text(129,15.8,'… Fault C',fontsize=6); ax.text(126,10.5,r'$\hat y$',ha='center',fontsize=7)
# small implementation note
ax.text(70,1.0,'Diagram follows the delivered implementation; red modules indicate rejected/non-instantiated alternatives.',fontsize=5.7,color=RED)

fig.savefig(os.path.join(OUT,'fig3_arch.png'),dpi=220,bbox_inches='tight')
fig.savefig(os.path.join(OUT,'fig3_arch.pdf'),bbox_inches='tight')
print('saved', os.path.join(OUT,'fig3_arch.png'))
