import sys; sys.path.insert(0,'/home/balrog/Documents/OpenCode/holographic_enhancement')
import numpy as np, torch
from chain.qwen7b import load7b
g, tok = load7b()
dt=torch.float32
def H_of(prompt, layer=26):
    ids = tok(prompt, return_tensors='pt')['input_ids'][0].numpy()
    n = len(ids)
    E = torch.tensor(g('model.embed_tokens.weight')[ids], dtype=dt, device='cuda')
    eps=1e-6
    def rms(x,w): return x/torch.sqrt((x**2).mean(-1,keepdim=True)+eps)*w
    def rope(x, base=1000000.0):
        Sq,Hh,D=x.shape; i=torch.arange(D//2,dtype=dt,device='cuda'); th=base**(-2.0*i/D)
        ang=torch.arange(Sq,dtype=dt,device='cuda')[:,None]*th[None,:]; c,s=torch.cos(ang),torch.sin(ang)
        c,s=c[:,None,:],s[:,None,:]; y=torch.empty_like(x)
        y[...,0::2]=x[...,0::2]*c-x[...,1::2]*s; y[...,1::2]=x[...,0::2]*s+x[...,1::2]*c; return y
    x=E
    for L in range(layer+1):
        p=f'model.layers.{L}.'
        w={k:torch.tensor(g(p+k),dtype=dt,device='cuda') for k in ('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.up_proj.weight','mlp.gate_proj.weight','mlp.down_proj.weight','input_layernorm.weight','post_attention_layernorm.weight')}
        bl=f'model.layers.{L}.self_attn.'
        bq=torch.tensor(g(bl+'q_proj.bias'),dtype=dt,device='cuda'); bk=torch.tensor(g(bl+'k_proj.bias'),dtype=dt,device='cuda'); bv=torch.tensor(g(bl+'v_proj.bias'),dtype=dt,device='cuda')
        xn=rms(x,w['input_layernorm.weight'])
        Q=(xn@w['self_attn.q_proj.weight'].T+bq).reshape(n,28,128); K=(xn@w['self_attn.k_proj.weight'].T+bk).reshape(n,4,128); V=(xn@w['self_attn.v_proj.weight'].T+bv).reshape(n,4,128)
        K=K.repeat_interleave(7,dim=1); V=V.repeat_interleave(7,dim=1)
        _QR,_KR,_V=(t.permute(1,0,2) for t in (rope(Q),rope(K),V))
        P=torch.softmax(_QR@_KR.transpose(-1,-2)/np.sqrt(128)+torch.triu(torch.full((n,n),float('-inf'),device='cuda'),1),dim=-1)
        x=x+(P@_V).permute(1,0,2).reshape(n,3584)@w['self_attn.o_proj.weight'].T
        hn=rms(x,w['post_attention_layernorm.weight'])
        x=x+(torch.nn.functional.silu(hn@w['mlp.gate_proj.weight'].T)*(hn@w['mlp.up_proj.weight'].T))@w['mlp.down_proj.weight'].T
    return x.detach().cpu().numpy(), ids
Hf,_ = H_of('The capital of France is')
Hg,_ = H_of('The capital of Germany is')
d = Hf[4]-Hg[4]; d/=np.linalg.norm(d)
W = g('lm_head.weight').astype(np.float64)
cos = (W@d)/np.linalg.norm(W,axis=1)
order = np.argsort(-cos)
print('top aligned tokens:', [tok.decode([int(i)]) for i in order[:8]])
print('cos top:', np.round(cos[order[:8]],3))
paris = tok(' Paris', return_tensors='pt')['input_ids'][0].tolist()
print('Paris ids:', paris, 'ranks:', [int((cos>cos[p]).sum())+1 for p in paris])
