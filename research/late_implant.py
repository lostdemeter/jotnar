import sys; sys.path.insert(0,'/home/balrog/Documents/OpenCode/holographic_enhancement')
import numpy as np, torch
from chain.qwen7b import load7b
exec(open('/tmp/latedir.py').read().split("Hf,_ = H_of")[0])
Hf,_ = H_of('The capital of France is')
Hg,ids = H_of('The capital of Germany is')
d = Hf[4]-Hg[4]; d/=np.linalg.norm(d)
dt=torch.float32
def gen_with_steer(prompt, alpha, at_layer=26):
    ids = tok(prompt, return_tensors='pt')['input_ids'][0].numpy(); n=len(ids)
    E = torch.tensor(g('model.embed_tokens.weight')[ids], dtype=dt, device='cuda')
    eps=1e-6
    def rms(x,w): return x/torch.sqrt((x**2).mean(-1,keepdim=True)+eps)*w
    def rope(x, base=1000000.0):
        Sq,Hh,D=x.shape; i=torch.arange(D//2,dtype=dt,device='cuda'); th=base**(-2.0*i/D)
        ang=torch.arange(Sq,dtype=dt,device='cuda')[:,None]*th[None,:]; c,s=torch.cos(ang),torch.sin(ang)
        c,s=c[:,None,:],s[:,None,:]; y=torch.empty_like(x)
        y[...,0::2]=x[...,0::2]*c-x[...,1::2]*s; y[...,1::2]=x[...,0::2]*s+x[...,1::2]*c; return y
    x=E
    for L in range(28):
        p=f'model.layers.{L}.'
        w={k:torch.tensor(g(p+k),dtype=dt,device='cuda') for k in ('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.up_proj.weight','mlp.gate_proj.weight','mlp.down_proj.weight','input_layernorm.weight','post_attention_layernorm.weight')}
        bl=f'model.layers.{L}.self_attn.'
        bq=torch.tensor(g(bl+'q_proj.bias'),dtype=dt,device='cuda'); bk=torch.tensor(g(bl+'k_proj.bias'),dtype=dt,device='cuda'); bv=torch.tensor(g(bl+'v_proj.bias'),dtype=dt,device='cuda')
        if L==at_layer:
            mag=float(x[n-1].norm())
            x = x + torch.tensor(d*alpha*mag, dtype=dt, device='cuda')
        xn=rms(x,w['input_layernorm.weight'])
        Q=(xn@w['self_attn.q_proj.weight'].T+bq).reshape(n,28,128); K=(xn@w['self_attn.k_proj.weight'].T+bk).reshape(n,4,128); V=(xn@w['self_attn.v_proj.weight'].T+bv).reshape(n,4,128)
        K=K.repeat_interleave(7,dim=1); V=V.repeat_interleave(7,dim=1)
        _QR,_KR,_V=(t.permute(1,0,2) for t in (rope(Q),rope(K),V))
        P=torch.softmax(_QR@_KR.transpose(-1,-2)/np.sqrt(128)+torch.triu(torch.full((n,n),float('-inf'),device='cuda'),1),dim=-1)
        x=x+(P@_V).permute(1,0,2).reshape(n,3584)@w['self_attn.o_proj.weight'].T
        hn=rms(x,w['post_attention_layernorm.weight'])
        x=x+(torch.nn.functional.silu(hn@w['mlp.gate_proj.weight'].T)*(hn@w['mlp.up_proj.weight'].T))@w['mlp.down_proj.weight'].T
    lnf=torch.tensor(g('model.norm.weight'),dtype=dt,device='cuda')
    hn=rms(x,lnf)
    lg=(hn@torch.tensor(g('lm_head.weight'),dtype=dt,device='cuda').T)[n-1].detach().cpu().numpy()
    return lg
base = gen_with_steer('The capital of Germany is', 0.0)
for a in (0.5, 1.0, 2.0):
    lg = gen_with_steer('The capital of Germany is', a)
    top = lg.argmax()
    print(f'alpha={a}: top={tok.decode([top])!r} (base top={tok.decode([base.argmax()])!r}) Paris-rank={int((lg>lg[12095]).sum())+1} base-Paris-rank={int((base>base[12095]).sum())+1}')
