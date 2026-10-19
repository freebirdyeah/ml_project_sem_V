"""Structural features of a strashed AIG, parsed from binary AIGER."""
import numpy as np
def read_aig(path):
    with open(path,'rb') as f: data=f.read()
    nl=data.index(b'\n'); hdr=data[:nl].split(); assert hdr[0]==b'aig'
    M,I,L,O,A=map(int,hdr[1:6]); pos=nl+1
    outs=[]
    for _ in range(O):
        e=data.index(b'\n',pos); outs.append(int(data[pos:e])); pos=e+1
    def varint():
        nonlocal pos
        x=0; s=0
        while True:
            b=data[pos]; pos+=1; x|=(b&0x7f)<<s
            if not b&0x80: return x
            s+=7
    f0=np.zeros(A,dtype=np.int64); f1=np.zeros(A,dtype=np.int64)
    for i in range(A):
        lhs=2*(I+L+1+i); d0=varint(); d1=varint()
        r0=lhs-d0; r1=r0-d1; f0[i]=r0; f1[i]=r1
    return I,O,A,outs,f0,f1
def features(path):
    I,O,A,outs,f0,f1=read_aig(path)
    N=I+A+1                       # var ids 0..I+A
    lvl=np.zeros(N,dtype=np.int64)
    for i in range(A):
        v=I+1+i; lvl[v]=1+max(lvl[f0[i]>>1],lvl[f1[i]>>1])
    outs=np.array(outs,dtype=np.int64)
    depth=int(lvl[outs>>1].max()) if O else int(lvl.max())
    fo=np.bincount(np.concatenate([f0>>1,f1>>1,outs>>1]),minlength=N).astype(float)
    fo_nodes=fo[1:]                # PIs + ANDs
    fo_and=fo[I+1:]
    comp=np.concatenate([f0&1,f1&1]).mean() if A else 0.0
    lv_and=lvl[I+1:]
    width=np.bincount(lv_and,minlength=depth+1)[1:].astype(float) if A else np.zeros(1)
    gap=np.abs(lvl[f0>>1]-lvl[f1>>1]) if A else np.zeros(1)
    # XOR/XNOR pattern: n = AND(~x,~y), x=AND(a,b), y=AND(~a,~b)
    fanin={I+1+i:(int(f0[i]),int(f1[i])) for i in range(A)}
    xor=0; reconv=0
    for i in range(A):
        a,b=int(f0[i]),int(f1[i]); u,v=a>>1,b>>1
        if (a&1) and (b&1) and u in fanin and v in fanin:
            xa,xb=fanin[u]; ya,yb=fanin[v]
            if {xa^1,xb^1}=={ya,yb}: xor+=1
        su={u}|({fanin[u][0]>>1,fanin[u][1]>>1} if u in fanin else set())
        sv={v}|({fanin[v][0]>>1,fanin[v][1]>>1} if v in fanin else set())
        if (su&sv)-{0}: reconv+=1
    A_=max(A,1)
    return dict(
      n_pi=I, n_po=O, n_and=A, depth=depth, n_edges=2*A+O,
      and_per_depth=A/max(depth,1), edges_per_node=(2*A+O)/max(A+I,1),
      pi_per_and=I/A_, po_per_and=O/A_,
      fo_mean=fo_nodes.mean(), fo_max=fo_nodes.max(), fo_std=fo_nodes.std(),
      fo_gt1_frac=(fo_nodes>1).mean(), fo_and_mean=fo_and.mean() if A else 0.0,
      comp_edge_frac=comp,
      width_max=width.max(), width_mean=width.mean(), width_cv=width.std()/max(width.mean(),1e-9),
      width_max_frac=width.max()/A_,
      lvl_gap_mean=gap.mean(), lvl_gap_ge2_frac=(gap>=2).mean(), lvl_gap_max=int(gap.max()),
      xor_frac=xor/A_, reconv_frac=reconv/A_)
if __name__=='__main__':
    import sys, json; print(json.dumps(features(sys.argv[1]),indent=1,default=float))
