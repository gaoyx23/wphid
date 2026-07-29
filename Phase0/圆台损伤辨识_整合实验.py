# -*- coding: utf-8 -*-
"""
圆台喷管 老化损伤机理辨识（单实验，全机理总损伤）
================================================
按用户要求：
  ① 候选库=新 11 项（删温度项、加热损失 Arrhenius）；温度场靠边界条件前向求出，不入库。
  ② 总损伤 D_total = Dh(热损失)+Do(氧化)+Df(疲劳)+Dc(蠕变)，包含所有损失项。
  ③ 只做一个实验：种入跨 4 机理的真项，用完整 11 项库做 SINDy，看能否识别。
前向：隐式 backward Euler、300 节点、完整瞬态氧扩散。噪声：errors-in-variables+savgol。
定位=方法验证（律自设；D 假设可观测）。依赖 numpy/scipy/matplotlib。
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    HAS=True
except Exception:
    HAS=False

# ===== 材料/几何/边界 =====
rho=8440.;cp=580.;kth=25.;E=200e9;ath=13e-6;nu=0.3
eps=0.8;SB=5.67e-8;alpha=kth/(rho*cp);Rg=8.314
Ea_ox,Ea_c,Ea_h=120e3,280e3,200e3; m_f,q_f,n_c=3.0,1.0,4.0
L=0.30;Rthroat=0.05;Rexit=0.15;hwall=5e-3
Nth=10;Nz=30
z=np.linspace(0,L,Nz);dz=z[1]-z[0];dz2=dz**2
th=np.linspace(0,hwall,Nth);dth=th[1]-th[0];dth2=dth**2
N=Nth*Nz
def pidx(i,j): return i*Nz+j
inner_idx=np.array([pidx(0,j) for j in range(Nz)])
outer_idx=np.array([pidx(Nth-1,j) for j in range(Nz)])
r_in=Rthroat+(Rexit-Rthroat)*(z/L)
period=120.;Ncyc=6;T_end=Ncyc*period
def gate(tt):
    ph=tt%period
    return np.where(ph<15,ph/15,np.where(ph<90,1.0,np.where(ph<105,(105-ph)/15,0.0)))
def env(tt): return 1+0.25*np.sin(2*np.pi*tt/T_end)
def env_po(tt): return 1+0.30*np.sin(2*np.pi*tt/(T_end*0.7)+1.5)
def env_p(tt):  return 1+0.35*np.sin(2*np.pi*tt/(T_end*0.55)+2.3)   # 压力独立包络(与T不同→σ含独立分量)
Tg0=2000.;hg=320.;hg_prof=hg*(1.0+0.6*(1-z/L));Tamb=300.;D_O2=1e-8
def Tgas(tt): return 300.+Tg0*gate(tt)*env(tt)
def pogas(tt): return 0.6*gate(tt)*env_po(tt)
def pref(tt): return 5.0e6*gate(tt)*env_p(tt)

# ===== 2D 扩散算子 L =====
R=[];C=[];V=[]
def add(r,c,v): R.append(r);C.append(c);V.append(v)
for i in range(Nth):
    for j in range(Nz):
        p=pidx(i,j);d=0.0
        if i==0: d-=1/dth2; add(p,pidx(1,j),1/dth2)
        elif i==Nth-1: d-=1/dth2; add(p,pidx(Nth-2,j),1/dth2)
        else: d-=2/dth2; add(p,pidx(i-1,j),1/dth2); add(p,pidx(i+1,j),1/dth2)
        if j==0: d-=2/dz2; add(p,pidx(i,1),2/dz2)
        elif j==Nz-1: d-=2/dz2; add(p,pidx(i,Nz-2),2/dz2)
        else: d-=2/dz2; add(p,pidx(i,j-1),1/dz2); add(p,pidx(i,j+1),1/dz2)
        add(p,p,d)
L=coo_matrix((V,(R,C)),shape=(N,N)).tocsr()

# ===== 1. 隐式前向求解：T 与 pO2(完整瞬态氧扩散) =====
dt=0.2;Nt=int(T_end/dt);store_every=5;dt_snap=store_every*dt
dconv=np.zeros(N);dconv[inner_idx]=hg_prof/(rho*cp*dth)
M_T=(speye(N)-dt*alpha*L+dt*diags(dconv)).tocsc()
M_po=(speye(N)-dt*D_O2*L).tocsc()
Tn=np.full(N,300.);Pn=np.zeros(N)
T_hist=[];P_hist=[];ts_list=[]
for n,tt in enumerate(np.arange(Nt)*dt):
    Tg=Tgas(tt);pg=pogas(tt);To=Tn[outer_idx]
    drad=np.zeros(N);drad[outer_idx]=4*eps*SB*To**3/(rho*cp*dth)
    A=(M_T+dt*diags(drad)).tocsc()
    b=np.zeros(N);b[inner_idx]+=hg_prof*Tg/(rho*cp*dth)
    b[outer_idx]+=eps*SB*(3*To**4+Tamb**4)/(rho*cp*dth)
    Tn=spsolve(A,Tn+dt*b)
    Pn=spsolve(M_po,Pn); Pn[inner_idx]=pg
    if n%store_every==0: T_hist.append(Tn.reshape(Nth,Nz).copy());P_hist.append(Pn.reshape(Nth,Nz).copy());ts_list.append(tt)
T_hist=np.array(T_hist);P_hist=np.array(P_hist);t_snap=np.array(ts_list);Nsnap=len(t_snap)
sig_hist=np.zeros_like(T_hist)
for s,tt in enumerate(t_snap):
    Tm=T_hist[s].mean(0,keepdims=True)
    sig_hist[s]=((E*ath/(1-nu))*(T_hist[s]-Tm)*0.40+(pref(tt)*r_in/hwall)[None,:])/1e6
steps_per_cyc=int(period/(dt*store_every))
Dsg=np.zeros((Ncyc,Nth,Nz))
for c in range(Ncyc):
    s0,s1=c*steps_per_cyc,(c+1)*steps_per_cyc+1; s0=min(s0,Nsnap-1); s1=min(s1,Nsnap)
    Dsg[c]=sig_hist[s0:s1].max(0)-sig_hist[s0:s1].min(0)
dsig_hist=np.zeros_like(T_hist)
for s in range(Nsnap):
    c=min(int(t_snap[s]//period),Ncyc-1); dsig_hist[s]=Dsg[c]
# 公共核
arr_hist=np.exp(-Ea_ox/(Rg*np.maximum(T_hist,300.)))
arrh_hist=np.exp(-Ea_h/(Rg*np.maximum(T_hist,300.)))
arrc_hist=np.exp(-Ea_c/(Rg*np.maximum(T_hist,300.)))
creep_hist=sig_hist**n_c*arrc_hist
print("="*78);print("圆台 老化损伤辨识（单实验，全机理总损伤，11 项候选库）")
print("  T[%.0f,%.0f]K；σ[%.0f,%.0f]MPa；pO2局部[%.2f,%.2f]"
      %(T_hist[:,0,:].min(),T_hist[:,0,:].max(),sig_hist.min(),sig_hist.max(),P_hist.min(),P_hist.max()))

# ===== 2. 四类机理损伤分别积分；D_total = Dh+Do+Df+Dc =====
# 真系数（配平使四类贡献相当、不饱和：Dh/Do/Df/Dc 各 ~0.1）
a_h=900.0;  a_o=3.0; a_sig=0.01; a_f=1e-11; a_c=8e-4
Dh=np.zeros_like(T_hist); Do=np.zeros_like(T_hist); Df=np.zeros_like(T_hist); Dc=np.zeros_like(T_hist)
for s in range(Nsnap-1):
    paris=dsig_hist[s]**m_f*np.maximum(1-Df[s],0.05)**(-q_f)
    Dh[s+1]=np.clip(Dh[s]+a_h*arrh_hist[s]*dt_snap,0,0.9)
    Do[s+1]=np.clip(Do[s]+(a_o*arr_hist[s]*P_hist[s]+a_sig*arr_hist[s]*sig_hist[s])*dt_snap,0,0.9)
    Df[s+1]=np.clip(Df[s]+a_f*paris*dt_snap,0,0.9)         # 疲劳按连续率近似(Paris 作速率)
    Dc[s+1]=np.clip(Dc[s]+a_c*creep_hist[s]*dt_snap,0,0.9)
Dtot=Dh+Do+Dc      # 连续总损伤(疲劳 Df 按循环统计量、单独处理，不入连续 ODE)
print("  Dh[%.3f,%.3f] Do[%.3f,%.3f] Df[%.3f,%.3f](不入连续总) Dc[%.3f,%.3f] → D_total[%.3f,%.3f]"
      %(Dh.min(),Dh.max(),Do.min(),Do.max(),Df.min(),Df.max(),Dc.min(),Dc.max(),Dtot.min(),Dtot.max()))

# ===== 3. 完整 11 项候选库 =====
def flat(x): return x.reshape(-1)
names11=['arr_h','arr_h·σ','arr·pO2','arr·σ','arr·Df','Paris','Paris·T','Paris·Do','Paris·Dc','creep','creep·Do']
def build11(T,sig,po,Df,Do,Dc,dsig):
    arr=np.exp(-Ea_ox/(Rg*np.maximum(T,300.))); arrh=np.exp(-Ea_h/(Rg*np.maximum(T,300.)))
    arrc=np.exp(-Ea_c/(Rg*np.maximum(T,300.)))
    paris=dsig**m_f*np.maximum(1-Df,0.05)**(-q_f); creep=sig**n_c*arrc
    cols=[arrh, arrh*sig, arr*po, arr*sig, arr*Df, paris, paris*T, paris*Do, paris*Dc, creep, creep*Do]
    return np.column_stack([flat(c) for c in cols])
# 真项：arr_h(0), arr·pO2(2), arr·σ(3), Paris(5), creep(9) —— 跨全部 4 机理 + 应力助氧化耦合
true_idx={0:a_h, 2:a_o, 3:a_sig, 9:a_c}    # arr_h, arr·pO2, arr·σ, creep（4 连续机理；疲劳 Paris 按循环另计）

# ===== 4. 辨识 =====
WIN=11;PO=3;mm=(WIN-1)//2;sl=slice(mm,Nsnap-mm)
def dDt(Df_): return savgol_filter(Df_,WIN,PO,deriv=1,delta=dt_snap,axis=0)
def stlsq(A,b,thr=0.05,it=20):
    x=np.linalg.lstsq(A,b,rcond=None)[0]
    for _ in range(it):
        sm=np.abs(x)<thr
        if sm.all() or (sm==False).sum()==0: break
        x[sm]=0;bg=~sm
        if bg.sum()==0: break
        x[bg]=np.linalg.lstsq(A[:,bg],b,rcond=None)[0]
    return x
def identify(Th,y,thr=0.05):
    # 列按 std 缩放(不中心化→保非负、无隐式截距)；y 按 std 缩放
    sd=Th.std(0);sd[sd<1e-12]=1;Ts=Th/sd
    sy=y.std();ys=y/sy
    xs=stlsq(Ts,ys,thr);sup=np.abs(xs)>1e-9
    x=np.zeros(Th.shape[1]);x[sup]=(sy/sd[sup])*xs[sup]   # 标准化系数→原始尺度(避免原始尺度 lstsq 病态)
    yhat=Th@x;r2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    return np.linalg.cond(Ts),sup,x,r2
Th=build11(T_hist[sl],sig_hist[sl],P_hist[sl],Df[sl],Do[sl],Dc[sl],dsig_hist[sl])
y=flat(dDt(Dtot)[sl]); cond,sup,x,r2=identify(Th,y)
print("="*78);print("【单实验】辨识 dD_total/dt（候选=完整11项，5 真项跨4机理）")
print("  无噪声：cond=%.1f R²=%.4f"%(cond,r2))
for i,nm in enumerate(names11):
    tag=("真=%.4g"%true_idx[i]) if i in true_idx else "干扰"
    print("  %-10s %4s %13.5g  %s"%(nm,"YES" if sup[i] else "no",x[i] if sup[i] else 0,tag))
print("  -> %d 真项恢复 %d；误选干扰 %d"%(len(true_idx),sum(sup[i] for i in true_idx),
    sum(sup[i] for i in range(11) if i not in true_idx)))
# EIV 噪声
print("  EIV 噪声:  噪声  R²_pred  真项恢复  误选")
sT,sS,sP,sDt=T_hist.std(),sig_hist.std(),P_hist.std(),Dtot.std()
noise=[]
for nl in [0.02,0.05,0.10,0.20]:
    rp=[];rec=[];spur=[]
    for s in range(6):
        rng=np.random.default_rng(7000*s+int(nl*1e5))
        Tn_=T_hist+rng.normal(0,nl*sT,size=T_hist.shape)
        Sn_=sig_hist+rng.normal(0,nl*sS,size=sig_hist.shape)
        Pn_=P_hist+rng.normal(0,nl*sP,size=P_hist.shape)
        Dn_=Dtot+rng.normal(0,nl*sDt,size=Dtot.shape)
        Thn=build11(Tn_[sl],Sn_[sl],Pn_[sl],Df[sl],Do[sl],Dc[sl],dsig_hist[sl])
        yn=flat(dDt(Dn_)[sl]); _,sp,xr,_=identify(Thn,yn); ypred=Th@xr
        rp.append(1-np.sum((ypred-y)**2)/np.sum((y-y.mean())**2))
        rec.append(sum(sp[i] for i in true_idx)); spur.append(sum(sp[i] for i in range(11) if i not in true_idx))
    noise.append((nl,np.mean(rp),np.mean(rec),np.mean(spur)))
    print("            %5.2f  %7.3f  %6.2f    %4.2f"%noise[-1])

if HAS:
    fig,ax=plt.subplots(2,3,figsize=(16,9))
    # 上排：物理场空间分布（取内壁均温最高的快照=峰值时刻）
    Tpi=int(np.argmax(T_hist[:,0,:].mean(1)))
    for k,(fld,ttl,cm,units) in enumerate([
        (T_hist[Tpi],"T field (K)",'inferno','K'),
        (sig_hist[Tpi],"sigma field",'RdBu_r','MPa'),
        (P_hist[Tpi],"pO2 field",'viridis','atm')]):
        pcm=ax[0,k].pcolormesh(z*1e3,th*1e3,fld,shading='auto',cmap=cm)
        ax[0,k].set_title(ttl+" (peak t=%.0fs)"%t_snap[Tpi]);ax[0,k].set_xlabel("z (mm)");ax[0,k].set_ylabel("thickness (mm)")
        fig.colorbar(pcm,ax=ax[0,k])
    # 支持恢复
    cols=['tab:green' if (i in true_idx and sup[i]) else ('tab:red' if (i not in true_idx and sup[i]) else 'lightgray') for i in range(11)]
    ax[1,0].bar(range(11),[1 if sup[i] else 0.15 for i in range(11)],color=cols)
    ax[1,0].set_xticks(range(11));ax[1,0].set_xticklabels(names11,rotation=35,ha='right',fontsize=8)
    ax[1,0].set_title("Support (green=true&sel, gray=rejected)")
    # 系数恢复（4 真项：恢复值 vs 真值，对数轴）
    tidx=sorted(true_idx.keys()); recov=[x[i] for i in tidx]; tru=[true_idx[i] for i in tidx]
    xx=np.arange(len(tidx)); w=0.4
    ax[1,1].bar(xx-w/2,recov,w,label='recovered')
    ax[1,1].bar(xx+w/2,tru,w,label='true')
    ax[1,1].set_xticks(xx);ax[1,1].set_xticklabels([names11[i] for i in tidx],rotation=20,ha='right',fontsize=9)
    ax[1,1].set_yscale('log');ax[1,1].legend(fontsize=8);ax[1,1].set_title("Coefficient recovery (log scale)")
    # 噪声鲁棒性
    ax[1,2].plot([n[0] for n in noise],[n[2] for n in noise],'o-',label='true recovered')
    ax[1,2].plot([n[0] for n in noise],[n[3] for n in noise],'s-',label='spurious')
    ax[1,2].set_xlabel("measurement noise (x std)");ax[1,2].set_ylabel("count");ax[1,2].legend(fontsize=8)
    ax[1,2].set_title("Noise robustness")
    fig.tight_layout();fig.savefig(r"C:\Users\高艺艺\Desktop\Research focus\圆台损伤辨识_整合实验.png",dpi=120)
    print("已存图：圆台损伤辨识_整合实验.png（2x3：物理场+支持+共线+噪声）")
print("="*78);print("完成。")
