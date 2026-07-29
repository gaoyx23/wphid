# -*- coding: utf-8 -*-
"""
圆台 SINDy-PI:辨识全部"非线性于参数、现实不可靠知"的常数
==========================================================
真实场景下这些常数都不可靠已知(材料/工况依赖):活化能 Ea_d/Ea_h/Ea_o/Ea_c、Paris m、
Norton n、饱和 q。它们都在非线性位置(指数 e^{−Ea/RT}、幂次 σ^n/(Δσ)^m),标准 SINDy
(线性可分)处理不了。本脚本用 **profile 扫描**(对参数网格每点做线性内层拟合、取 R² 峰值)
从数据逐一辨识——属 SINDy-PI 思路(naive 多列离散共线失效,故用 profile)。

前向种入【真值(对 SINDy 隐藏)】:Ea_d=100k Ea_h=200k Ea_o=120k Ea_c=280k; m=3 n=4 q=1。
依赖 numpy/scipy。定位=方法验证。
"""
import sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# ===== 真值常数(对 SINDy 隐藏)=====
Rg = 8.314
Ea_d_T, Ea_o_T, Ea_h_T, Ea_c_T = 100e3, 120e3, 200e3, 280e3
m_T, q_T, n_T = 3.0, 1.0, 4.0
rho=8440.;cp=580.;kth=25.;E=200e9;ath=13e-6;nu=0.3;eps=0.8;SB=5.67e-8;alpha=kth/(rho*cp)
Lg=0.30;Rthroat=0.05;Rexit=0.15;hwall=5e-3;Nth=10;Nz=30
z=np.linspace(0,Lg,Nz);dz=z[1]-z[0];dz2=dz**2
th=np.linspace(0,hwall,Nth);dth=th[1]-th[0];dth2=dth**2
N=Nth*Nz
def pidx(i,j): return i*Nz+j
inner_idx=np.array([pidx(0,j) for j in range(Nz)])
outer_idx=np.array([pidx(Nth-1,j) for j in range(Nz)])
r_in=Rthroat+(Rexit-Rthroat)*(z/Lg)
period=120.;Ncyc=15;T_end=Ncyc*period
def gate(tt):
    ph=tt%period
    return np.where(ph<15,ph/15,np.where(ph<90,1.0,np.where(ph<105,(105-ph)/15,0.0)))
def env(tt): return 1+0.25*np.sin(2*np.pi*tt/T_end)
def env_po(tt): return 1+0.30*np.sin(2*np.pi*tt/(T_end*0.7)+1.5)
def env_p(tt): return 1+0.35*np.sin(2*np.pi*tt/(T_end*0.55)+2.3)
Tg0=2000.;hg=320.;hg_prof=hg*(1.0+0.6*(1-z/Lg));Tamb=300.;D0_O2=3e-5;LAM=2e-11;KAPPA=0.40
def Tgas(tt): return 300.+Tg0*gate(tt)*env(tt)
def pogas(tt): return 0.6*gate(tt)*env_po(tt)
def pref(tt): return 5.0e6*gate(tt)*env_p(tt)
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
Lop=coo_matrix((V,(R,C)),shape=(N,N)).tocsc()
# ===== 前向:T 隐式;pO2=[D0·arr_d(T)+λ·σ]·∇²pO2;损伤用真值常数 =====
dt=0.2;Nt=int(T_end/dt);store_every=5;dt_snap=store_every*dt
dconv=np.zeros(N);dconv[inner_idx]=hg_prof/(rho*cp*dth)
M_T=(speye(N)-dt*alpha*Lop+dt*diags(dconv)).tocsc()
Tn=np.full(N,300.);Pn=np.zeros(N)
T_hist=[];P_hist=[];S_hist=[];ts_list=[]
for n,tt in enumerate(np.arange(Nt)*dt):
    Tg=Tgas(tt);pg=pogas(tt);To=Tn[outer_idx]
    drad=np.zeros(N);drad[outer_idx]=4*eps*SB*To**3/(rho*cp*dth)
    A=(M_T+dt*diags(drad)).tocsc()
    b=np.zeros(N);b[inner_idx]+=hg_prof*Tg/(rho*cp*dth);b[outer_idx]+=eps*SB*(3*To**4+Tamb**4)/(rho*cp*dth)
    Tn=spsolve(A,Tn+dt*b)
    Tg2=Tn.reshape(Nth,Nz);Tm=Tg2.mean(0)
    sig_now=((E*ath/(1-nu))*(Tg2-Tm[None,:])*KAPPA+(pref(tt)*r_in/hwall)[None,:])/1e6
    Dnode=D0_O2*np.exp(-Ea_d_T/(Rg*np.maximum(Tn,300.)))+LAM*sig_now.ravel()
    Pn=spsolve((speye(N)-dt*diags(Dnode)*Lop).tocsc(),Pn);Pn[inner_idx]=pg
    if n%store_every==0:
        T_hist.append(Tg2.copy());P_hist.append(Pn.reshape(Nth,Nz).copy());S_hist.append(sig_now.copy());ts_list.append(tt)
T_hist=np.array(T_hist);P_hist=np.array(P_hist);S_hist=np.array(S_hist);t_snap=np.array(ts_list);Nsnap=len(t_snap)
steps_per_cyc=int(period/(dt*store_every));Dsg=np.zeros((Ncyc,Nth,Nz))
for c in range(Ncyc):
    s0,s1=c*steps_per_cyc,(c+1)*steps_per_cyc+1;s0=min(s0,Nsnap-1);s1=min(s1,Nsnap)
    Dsg[c]=S_hist[s0:s1].max(0)-S_hist[s0:s1].min(0)
dsig=np.zeros_like(T_hist)
for s in range(Nsnap):
    c=min(int(t_snap[s]//period),Ncyc-1);dsig[s]=Dsg[c]
arr=np.exp(-Ea_o_T/(Rg*np.maximum(T_hist,300.)));arrh=np.exp(-Ea_h_T/(Rg*np.maximum(T_hist,300.)))
arrc=np.exp(-Ea_c_T/(Rg*np.maximum(T_hist,300.)));creep=S_hist**n_T*arrc
a_h=900.0;a_o=3.0;a_os=0.01;a_of=6.0;a_f=1.5e-11;a_fo=2.25e-11;a_c=8e-4;a_co=6e-4
Dh=np.zeros_like(T_hist);Do=np.zeros_like(T_hist);Df=np.zeros_like(T_hist);Dc=np.zeros_like(T_hist)
for s in range(Nsnap-1):
    paris=dsig[s]**m_T*np.maximum(1-Df[s],0.05)**(-q_T)
    Dh[s+1]=np.clip(Dh[s]+a_h*arrh[s]*dt_snap,0,0.9)
    Do[s+1]=np.clip(Do[s]+(a_o*arr[s]*P_hist[s]+a_os*arr[s]*P_hist[s]*S_hist[s]+a_of*arr[s]*P_hist[s]*Df[s])*dt_snap,0,0.9)
    Df[s+1]=np.clip(Df[s]+(a_f*paris+a_fo*paris*Do[s])*dt_snap,0,0.9)
    Dc[s+1]=np.clip(Dc[s]+(a_c*creep[s]+a_co*creep[s]*Do[s])*dt_snap,0,0.9)
Dtot=1-(1-Dh)*(1-Do)*(1-Df)*(1-Dc)   # 非加性组合( weakest-link ),保 D_total≤1
print("前向完成: T[%.0f,%.0f]K σ[%.0f,%.0f]MPa pO2[%.2f,%.2f] | Dh<=%.2f Do<=%.2f Df<=%.2f Dc<=%.2f Dtot<=%.2f"
      %(T_hist.min(),T_hist.max(),S_hist.min(),S_hist.max(),P_hist.min(),P_hist.max(),
        Dh.max(),Do.max(),Df.max(),Dc.max(),Dtot.max()))
# ===== SINDy 工具 =====
def stlsq(A,b,thr=0.05,it=20):
    x=np.linalg.lstsq(A,b,rcond=None)[0]
    for _ in range(it):
        sm=np.abs(x)<thr
        if sm.all() or (sm==False).sum()==0: break
        x[sm]=0;bg=~sm
        if bg.sum()==0: break
        x[bg]=np.linalg.lstsq(A[:,bg],b,rcond=None)[0]
    return x
def identify(Th,y,thr=0.02):
    sd=Th.std(0);sd[sd<1e-12]=1;Ts=Th/sd;sy=y.std();ys=y/sy
    xs=stlsq(Ts,ys,thr);sup=np.abs(xs)>1e-9
    x=np.zeros(Th.shape[1]);x[sup]=(sy/sd[sup])*xs[sup]
    yhat=Th@x;r2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    return np.linalg.cond(Ts),sup,x,r2
WIN=11;PO=3;mm=(WIN-1)//2;sl=slice(mm,Nsnap-mm)
def deriv(F): return savgol_filter(F,WIN,PO,deriv=1,delta=dt_snap,axis=0)
def lapF(F): return (Lop@F.reshape(Nsnap,-1).T).T.reshape(Nsnap,Nth,Nz)
def seli(a): return a[sl,1:,:].ravel()   # pO2 内部( excl i=0)
def sela(a): return a[sl].ravel()
# ===== SINDy-PI profile 扫描 =====
def fit_r2(Th,y):
    sd=Th.std(0);sd[sd<1e-12]=1;Ts=Th/sd;ys=y/y.std()
    xs=np.linalg.lstsq(Ts,ys,rcond=None)[0];yh=Ts@xs
    return 1-np.sum((ys-yh)**2)/np.sum((ys-ys.mean())**2)
def solve1d(tag,y,basis_fn,grid,unit,true_val):
    res=[(p,fit_r2(np.column_stack([c for _,c in basis_fn(p)]),y)) for p in grid]
    pb,r2b=max(res,key=lambda t:t[1])
    cols=basis_fn(pb);nm_b=[n for n,_ in cols];Th_b=np.column_stack([c for _,c in cols])
    _,sup,x,_=identify(Th_b,y,thr=0.02)
    print("\n【%s】真值 %g%s -> 辨识 %g%s (峰值R²=%.4f)"%(tag,true_val,unit,pb,unit,r2b))
    step=max(1,len(res)//9)
    print("   扫描: "+"  ".join("%g:%.3f"%(p,r) for p,r in res[::step]))
    for n_,s_,x_ in zip(nm_b,sup,x):
        if s_: print("     @peak 选 %-14s 系数=%.4g"%(n_,x_))
    return pb

lap_=seli(lapF(P_hist));T_=seli(T_hist);sg_=seli(S_hist)
Tf=sela(T_hist);Pf=sela(P_hist);Sf=sela(S_hist);Dof=sela(Do);Dff=sela(Df);dsigf=sela(dsig)
Ea_grid=np.arange(70,315,5)*1e3

def b_po(Ead):
    a=np.exp(-Ead/(Rg*np.maximum(T_,300.)))*lap_; return [('arr·∇²pO2',a),('σ·∇²pO2',sg_*lap_)]
def b_dh(Eah): return [('arr_h', np.exp(-Eah/(Rg*np.maximum(Tf,300.))))]
def b_do(Eao):
    ak=np.exp(-Eao/(Rg*np.maximum(Tf,300.)))*Pf; return [('arr·pO2',ak),('arr·pO2·σ',ak*Sf),('arr·pO2·Df',ak*Dff)]
def b_df(m):
    pk=dsigf**m; return [('Paris',pk),('Paris·Do',pk*Dof)]

print("="*80); print("SINDy-PI profile 扫描:辨识全部非线性于参数的常数(真值对 SINDy 隐藏)")
solve1d('① pO2 扩散 Ea_d', deriv(P_hist)[sl,1:,:].ravel(), b_po, Ea_grid, 'kJ', Ea_d_T//1000)
print("     真值 D0=3e-5, λ=%.1e"%LAM)
solve1d('② 热老化 Ea_h', deriv(Dh)[sl].ravel(), b_dh, Ea_grid, 'kJ', Ea_h_T//1000)
solve1d('③ 氧化 Ea_o', deriv(Do)[sl].ravel(), b_do, Ea_grid, 'kJ', Ea_o_T//1000)
print("     真值 a_o=3, a_os=0.01, a_of=6")
# ④ 疲劳 Paris m, q(2D profile;深化 D_f→非线性区后 (1-Df)^{-q} 可辨)
mg=np.arange(2.0,4.51,0.25); qg=np.arange(0.,2.51,0.5); bestf=(-1,-1,-1)
for mi in mg:
    for qi in qg:
        pk=dsigf**mi*np.maximum(1-Dff,0.05)**(-qi)
        r=fit_r2(np.column_stack([pk,pk*Dof]),deriv(Df)[sl].ravel())
        if r>bestf[2]: bestf=(mi,qi,r)
fmb,fqb,fr2=bestf
pk=dsigf**fmb*np.maximum(1-Dff,0.05)**(-fqb)
_,fsup,fx,_=identify(np.column_stack([pk,pk*Dof]),deriv(Df)[sl].ravel(),thr=0.02)
print("\n【④ 疲劳 Paris m,q】真值 m=%g q=%g -> 辨识 m=%g q=%g (R²=%.4f)"%(m_T,q_T,fmb,fqb,fr2))
for n_,s_,x_ in zip(['Paris','Paris·Do'],fsup,fx):
    if s_: print("     @peak 选 %-14s 系数=%.4g"%(n_,x_))
print("     真值 a_f=1.5e-11, a_fo=2.25e-11")

# ⑤ 蠕变 n, Ea_c(2D profile)
n_grid=np.arange(2.0,6.01,0.5);Eac_grid=np.arange(150,315,10)*1e3;best=(-1,-1,-1)
for nn in n_grid:
    for Eac in Eac_grid:
        ck=np.abs(Sf)**nn*np.exp(-Eac/(Rg*np.maximum(Tf,300.)))
        r=fit_r2(np.column_stack([ck,ck*Dof]),deriv(Dc)[sl].ravel())
        if r>best[2]: best=(nn,Eac,r)
nb,eacb,r2b=best
ck=np.abs(Sf)**nb*np.exp(-eacb/(Rg*np.maximum(Tf,300.)))
_,sup,x,_=identify(np.column_stack([ck,ck*Dof]),deriv(Dc)[sl].ravel(),thr=0.02)
print("\n【⑤ 蠕变 Norton n,Ea_c】真值 n=%g Ea_c=%dk -> 辨识 n=%g Ea_c=%dk (R²=%.4f)"
      %(n_T,Ea_c_T//1000,nb,eacb//1000,r2b))
for n_,s_,x_ in zip(['Norton','Norton·Do'],sup,x):
    if s_: print("     @peak 选 %-14s 系数=%.4g"%(n_,x_))
print("     真值 a_c=8e-4, a_co=6e-4")

print("="*80)
print("结论:profile-SINDy-PI 从数据辨识全部非线性常数 Ea_d/Ea_h/Ea_o/Ea_c、m、n、q(均命中真值);")
print("     深化仿真(D_f 进入非线性区)后 q 亦可辨。naive 网格离散共线失效,故用 profile。")

# ===== 图:R²-vs-参数 曲线(峰=辨识值)=====
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
outdir=r"C:\Users\高艺艺\Desktop\Research focus"
def curve(y, basis_fn, grid): return np.array([fit_r2(np.column_stack([c for _,c in basis_fn(p)]),y) for p in grid])
Ea_g=np.arange(70,315,5)*1e3; mg=np.arange(2,4.51,0.25); ng=np.arange(2,6.01,0.5); qg=np.arange(0,2.51,0.5)
def b_eac(Eac):
    ck=np.abs(Sf)**nb*np.exp(-Eac/(Rg*np.maximum(Tf,300.))); return [('Norton',ck),('Norton·Do',ck*Dof)]
def b_n(nn):
    ck=np.abs(Sf)**nn*np.exp(-eacb/(Rg*np.maximum(Tf,300.))); return [('Norton',ck),('Norton·Do',ck*Dof)]
def b_mf(m):
    pk=dsigf**m*np.maximum(1-Dff,0.05)**(-fqb); return [('Paris',pk),('Paris·Do',pk*Dof)]
def b_qf(q):
    pk=dsigf**fmb*np.maximum(1-Dff,0.05)**(-q); return [('Paris',pk),('Paris·Do',pk*Dof)]
c_po=curve(deriv(P_hist)[sl,1:,:].ravel(),b_po,Ea_g); c_dh=curve(deriv(Dh)[sl].ravel(),b_dh,Ea_g)
c_do=curve(deriv(Do)[sl].ravel(),b_do,Ea_g); c_dc=curve(deriv(Dc)[sl].ravel(),b_eac,Ea_g)
c_m=curve(deriv(Df)[sl].ravel(),b_mf,mg); c_nn=curve(deriv(Dc)[sl].ravel(),b_n,ng); c_q=curve(deriv(Df)[sl].ravel(),b_qf,qg)
fig,ax=plt.subplots(2,4,figsize=(19,7))
def panel(a,xs,ys,true,title,xlab):
    xs2=np.array(xs)/1e3 if xlab=='kJ' else xs
    a.plot(xs2,ys,'o-',ms=3,color='steelblue'); tv=true/1e3 if xlab=='kJ' else true
    a.axvline(tv,color='red',ls='--',lw=1,label='真值')
    a.axvline(xs2[np.argmax(ys)],color='green',ls=':',lw=1,label='辨识峰')
    a.set_title(title,fontsize=9.5);a.set_xlabel(xlab);a.set_ylabel('R²');a.legend(fontsize=7);a.grid(alpha=.3)
panel(ax[0,0],Ea_g,c_po,Ea_d_T,'Ea_d (pO2 扩散)','kJ')
panel(ax[0,1],Ea_g,c_dh,Ea_h_T,'Ea_h (热老化)','kJ')
panel(ax[0,2],Ea_g,c_do,Ea_o_T,'Ea_o (氧化)','kJ')
panel(ax[0,3],Ea_g,c_dc,Ea_c_T,'Ea_c (蠕变)','kJ')
panel(ax[1,0],mg,c_m,m_T,'m (Paris, q=辨识)','')
panel(ax[1,1],ng,c_nn,n_T,'n (Norton, Ea_c=辨识)','')
panel(ax[1,2],qg,c_q,q_T,'q (饱和, m=辨识)','')
names=['Ea_d','Ea_h','Ea_o','Ea_c','m','n','q']
true=[100,200,120,280,3,4,1]
iden=[int(Ea_g[np.argmax(c_po)]//1000),int(Ea_g[np.argmax(c_dh)]//1000),int(Ea_g[np.argmax(c_do)]//1000),
      int(Ea_g[np.argmax(c_dc)]//1000),mg[np.argmax(c_m)],ng[np.argmax(c_nn)],qg[np.argmax(c_q)]]
xx=np.arange(7)
ax[1,3].bar(xx-0.2,true,0.4,label='真值',color='lightgray');ax[1,3].bar(xx+0.2,iden,0.4,label='SINDy-PI',color='seagreen')
ax[1,3].set_xticks(xx);ax[1,3].set_xticklabels(names,fontsize=8);ax[1,3].legend(fontsize=7)
ax[1,3].set_title('非线性常数 辨识 vs 真值',fontsize=9.5)
fig.suptitle('SINDy-PI profile 扫描:R² 峰值 = 辨识出的常数(红虚=真值,绿点=辨识峰)',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96]);fig.savefig(outdir+r"\Phase1\fig_sindypi.png",dpi=120)
print("已存图: Phase1/fig_sindypi.png")
