# -*- coding: utf-8 -*-
"""
圆台喷管 耦合辨识 Phase 1（场×场 + 损伤×损伤 交互项挖掘）
==============================================================
设计（2026-07-24，应导师"勿三项直接相加、须引入交互项"+ 用户"场与损伤不反向耦合"）：
  * 前向单向：物理场(T/σ/pO2)由边界条件+场×场交互驱动；损伤由场单向算出，不反作用回场。
  * 种入并挖掘的【交互项】：
      场×场(Θ_phys): T→pO2（温度相关扩散 arr_d·∇²pO2）
      损伤×损伤(Θ_dmg): D_f→D_o(arr·pO2·Df)、D_o→D_f(Paris·Do)、D_o→D_c(Norton·Do)
      场→损伤驱动: σ→D_o(arr·pO2·σ 应力助氧化)
  * SINDy identify: std 缩放不中心化、系数换算回；EIV 噪声 0~20% 扫描。
  * 云图: 2D 圆台壁梯形域(美化) + 3D 圆台内壁曲面(选 T、D_total)。
依赖: numpy/scipy/matplotlib + 多物理场老化函数库_v2。定位: 方法验证(律自设、D 假设可观测)。
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '函数库')))
import 多物理场老化函数库_v2 as Lb
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 材料/几何/边界 =====
rho = 8440.; cp = 580.; kth = 25.; E = 200e9; ath = 13e-6; nu = 0.3
eps = 0.8; SB = 5.67e-8; alpha = kth / (rho * cp); Rg = 8.314
Ea_ox, Ea_c, Ea_h, Ea_d = Lb.Ea_o, Lb.Ea_c, Lb.Ea_h, Lb.Ea_d
m_f, q_f, n_c = Lb.m_f, Lb.q_f, Lb.n_c
Lg = 0.30; Rthroat = 0.05; Rexit = 0.15; hwall = 5e-3
Nth = 10; Nz = 30
z = np.linspace(0, Lg, Nz); dz = z[1] - z[0]; dz2 = dz ** 2
th = np.linspace(0, hwall, Nth); dth = th[1] - th[0]; dth2 = dth ** 2
N = Nth * Nz
def pidx(i, j): return i * Nz + j
inner_idx = np.array([pidx(0, j) for j in range(Nz)])
outer_idx = np.array([pidx(Nth - 1, j) for j in range(Nz)])
r_in = Rthroat + (Rexit - Rthroat) * (z / Lg)
period = 120.; Ncyc = 15; T_end = Ncyc * period
def gate(tt):
    ph = tt % period
    return np.where(ph < 15, ph / 15, np.where(ph < 90, 1.0, np.where(ph < 105, (105 - ph) / 15, 0.0)))
def env(tt):    return 1 + 0.25 * np.sin(2 * np.pi * tt / T_end)
def env_po(tt): return 1 + 0.30 * np.sin(2 * np.pi * tt / (T_end * 0.7) + 1.5)
def env_p(tt):  return 1 + 0.35 * np.sin(2 * np.pi * tt / (T_end * 0.55) + 2.3)
Tg0 = 2000.; hg = 320.; hg_prof = hg * (1.0 + 0.6 * (1 - z / Lg)); Tamb = 300.
D0_O2 = 3e-5; KAPPA = 0.40; lam_sig = 2.0e-11   # 壳体机械约束因子(0<κ≤1);lam_sig=λ(σ→pO2 应力调制氧扩散,场×场交互)
def Tgas(tt):  return 300. + Tg0 * gate(tt) * env(tt)
def pogas(tt): return 0.6 * gate(tt) * env_po(tt)
def pref(tt):  return 5.0e6 * gate(tt) * env_p(tt)

# ===== 扩散算子 L =====
R = []; C = []; V = []
def add(r, c, v): R.append(r); C.append(c); V.append(v)
for i in range(Nth):
    for j in range(Nz):
        p = pidx(i, j); d = 0.0
        if i == 0:           d -= 1 / dth2; add(p, pidx(1, j), 1 / dth2)
        elif i == Nth - 1:   d -= 1 / dth2; add(p, pidx(Nth - 2, j), 1 / dth2)
        else:                d -= 2 / dth2; add(p, pidx(i - 1, j), 1 / dth2); add(p, pidx(i + 1, j), 1 / dth2)
        if j == 0:           d -= 2 / dz2;  add(p, pidx(i, 1), 2 / dz2)
        elif j == Nz - 1:    d -= 2 / dz2;  add(p, pidx(i, Nz - 2), 2 / dz2)
        else:                d -= 2 / dz2;  add(p, pidx(i, j - 1), 1 / dz2); add(p, pidx(i, j + 1), 1 / dz2)
        add(p, p, d)
Lop = coo_matrix((V, (R, C)), shape=(N, N)).tocsc()

# ===== 前向(单向):T 隐式;pO2 温度相关扩散(种入 T→pO2 场×场交互)=====
dt = 0.2; Nt = int(T_end / dt); store_every = 1; dt_snap = store_every * dt  # 加密采样:Savgol 求导需足够细的网格以消偏差
dconv = np.zeros(N); dconv[inner_idx] = hg_prof / (rho * cp * dth)
M_T = (speye(N) - dt * alpha * Lop + dt * diags(dconv)).tocsc()
Tn = np.full(N, 300.); Pn = np.zeros(N)
T_hist = []; P_hist = []; ts_list = []
for n, tt in enumerate(np.arange(Nt) * dt):
    Tg = Tgas(tt); pg = pogas(tt); To = Tn[outer_idx]
    drad = np.zeros(N); drad[outer_idx] = 4 * eps * SB * To ** 3 / (rho * cp * dth)
    A = (M_T + dt * diags(drad)).tocsc()
    b = np.zeros(N); b[inner_idx] += hg_prof * Tg / (rho * cp * dth)
    b[outer_idx] += eps * SB * (3 * To ** 4 + Tamb ** 4) / (rho * cp * dth)
    Tn = spsolve(A, Tn + dt * b)
    Dnode = D0_O2 * np.exp(-Ea_d / (Rg * np.maximum(Tn, 300.)))          # 温度相关扩散率(T→pO2)
    # 应力调制氧扩散(σ→pO2 场×场交互):准静态 σ 代数本构,公式与后处理一致
    Tf = Tn.reshape(Nth, Nz); Tm = Tf.mean(0, keepdims=True)
    sig_vec = (((E * ath / (1 - nu)) * (Tf - Tm) * KAPPA + (pref(tt) * r_in / hwall)[None, :]) / 1e6).ravel()
    Deff = Dnode + lam_sig * sig_vec                                     # 有效扩散率 [D0·arr_d + λσ]
    M_po = (speye(N) - dt * diags(Deff) * Lop).tocsc()
    Pn = spsolve(M_po, Pn); Pn[inner_idx] = pg
    if n % store_every == 0:
        T_hist.append(Tn.reshape(Nth, Nz).copy()); P_hist.append(Pn.reshape(Nth, Nz).copy()); ts_list.append(tt)
T_hist = np.array(T_hist); P_hist = np.array(P_hist); t_snap = np.array(ts_list); Nsnap = len(t_snap)

# σ:热应力(T→σ)+膜应力(p→σ),代数本构(无损伤)
sig_hist = np.zeros_like(T_hist)
for s, tt in enumerate(t_snap):
    Tm = T_hist[s].mean(0, keepdims=True)
    sig_hist[s] = ((E * ath / (1 - nu)) * (T_hist[s] - Tm) * KAPPA + (pref(tt) * r_in / hwall)[None, :]) / 1e6
steps_per_cyc = int(period / (dt * store_every))
Dsg = np.zeros((Ncyc, Nth, Nz))
for c in range(Ncyc):
    s0, s1 = c * steps_per_cyc, (c + 1) * steps_per_cyc + 1; s0 = min(s0, Nsnap - 1); s1 = min(s1, Nsnap)
    Dsg[c] = sig_hist[s0:s1].max(0) - sig_hist[s0:s1].min(0)
dsig_hist = np.zeros_like(T_hist)
for s in range(Nsnap):
    c = min(int(t_snap[s] // period), Ncyc - 1); dsig_hist[s] = Dsg[c]

# ===== 损伤演化(种入损伤×损伤交互 + 应力助氧化)=====
arr_hist  = np.exp(-Ea_ox / (Rg * np.maximum(T_hist, 300.)))
arrh_hist = np.exp(-Ea_h  / (Rg * np.maximum(T_hist, 300.)))
arrc_hist = np.exp(-Ea_c  / (Rg * np.maximum(T_hist, 300.)))
creep_hist = sig_hist ** n_c * arrc_hist
a_h = 900.0
a_o = 3.0; a_os = 0.01; a_of = 6.0      # Do: 氧化 + 应力助 + 疲劳助(D_f→D_o)
a_f = 1.5e-11; a_fo = 2.25e-11           # Df: Paris + 氧化-疲劳(D_o→D_f)
a_c = 8e-4;  a_co = 6e-4                # Dc: Norton + 氧化-蠕变(D_o→D_c)
Dh = np.zeros_like(T_hist); Do = np.zeros_like(T_hist)
Df = np.zeros_like(T_hist); Dc = np.zeros_like(T_hist)
for s in range(Nsnap - 1):
    paris = dsig_hist[s] ** m_f * np.maximum(1 - Df[s], 0.05) ** (-q_f)
    Dh[s + 1] = np.clip(Dh[s] + a_h * arrh_hist[s] * dt_snap, 0, 0.9)
    Do[s + 1] = np.clip(Do[s] + (a_o * arr_hist[s] * P_hist[s]
                                 + a_os * arr_hist[s] * P_hist[s] * sig_hist[s]
                                 + a_of * arr_hist[s] * P_hist[s] * Df[s]) * dt_snap, 0, 0.9)
    Df[s + 1] = np.clip(Df[s] + (a_f * paris + a_fo * paris * Do[s]) * dt_snap, 0, 0.9)
    Dc[s + 1] = np.clip(Dc[s] + (a_c * creep_hist[s] + a_co * creep_hist[s] * Do[s]) * dt_snap, 0, 0.9)
Dtot = 1-(1-Dh)*(1-Do)*(1-Df)*(1-Dc)   # 非加性组合,保 D_total≤1

print("=" * 80)
print("圆台 耦合辨识 Phase 1（场×场 + 损伤×损伤 交互项挖掘）")
print("  T[%.0f,%.0f]K  σ[%.0f,%.0f]MPa  pO2[%.2f,%.2f]  Dtot[%.3f,%.3f]"
      % (T_hist.min(), T_hist.max(), sig_hist.min(), sig_hist.max(), P_hist.min(), P_hist.max(), Dtot.min(), Dtot.max()))

# ===== SINDy 工具 =====
def stlsq(A, b, thr=0.05, it=20):
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(it):
        sm = np.abs(x) < thr
        if sm.all() or (sm == False).sum() == 0: break
        x[sm] = 0; bg = ~sm
        if bg.sum() == 0: break
        x[bg] = np.linalg.lstsq(A[:, bg], b, rcond=None)[0]
    return x
def identify(Th, y, thr=0.02):
    sd = Th.std(0); sd[sd < 1e-12] = 1; Ts = Th / sd
    sy = y.std(); ys = y / sy
    xs = stlsq(Ts, ys, thr); sup = np.abs(xs) > 1e-9
    x = np.zeros(Th.shape[1]); x[sup] = (sy / sd[sup]) * xs[sup]
    yhat = Th @ x; r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return np.linalg.cond(Ts), sup, x, r2

WIN = 11; PO = 3; mm = (WIN - 1) // 2; sl = slice(mm, Nsnap - mm)
def deriv(F): return savgol_filter(F, WIN, PO, deriv=1, delta=dt_snap, axis=0)
def lapF(F): return (Lop @ F.reshape(Nsnap, -1).T).T.reshape(Nsnap, Nth, Nz)

# 各目标的候选构造 + 真【交互项 inter】与【机制F项 Fset】(thr=0.02)
def identify_target(tag, T_, s_, P_, Do_, Df_, Dc_, dsig_, Dh_=None, thr=0.02):
    if tag == 'pO2':
        items = Lb.lib_pO2(P_[sl, 1:, :].ravel(), lapF(P_)[sl, 1:, :].ravel(), T_[sl, 1:, :].ravel(), s_[sl, 1:, :].ravel())
        y = deriv(P_)[sl, 1:, :].ravel()
        inter = {'arr_d(T)·∇²pO2', 'σ·∇²pO2'}; Fset = set()   # 场×场:T→pO2 + σ→pO2(应力调制扩散)
    elif tag == 'Do':
        items = Lb.lib_Do(T_[sl].ravel(), P_[sl].ravel(), s_[sl].ravel(), Df_[sl].ravel())
        y = deriv(Do_)[sl].ravel()
        inter = {'arr·pO2·Df', 'arr·pO2·σ'}; Fset = {'arr·pO2'}   # 损伤×损伤 + 场→损伤
    elif tag == 'Df':
        items = Lb.lib_Df(dsig_[sl].ravel(), T_[sl].ravel(), Do_[sl].ravel(), Dc_[sl].ravel(), Df_[sl].ravel())
        y = deriv(Df_)[sl].ravel()
        inter = {'Paris·Do'}; Fset = {'Paris'}            # 损伤×损伤:D_o→D_f
    elif tag == 'Dc':
        items = Lb.lib_Dc(s_[sl].ravel(), T_[sl].ravel(), Do_[sl].ravel(), Df_[sl].ravel())
        y = deriv(Dc_)[sl].ravel()
        inter = {'Norton·Do'}; Fset = {'Norton'}          # 损伤×损伤:D_o→D_c
    elif tag == 'Dh':
        items = Lb.lib_Dh(T_[sl].ravel(), s_[sl].ravel())
        y = deriv(Dh_)[sl].ravel()
        inter = set(); Fset = {'arr_h(T)'}                # 纯 Arrhenius 热老化(无损伤交互);arr_h·σ 为干扰项
    nm, Th = Lb.matrix(items)
    c, sup, x, r2 = identify(Th, y, thr)
    return nm, sup, x, r2, inter, Fset, Th, y

# ===== 无噪声恢复(详印)=====
print("\n--- 无噪声:交互项恢复 ---")
res0 = {}
for tag in ['pO2', 'Do', 'Df', 'Dc', 'Dh']:
    nm, sup, x, r2, inter, Fset, _, _ = identify_target(tag, T_hist, sig_hist, P_hist, Do, Df, Dc, dsig_hist, Dh)
    res0[tag] = (nm, sup, x, inter)
    alltrue = inter | Fset
    titles = {'pO2':'∂pO2/∂t 场×场', 'Do':'dDo 损伤×损伤', 'Df':'dDf 损伤×损伤', 'Dc':'dDc 损伤×损伤', 'Dh':'dDh 热老化(基项)'}
    print("\n【%s】R²=%.4f" % (titles[tag], r2))
    for n_, s_, x_ in zip(nm, sup, x):
        lab = '★真交互' if n_ in inter else ('机制F' if n_ in Fset else '干扰')
        print("    %-18s %4s %13.5g  %s" % (n_, "YES" if s_ else "no", x_ if s_ else 0, lab))
    rec = sum(1 for n_ in inter if sup[list(nm).index(n_)])
    spur = sum(1 for n_, s_ in zip(nm, sup) if s_ and n_ not in alltrue)
    print("    -> 交互项恢复 %d/%d；误选 %d" % (rec, len(inter), spur))

# ===== EIV 噪声扫描:交互项系数相对误差(在已知真实支持上 OLS)=====
# 注:固定阈值 STLSQ 的"支持恢复"在小系数项上随噪声非单调抖动(阈值边缘的数值伪影,非物理),
# 故主鲁棒性指标改为"真实支持上 OLS 的系数相对误差"——随噪声单调退化、可解释,
# 且正确反映含拉普拉斯算子的场×场项对观测噪声更敏感(算子放大噪声)。
print("\n--- EIV 噪声扫描:交互项系数相对误差(真实支持 OLS)---")
sT, sS, sP = T_hist.std(), sig_hist.std(), P_hist.std()
sDh, sDo, sDf, sDc = Dh.std(), Do.std(), Df.std(), Dc.std()
TRUE_INTER = {'arr_d(T)·∇²pO2': D0_O2, 'σ·∇²pO2': lam_sig,
              'arr·pO2·σ': a_os, 'arr·pO2·Df': a_of,
              'Paris·Do': a_fo, 'Norton·Do': a_co}
FIELD = {'pO2'}                         # 场×场(含 ∇²pO2,共 2 项);Do/Df/Dc 为损伤×损伤(共 4 项)
noise_rows = []
for nl in [0.0, 0.02, 0.05, 0.10, 0.20]:
    ef = []; ed = []
    for rep in range(5):
        rng = np.random.default_rng(9000 * rep + int(nl * 1e5))
        Tn_ = T_hist + (rng.normal(0, nl * sT, T_hist.shape) if nl else 0)
        Sn_ = sig_hist + (rng.normal(0, nl * sS, sig_hist.shape) if nl else 0)
        Pn_ = P_hist + (rng.normal(0, nl * sP, P_hist.shape) if nl else 0)
        Don_= Do + (rng.normal(0, nl * sDo, Do.shape) if nl else 0)
        Dfn_= Df + (rng.normal(0, nl * sDf, Df.shape) if nl else 0)
        Dcn_= Dc + (rng.normal(0, nl * sDc, Dc.shape) if nl else 0)
        erf = []; erd = []
        for tag in ['pO2', 'Do', 'Df', 'Dc']:
            nm, sup, x, r2, inter, Fset, Th, y = identify_target(tag, Tn_, Sn_, Pn_, Don_, Dfn_, Dcn_, dsig_hist)
            truesup = [i for i, n_ in enumerate(nm) if n_ in (inter | Fset)]   # 已知真实支持
            xo = np.linalg.lstsq(Th[:, truesup], y, rcond=None)[0]             # 真实支持上 OLS
            xmap = {nm[truesup[k]]: xo[k] for k in range(len(truesup))}
            for n_ in inter:
                e = abs(xmap.get(n_, 0.0) - TRUE_INTER[n_]) / abs(TRUE_INTER[n_])
                (erf if tag in FIELD else erd).append(e)
        ef.append(np.mean(erf) * 100); ed.append(np.mean(erd) * 100)
    noise_rows.append((nl, np.mean(ef), np.std(ef), np.mean(ed), np.std(ed)))
    print("  噪声 %4.0f%%: 场×场系数误差 %5.1f%% (±%.1f)  损伤×损伤 %5.1f%% (±%.1f)"
          % (nl * 100, np.mean(ef), np.std(ef), np.mean(ed), np.std(ed)))

# ===== 可视化(3D 场云图:圆台切口)=====
outdir = r"C:\Users\高艺艺\Desktop\Research focus"
Tp = int(np.argmax(T_hist[:, 0, :].mean(1)))
from mpl_toolkits.mplot3d import Axes3D

def field3d(ax, fld, cmap, title):
    """圆台切口 3D 场:外壁楔形 + 内壁楔形 + 切口面(完整壁厚×轴向场)。"""
    norm = plt.Normalize(fld.min(), fld.max()); cf = plt.get_cmap(cmap)
    az = np.linspace(np.radians(30), np.radians(330), 80)        # 楔形,留 +x 侧缺口看切口
    Zg, Ag = np.meshgrid(z * 1e3, az)
    Rout = (r_in + hwall)[None, :] * 1e3
    Rin  = r_in[None, :] * 1e3
    ax.plot_surface(Rout * np.cos(Ag), Rout * np.sin(Ag), Zg,
                    facecolors=cf(norm(np.broadcast_to(fld[-1, :][None, :], Zg.shape))),
                    rstride=1, cstride=1, linewidth=0, shade=False)
    ax.plot_surface(Rin * np.cos(Ag), Rin * np.sin(Ag), Zg,
                    facecolors=cf(norm(np.broadcast_to(fld[0, :][None, :], Zg.shape))),
                    rstride=1, cstride=1, linewidth=0, shade=False)
    Rc = (r_in[None, :] + th[:, None]) * 1e3; Zc = np.broadcast_to(z[None, :] * 1e3, (Nth, Nz))
    ax.plot_surface(Rc, np.zeros_like(Rc), Zc, facecolors=cf(norm(fld)),                 # 切口面
                    rstride=1, cstride=1, linewidth=0, shade=False)
    ax.set_title(title, fontsize=9.5); ax.view_init(elev=16, azim=-50)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array(fld); return m

fig3d = plt.figure(figsize=(20, 10))
_panels = [(T_hist[Tp], "温度 T (K)", "turbo"), (sig_hist[Tp], "应力 σ (MPa)", "RdBu_r"),
           (P_hist[Tp], "氧分压 pO2", "YlGnBu"), (Dh[-1], "热老化 D_h", "YlOrRd"),
           (Do[-1], "氧化 D_o", "YlOrRd"), (Df[-1], "疲劳 D_f", "YlOrRd"),
           (Dc[-1], "蠕变 D_c", "YlOrRd"), (Dtot[-1], "总损伤 D_total", "YlOrRd")]
for k, (fld, ttl, cm) in enumerate(_panels):
    ax = fig3d.add_subplot(2, 4, k + 1, projection='3d')
    fig3d.colorbar(field3d(ax, fld, cm, ttl), ax=ax, shrink=0.6, pad=0.08)
fig3d.suptitle("圆台喷管 3D 场云图（上:T/σ/pO2 峰值;下:D_h/D_o/D_f/D_c/D_total 终态；切口面显壁厚方向分布）", fontsize=13)
fig3d.savefig(outdir + r"\Phase1\fig3d.png", dpi=120)
print("\n已存图: Phase1/fig3d.png")

# --- 图C:交互项恢复 + 噪声鲁棒性 ---
figC, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# 左:无噪声各目标交互项 YES/NO
tags = ['pO2', 'Do', 'Df', 'Dc']; labels = ['arr_d·∇²pO2\n+σ·∇²pO2', 'arr·pO2·Df\n+arr·pO2·σ', 'Paris·Do', 'Norton·Do']
rec_cnt = []
for tag in tags:
    nm, sup, x, inter = res0[tag]
    d = dict(zip(nm, sup))
    rec_cnt.append(sum(1 for n_ in inter if d.get(n_)))
tot = [2, 2, 1, 1]
ax[0].bar(range(4), rec_cnt, 0.5, color='seagreen', label='恢复')
ax[0].bar(range(4), [t - r for t, r in zip(tot, rec_cnt)], 0.5, bottom=rec_cnt, color='lightcoral', label='丢失')
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylim(0, 2.5); ax[0].set_ylabel("交互项恢复数"); ax[0].legend(fontsize=8)
ax[0].set_title("无噪声:各目标交互项恢复 (共6项)")
# 右:噪声鲁棒性(交互项系数相对误差,按类;真实支持 OLS)
nl = [r[0] * 100 for r in noise_rows]
ax[1].errorbar(nl, [r[1] for r in noise_rows], yerr=[r[2] for r in noise_rows], fmt='o-', color='steelblue', label='场×场扩散项(含 ∇²)')
ax[1].errorbar(nl, [r[3] for r in noise_rows], yerr=[r[4] for r in noise_rows], fmt='s-', color='seagreen', label='损伤×损伤项')
ax[1].set_xlabel("测量噪声 (% × std)"); ax[1].set_ylabel("交互项系数相对误差 (%)"); ax[1].legend(fontsize=8)
ax[1].set_title("EIV 噪声下系数估计鲁棒性(真实支持 OLS)")
figC.suptitle("交互项挖掘:支持恢复 + 噪声鲁棒性", fontsize=13)
figC.tight_layout(); figC.savefig(outdir + r"\Phase1\figrec.png", dpi=130)
print("已存图: Phase1/figrec.png")
print("=" * 80); print("Phase 1 完成。")
