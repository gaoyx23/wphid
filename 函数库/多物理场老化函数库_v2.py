# -*- coding: utf-8 -*-
"""
多物理场老化函数库 v2 —— Θ_phys + Θ_dmg 双库（F+G 显式分组，含交互项）
====================================================================
设计原则（2026-07-24 初订；2026-07-28 与主脚本/论文同步）：
  * 场与损伤**不反向耦合**：物理场(T/σ/pO2)由边界条件+场×场交互驱动演化，
    损伤是被场**单向驱动**的结果（场→损伤 在损伤库 F 项里）；损伤不进任何场方程。
  * **先验 vs 辨识边界**（与论文一致）：
      - 已知先验、不予辨识：温度场(傅里叶导热 α=k/(ρcp)+边界换热/辐射)、应力代数本构 σ(T,p)
        （E,α_th,ν,κ）→ lib_T / lib_sigma 仅作备选，主流程不调用。
      - SINDy 辨识目标：pO2 + 四类损伤(Dh/Do/Df/Dc) 方程的 F 基项与 G 交互项系数（共 15 候选）。
      - 非线性常数 Ea,m,q,n 由 SINDy-PI 另行辨识；本库建候选时先固定之，保线性可估。
  * 交互项分两类、各在库**内部**：
      Θ_phys.G = 场×场交互（T→pO2 温度相关扩散 arr_d·∇²pO2、σ→pO2 应力调制扩散 σ·∇²pO2）
      Θ_dmg.G  = 损伤×损伤 / 场→损伤交互（应力助氧化 arr·pO2·σ、疲劳助氧化 arr·pO2·Df、
                 氧化-疲劳 Paris·Do、氧化-蠕变 Norton·Do）
  * Dh(纯 Arrhenius 热老化) 无损伤交互，仅 F 基项 arr_h(T)；arr_h·σ 为干扰候选(主流程中应被剔除)。

API：各 builder 接收 1D 状态数组，返回 [(列名, 数值列, 'F'/'G', 物理理由), ...]；
     matrix() 装配；report() 打印项 + 条件数 + 高共线对。依赖：仅 numpy。
"""
import numpy as np

# ---- 非线性常数(本合成研究的种入真值；建候选时固定以保线性可估，由 SINDy-PI 辨识恢复)----
m_f, q_f = 3.0, 1.0       # Paris 指数、(1-Df)^(-q) 饱和指数
n_c      = 4.0            # Norton 蠕变应力指数
Ea_o = 120e3              # 氧化活化能 J/mol
Ea_c = 280e3              # 蠕变活化能
Ea_h = 200e3              # 热老化活化能
Ea_d = 100e3              # 氧扩散温度相关活化能（典型值）
Rg  = 8.314


def arr(T, Ea):
    """Arrhenius 因子 exp(-Ea/RT)，T 为开尔文。"""
    return np.exp(-Ea / (Rg * np.asarray(T, dtype=float)))


# =====================================================================
# Θ_phys 物理场库（仅场×场交互，不含任何损伤量）
# =====================================================================
def lib_T(T, lapT, Tgas=None):
    """∂T/∂t 候选：F 导热 + G 辐射/对流。"""
    T = np.asarray(T)
    items = [('∇²T', np.asarray(lapT), 'F', 'Fourier 导热（确认项）'),
             ('T⁴', T ** 4, 'G', 'Stefan-Boltzmann 辐射（外壁面项）')]
    if Tgas is not None:
        items.append(('(Tgas−T)', np.asarray(Tgas) - T, 'G', '流体对流换热 fluid→T'))
    return items


def lib_pO2(pO2, lapPO2, T, sig=None):
    """∂pO2/∂t 候选：F 扩散 + G 温度相关扩散率(T→pO2)/应力调制扩散(σ→pO2)。"""
    pO2 = np.asarray(pO2); T = np.asarray(T)
    items = [('∇²pO2', np.asarray(lapPO2), 'F', 'Fick 扩散（确认项，常数扩散率）'),
             ('arr_d(T)·∇²pO2', arr(T, Ea_d) * np.asarray(lapPO2), 'G', '温度相关扩散率（T→pO2 场×场交互）')]
    if sig is not None:
        items.append(('σ·∇²pO2', np.asarray(sig) * np.asarray(lapPO2), 'G', '应力调制扩散（σ→pO2 候选）'))
    return items


def lib_sigma(T, p):
    """σ 代数本构（非 ∂σ/∂t 目标）：热应力(T→σ)+膜应力(p→σ)+非线性/交互候选。"""
    T = np.asarray(T); p = np.asarray(p)
    return [('T', T, 'G', '热应力 σ_th∝E·α_th·(T−Tref)/(1−ν)（T→σ）'),
            ('p', p, 'G', '膜应力 ∝p·R/t（fluid→σ）'),
            ('T²', T ** 2, 'G', '非线性热应力（α 随 T 变，候选）'),
            ('T·p', T * p, 'G', '热-压交互（T×p 候选）')]

# v（流场）：作已知强迫（气动热 Q_aero、压力 p）输入，不入辨识库。


# =====================================================================
# Θ_dmg 损伤库（每损伤量一个目标导数；F=场驱动机制律，G=损伤×损伤交互）
# =====================================================================
def lib_Dh(T, sig):
    """dDh/dt：F Arrhenius 热老化；G 应力调制。"""
    ah = arr(T, Ea_h)
    return [('arr_h(T)', ah, 'F', 'Arrhenius 热致退化'),
            ('arr_h·σ', ah * np.asarray(sig), 'G', '应力调制热老化（σ→Dh）')]


def lib_Do(T, pO2, sig, Df):
    """dDo/dt：F 氧化；G 疲劳助氧化(损伤×损伤)/应力助氧化。"""
    ao = arr(T, Ea_o) * np.asarray(pO2)
    return [('arr·pO2', ao, 'F', 'Arrhenius 氧化'),
            ('arr·pO2·Df', ao * np.asarray(Df), 'G', '疲劳裂纹露新面→加速氧化（D_f→D_o 损伤交互）'),
            ('arr·pO2·σ', ao * np.asarray(sig), 'G', '应力助氧化（σ→D_o，Molins 1997 等）')]


def lib_Df(dsig, T, Do, Dc, Df=None):
    """dDf/dN（按循环）：F Paris；G 氧化-疲劳/蠕变-疲劳(损伤×损伤)/TMF。dsig=Δσ。"""
    paris = np.asarray(dsig) ** m_f
    if Df is not None:
        paris = paris * (1 - np.asarray(Df)) ** (-q_f)
    return [('Paris', paris, 'F', 'Paris 疲劳律（按循环）'),
            ('Paris·Do', paris * np.asarray(Do), 'G', '氧化致脆→加速疲劳（D_o→D_f 损伤交互）'),
            ('Paris·Dc', paris * np.asarray(Dc), 'G', '蠕变-疲劳交互（D_c→D_f 损伤交互）'),
            ('Paris·T', paris * np.asarray(T), 'G', '热-机械疲劳 TMF（T→D_f）')]


def lib_Dc(sig, T, Do, Df):
    """dDc/dt：F Norton；G 氧化-蠕变/蠕变-疲劳(损伤×损伤)。"""
    creep = np.asarray(sig) ** n_c * arr(T, Ea_c)
    return [('Norton', creep, 'F', 'Norton 蠕变 σ^n·e^−Eac/RT'),
            ('Norton·Do', creep * np.asarray(Do), 'G', '氧化影响蠕变（D_o→D_c 损伤交互）'),
            ('Norton·Df', creep * np.asarray(Df), 'G', '蠕变-疲劳交互（D_f→D_c 损伤交互）')]


# =====================================================================
# 工具：装配矩阵 + 共线性预检
# =====================================================================
def matrix(items):
    names = [x[0] for x in items]
    Theta = np.column_stack([np.asarray(x[1], float).ravel() for x in items])
    return names, Theta


def report(name, items):
    names, Theta = matrix(items)
    print("\n" + "=" * 80)
    print("【%s】候选 %d 项（F=自演化/机制律，G=交叉耦合/交互）：" % (name, len(items)))
    for cn, _, tag, why in items:
        print("  [%s] %-20s | %s" % (tag, cn, why))
    mu = Theta.mean(0); sd = Theta.std(0); sd[sd < 1e-12] = 1
    Thetas = (Theta - mu) / sd
    corr = np.corrcoef(Thetas.T); cond = np.linalg.cond(Thetas)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
             if abs(corr[i, j]) > 0.95]
    print("  -> 条件数 cond = %.1f ；高共线对(|r|>0.95)：%s" % (cond, "无" if not pairs else ""))
    for a, b in pairs:
        print("       %s ~ %s" % (a, b))
    return cond, pairs


# =====================================================================
# 演示（占位多工况轨迹，仅共线性预检）
# =====================================================================
if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    np.random.seed(0)
    M = 800
    t = np.arange(M, dtype=float)

    def w(f, ph=0.0, a=1.0):
        return a * np.sin(2 * np.pi * t / f + ph)

    T    = 1050 + 300 * w(110) + 90 * w(41, 1.2)
    sig  = 220 + 80 * w(82, 0.5) + 25 * w(29, 2.1)
    dsig = 160 + 55 * w(63, 2.0) + 18 * w(25, 0.3)
    pO2  = 0.5 + 0.25 * w(130, 3.0) + 0.08 * w(47, 1.7)
    p    = 8.0 + 2.0 * w(95, 0.7)
    Dh   = 0.20 * (t / M) ** 1.3
    Do   = 0.30 * (t / M) ** 1.8
    Df   = 0.40 * (t / M) ** 1.5
    Dc   = 0.15 * (t / M) ** 2.0
    lapT   = -40 + 18 * w(95, 0.8)
    lapPO2 = -0.05 + 0.02 * w(70, 1.1)
    Tgas   = 2400 + 200 * w(120, 0.2)

    print("############ Θ_phys 物理场库（仅场×场交互，无损伤量）############")
    report("物理场-温度 ∂T/∂t", lib_T(T, lapT, Tgas))
    report("物理场-氧分压 ∂pO2/∂t", lib_pO2(pO2, lapPO2, T, sig))
    report("物理场-应力 σ（代数本构）", lib_sigma(T, p))
    print("\n  （v 流场作已知强迫 Q_aero/p 输入，不入辨识库）")

    print("\n############ Θ_dmg 损伤库（F=场驱动机制律，G=损伤×损伤交互）############")
    report("损伤-热老化 dDh/dt", lib_Dh(T, sig))
    report("损伤-氧化 dDo/dt", lib_Do(T, pO2, sig, Df))
    report("损伤-疲劳 dDf/dN", lib_Df(dsig, T, Do, Dc, Df))
    report("损伤-蠕变 dDc/dt", lib_Dc(sig, T, Do, Df))

    print("\n" + "=" * 80)
    print("库共 22 候选：Θ_phys=10（T三+pO2三+σ四）+ Θ_dmg=12（Dh二+Do三+Df四+Dc三）。")
    print("主流程实际辨识 5 个方程(pO2+Dh/Do/Df/Dc)：3+2+3+4+3 = 15 候选。")
    print("先验不辨识：T(导热 α+边界)、σ 代数本构 —— lib_T/lib_sigma 仅备选。")
    print("非线性常数 Ea,m,q,n 由 SINDy-PI 另辨识；此库固定之以保候选线性可估。")
    print("注：共线性基于占位轨迹；真实 FEA/遥测数据上线后须对每个目标库重跑 report()。")
    print("=" * 80)
