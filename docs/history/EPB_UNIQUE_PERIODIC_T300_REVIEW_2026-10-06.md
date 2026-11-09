# 唯一周期格三轨300 s验收与同时间对照

2026-10-06 16:48跟进（Asia/Shanghai）。从既有60 s checkpoint同hash续算至300 s，未重跑旧粗/细算例。实际源码仍为C盘dirty工作树。

## 状态与门禁

schema3 checkpoint为300 s/1200步，hash为28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1。driver已退出，命令行核查无Python进程，stderr为空。三轨checkpoint/final/末快照密度逐元素一致，EM线变量checkpoint/final一致，全部浮点checkpoint数组有限，密度最小约1.52898e7 m^-3且为正。60 s成果仍在archive_t60。

**严格300 s阶段门禁通过：0失败、2警告。** dt=0.25 s、零宏观减步；ES最大CFL local0.0160429、column0.00266135；EM combined CFL=0.4接近设定上限（警告），最大耦合残差6.74112e-13、功率配对缺陷6.22969e-16。第二警告是开放垂直边界缺少通量历史，原始粒子列漂移不能算封闭守恒误差。表面电流平衡归一比8.03310e-17，FAC/重力有效期中位数0.241603，末点0.240507。R=0、feedback=1、物理表面FAC、独立H_n/H_nu均保持。

线能量36608.07868 J；采样净边界功36612.48241 J，功减能4.40374 J，相对总交换差8.98130e-6，数值生能比例0。远端吸收490322.95703 J。预算为代表stage采样功率梯形积分，非精确离散SSPRK-stage功。

## 数值契约修正与旧失败

旧算子、输运和FAC模态把全部存储列视为独立周期格，但旧坐标/种子/验收又解释为重复端点；历史细/粗实际计算周期251.25/252.5 km，种子波长250 km。旧f7轨迹3150 s的两ES端点门禁失败仍保持原判，不改旧数组/hash/门限。

新periodic_x_layout默认legacy_stored_cells并保持两旧生产hash。显式unique_periodic_cells进入hash，以200个唯一x格、dx1.25 km覆盖250 km周期，坐标存0至248.75 km。runner检查周期适配种子，acceptance/postprocess保留全部唯一列；不同首末格无重复端点相等要求。39项相关测试通过。源码修正在em_growth.py、em_growth_acceptance.py、em_growth_postprocess.py，永久测试tests/test_em_periodic_grid_layout.py。源文件修改前副本保存于E盘tmp/periodic_grid_patch_backup_20261006。

## 同300 s增长与形态

200–300 s每宏观dt0.25 s等权OLS拟合ln归档RMS，各401样本。旧RMS包含全部历史存储列；空间比较另映射到共同物理节点，不混用定义。

| 轨道 | 修正细gamma(s^-1) | 旧细gamma(s^-1) | 旧粗gamma(s^-1) |
|---|---:|---:|---:|
| local ES | 6.25285e-6 | 6.65631e-6 | 6.89475e-6 |
| column ES | 4.52440e-4 | 4.52819e-4 | 4.52820e-4 |
| column EM | 3.24473e-4 | 3.24620e-4 | 3.26062e-4 |

local ES在此窗经过幅度谷值，R²仅0.136，不能当稳定指数增长；column ES/EM R²均>0.9999。修正EM相对旧细拟合率差约−0.0453%，相对旧粗约−0.4875%。纯EM抑制须对column ES比较：修正晚窗抑制约28.28%；几何效应仍column ES−local ES。

共同节点EM总密度/横向扰动相对L2差：修正对旧细0.006163%/0.013762%，对旧粗0.018012%/0.166385%。旧端点列仅在比较时去除，不修改存档；对粗网格新数组[::2,::2]映射。FAC有效期中位数修正/旧细/旧粗为0.241603/0.241705/0.241635；采样远端吸收分别490322.96/495182.38/500099.33 J。计算域面积和周期不同，能量差不能全归为物理电磁效应。

新与旧细300 s图件均已view_image QA：上部主密度结构平滑，无明显接缝或绘图缺陷。修正两ES有小型D>=0.25耗空连通域（local顶部266 km、column顶部264 km），EM最大耗空0.13735，尚无D>=0.25连通域，无清晰Y分叉。300 s只验证早期阶段；不同周期定义、两分辨率及同步FAC截断变化均不支持严格收敛声明。

## 尚未完成与下一步

建议先归档300 s成果，再从现有checkpoint同hash续算3150 s以检验成熟期；当前未启动。5000 s修正轨迹、真实E-region终端电导、默认零且三轨共同施加0/10/20 m/s上升ExB、场线长度/Alfvén速度扫描均未实施，保持用户规定次序。

既有物理失败根因与方程仍沿用：固定电势R=-1短路反射、错误FAC体源1/ds、单一H混淆粒子列与电导列。现matched R=0，推进N=H_n*n，

div[H_nu*n*nu_a*grad(psi)] = −g*d_x(H_n*n) + (B_a/m_i)*J_surface。

没有翻FAC符号或减弱反馈。模型缺平行质量输运、完整双半球metric、化学、中性风/剪切、PRE和动态E层电导，不能视为定量预测。

## 可复核路径

- [终态/能量/对照审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T300_AUDIT_2026-10-06.json)
- [严格阶段验收](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/acceptance_t300.json)
- [新三轨图件](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/postprocess_t300/em_growth_comparison.png)
- [旧细同时间图件](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/postprocess_t300/em_growth_comparison.png)
- [新图件源数据NPZ](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/postprocess_t300/em_growth_source_data.npz)
- [可信checkpoint](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/checkpoint.npz)
- [只读审计脚本](<ORIGINAL_CHECKOUT>/tmp/audit_em_unique_periodic_t300_20261006.py)

## 17. 2026-10-06 20:38：用户授权同hash从300 s续算3150 s

已归档300 s成果51文件至archive_t300，含checkpoint/final/report/诊断/全部快照/后处理与验收；SHA256清单保存，checkpoint复制逐字节一致，不覆盖archive_t60。启动前两次按命令行确认无EPB Python计算。唯一新driver初始PID21148，resume=True，从300 s/1200步接续至3150 s，同hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1。仅EM runner终止时间变化，Li配置完整保留。driver含终态hash/三轨数组/有限正密度审计、后处理和严格验收；完成后另需view_image及同3150 s粗/旧细比较。现在仅启动，不声称验收完成。旧f7、粗3150/5000 s成果未改，未启动修正5000 s或后续物理扫描。

不早于21:08再查进度，此后每30分钟至多一次；以checkpoint而非300 s旧final/report/status为真值。现有唯一driver持续运行时禁止重复启动。

[启动记录与日志](<ORIGINAL_CHECKOUT>/tmp/em_unique_periodic_t3150_launch_20261006.json)、[新续算driver](<ORIGINAL_CHECKOUT>/tmp/run_em_unique_periodic_t3150_20261006.py)、[300 s归档清单](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/archive_t300/archive_manifest.json)。
