# 细网格3150 s：完成、严格验收失败与同时间比较

2026-10-06，Asia/Shanghai。实际源码工作树<ORIGINAL_CHECKOUT>；本次只恢复既有1050 s checkpoint，未重跑粗网格3150/5000 s，该生产续算未改源码、物理或配置；之后的默认关闭数值修正见下文。

## 完成状态与门禁

细网格201×401、FAC(8,8)/152模态三轨到3150 s/12600步。schema3 checkpoint，hash仍为f7aecc67b3ddaae0c9c6325d0ef6fee2b6084060db301c8c4df5d6b9b52ad055。driver终态审计通过：checkpoint/final/末快照密度逐元素相同，EM特征变量checkpoint/final相同。316帧覆盖0–3150 s，诊断3150条、严格递增且到终点。进程已结束，stderr为空；没有另启算例。

**严格acceptance未通过：36项检查、2失败、4警告。** 两项失败是local ES、column ES周期重复端点相对差分别3.60870774e-5、1.30765842e-4，超1e-6门限；column EM仅5.71457693e-15。警告为两ES CFL接近上限、EM combined CFL接近上限和开放边界粒子账未闭合。不能用status completed替代验收通过。

宏观dt=0.25 s、零宏观减步；ES自动subcycling，末期四子步dt=0.0625 s，不是连续拒步或宏观减步。最大local/column ES CFL=0.249838238/0.249735641，EM combined CFL=0.4；最大耦合残差5.06417e-12，功率配对缺陷9.89556e-16。浮点数组有限，三轨正密度。FAC/重力有效期中位数0.240191，终点0.266421；反射R=0、反馈=1，表面FAC未改符号。

线能量125777.93465 J，采样净边界功125757.95869 J，差19.97596 J，相对总交换生能比例2.19017e-6，低于5e-3门限。远端吸收9120750.87111 J。预算是代表stage采样功率梯形积分，非精确离散stage功；开放垂直边界的粒子列漂移不能当封闭守恒误差。

## 新数值失败根因证据与待修复工作

配置nx=201、dx=1.25 km、x跨度250 km、包含x=0与x=250重复物理点；acceptance及形态统计按200个唯一x点处理。实际mimetic势算子、速度差分和输运却在全部201列上np.roll，构成201个独立周期格，重复首尾没有一致的相邻 stencil。该路径不能保证n[:,0]=n[:,-1]是不变量。代码位置：src/esf_sim/mimetic_potential.py的build_harmonic_equal_extrapolated_matrix/solve、src/esf_sim/_model_impl.py的velocity_from_potential/evaluate_rhs、src/esf_sim/_numerics_impl.py的flux_divergence，以及column ES的bidirectional_em._density_advection_rhs。

本次用既有PDM7算子进行小型确定性复现（不推进生产状态）：q每行为[2,3,5,4,2]、vx=1、vy=0、dx=dy=1；首尾初值严格相同，但RHS首列0、末列0.8666667，证明重复点等式不是该算子不变量。细网格存档快照首次超1e-6门限：column ES2600 s、local ES2810 s，EM没有超限。具体哪个非线性扰动使误差放大仍需补充RHS/势算子级审计；不应仅归因为subcycling。

粗网格使用相同历史网格语义；粗网格旧验收通过不消除该算子一致性风险，原成果继续保留，不重跑不覆盖。**截至08:48尚未修复源码；09:18跟进已补充默认关闭修正与39项针对性测试，详见下文**，没有补缝终态数组，没有放宽验收阈值，没有改hash。下一工作是明确周期网格语义、做针对性回归与版本保护；任何改变离散轨迹的修复必须有显式数值版本/hash隔离，不能在旧f7 hash下静默续算。细网格5000 s及后续物理扫描暂不启动，直到该数值问题有可审计解决方案。

## 同3150 s形态与增长率（带失败限定的探索性比较）

粗/细图件已用view_image直接检查。ES细网格有更多分支和次级结构；两个EM都为宽平滑耗空通道，无清晰Li式Y分叉。不能把连通域计数当Y分叉数量。

300–3150 s、归档每个宏观时间步等权OLS拟合ln RMS（每轨11401样本，dt=0.25 s；ES子步不加密宏观历史采样）：

| track | 粗gamma(s^-1) | 细gamma(s^-1) | 细/粗变化 |
|---|---:|---:|---:|
| local ES | 6.4390725e-4 | 6.4365243e-4 | -0.0396% |
| column ES | 6.6951358e-4 | 6.7054395e-4 | +0.1539% |
| column EM | 4.1591322e-4 | 4.0989178e-4 | -1.4478% |

EM相对同几何column ES抑制粗37.8783%、细38.8718%。纯EM用column EM-column ES，几何用column ES-local ES。历史RMS包含归档重复末列；空间差异另按共同唯一节点计算，两个定义明确区分。

细网格取[::2,::2]映射到粗共同节点、双方去掉重复末列：EM总密度相对L2差0.8032%，横向扰动差2.8841%；local/column ES总密度差24.8300%/15.0134%，横向扰动差61.1799%/33.8156%。几何效应差84.3880%、纯EM效应差44.2958%，反映非线性ES结构敏感性。D>=0.25主连通域EM顶粗454 km、细455 km；local顶480/501 km、column ES顶482/484 km。仅两个分辨率且同步改变FAC截断，不能证明严格收敛，不能单独归因网格或模态。细ES端点失败进一步限制解释。

同3150 s采样功率积分：粗/细Li输入9288967.2894/9246508.8298 J，远端吸收9161758.3740/9120750.8711 J；终点FAC/重力0.262414/0.266421。粗3150 s的线特征变量不在当前5000 s checkpoint内，本轮没有重建其终态线能量，不声称完整粗3150能量闭合对照。

## 既有物理修复与未解决限制

沿用已建立失败根因：固定远端电势R=-1短路反射；FAC体源错误1/ds；单一H混淆粒子列与电导列。生产为matched R=0、物理表面FAC和独立H_n/H_nu，推进N=H_n*n：

H_n=∫(B_a/B)(n(s)/n_a)ds；H_nu=∫(B_a/B)(n(s)nu(s)/(n_a nu_a))ds；
div[H_nu*n*nu_a*grad(psi)]=-g*d_x(H_n*n)+(B_a/m_i)*J_surface。

没有翻转FAC、短路捷径或弱反馈作弊。模型仍缺平行质量输运、完整双半球metric、化学、中性风/剪切、PRE、动态E层电导，不是定量预测。E-region电导扫描、背景漂移默认零三轨实现/0、10、20 m/s扫描、长度/VA扫描均未实施。先解决当前数值验收失败，然后按用户顺序推进。

## 可复核证据

- [同时间对照JSON与端点历史](<ORIGINAL_CHECKOUT>/docs/EPB_RESOLUTION_COMPARISON_T3150_2026-10-06.json)
- [本次终态检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0848.json)
- [细网格严格验收（失败）](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/acceptance_t3150.json)
- [细网格终态一致性审计](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/terminal_audit_t3150.json)
- [细网格3150 s PNG](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/postprocess_t3150/em_growth_comparison.png)
- [粗网格3150 s PNG](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t3150/em_growth_comparison.png)
- [细网格metrics JSON](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/postprocess_t3150/em_growth_metrics.json)
- [细网格源数据NPZ](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/postprocess_t3150/em_growth_source_data.npz)
- [只读审计与小型算子复现脚本](<ORIGINAL_CHECKOUT>/tmp/audit_em_resolution_t3150_20261006.py)

## 09:18跟进：显式唯一周期格修正（尚未生产验证）

进一步审计确认FAC ModalBasis同样把全部存储列当独立周期格，离散波数为2πk/nx、周期nx·dx。因此历史细/粗实际算子周期为251.25/252.5 km，而种子波长为250 km。问题是全套周期离散、坐标与验收语义冲突，不只ES输运或subcycling。旧结果仍按原验收判失败，不放宽门限。

已在实际C盘源码增加periodic_x_layout，默认legacy_stored_cells并在hash序列化时省略默认字段；两个旧生产hash逐字保持。显式unique_periodic_cells进入hash，要求坐标跨度为(nx−1)dx、计算周期nx·dx整除种子波长，验收和图像处理保留全部独立列。新细网格存200列、dx=1.25 km、坐标0至248.75 km，计算周期250 km；三轨、势和FAC继续使用相同独立列。物理方程、R=0、反馈1和FAC符号未改。

新增测试覆盖旧hash、新hash拒绝旧checkpoint、非法布局和不闭合种子、PDM周期平移等变、唯一列不被误删除、0.02 s三轨正密度/有限数组及完成后幂等resume。相关runner/acceptance/postprocess测试共39项通过（7.52 s）。未重跑已完成生产算例或143项整套回归。原dirty文件修改前完整副本保存在E盘tmp/periodic_grid_patch_backup_20261006。

已准备独立200×401/FAC(8,8)三轨60 s配置，但未启动。它改变离散轨迹，须使用新hash与独立输出，不能将旧f7 checkpoint裁剪或继续作为修正轨迹。小型测试不代表成熟期稳定性或收敛。根据用户保持f7与不重复生产算例的约束，启动独立修正算例前需要明确选择；原3150 s输出全部保留。5000 s延伸及后三项物理扫描仍未实施。

- [修正审计与新配置hash](<ORIGINAL_CHECKOUT>/docs/EPB_PERIODIC_GRID_CORRECTION_2026-10-06.json)
- [39项测试日志](<ORIGINAL_CHECKOUT>/tmp/periodic_grid_focused_tests_20261006.log)
- [独立60 s EM配置（未启动）](<ORIGINAL_CHECKOUT>/tmp/em_growth_unique_periodic_200x401_k8_t60_20261006.json)
- [唯一周期格Li配置](<ORIGINAL_CHECKOUT>/tmp/em_growth_li_unique_periodic_200x401_20261006.json)
- [永久回归测试](<ORIGINAL_CHECKOUT>/tests/test_em_periodic_grid_layout.py)

## 13. 2026-10-06 14:10：用户授权并启动独立唯一周期格60 s验证

用户明确回复“启动”。按命令行再次核查，无活跃EPB Python计算；以新driver、resume=True启动独立200×401/FAC(8,8)152模态三轨，初始PID23740。新hash为28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1，唯一200列覆盖250 km周期。实际源码与新输出仍在C盘工作树；原f7的3150 s输出/checkpoint/hash保留，没有重启旧driver、prepare或重复粗网格算例。新driver自带60 s终态数组/hash审计、后处理和严格验收。启动不等于完成或验收通过；完成后仍需view_image QA。仅60 s独立验证获授权，未启动300/3150/5000 s延伸和新物理扫描。

[启动记录与日志路径](<ORIGINAL_CHECKOUT>/tmp/em_unique_periodic_t60_launch_20261006.json)、[新driver](<ORIGINAL_CHECKOUT>/tmp/run_em_unique_periodic_t60_20261006.py)。不早于14:40再做进度检查，此后每30分钟至多一次，不高频轮询。


## 14. 2026-10-06 14:48：独立唯一周期格60 s验收通过、视觉QA完成

按命令行查无Python进程，driver正常完成且stderr为空。可信schema3 checkpoint为60 s/240步，hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1；终态checkpoint/final/末快照密度逐元素一致，线变量checkpoint/final一致，所有浮点数组有限、三轨正密度。dt=0.25 s，零宏观减步；60条诊断严格递增到60 s。唯一200列计算周期250 km，不删除末列、不要求两个不同周期格首尾相等；历史f7验收失败保持原判。

严格60 s门禁通过，0失败、2警告（EM combined CFL=0.4接近设定上限；开放垂直边界未记录通量以闭合粒子账）。ES CFL最大local0.0160429、column0.00266135；最大耦合残差6.72059e-13、功率配对缺陷5.93530e-16、表面电流平衡比7.73666e-17。FAC/重力有效期中位数0.242602。R=0和feedback=1保持。

终态线能量36509.49783 J；采样净边界功36514.38699 J，功减能4.88916 J，相对差9.37869e-5、生能比例0；远端吸收52130.49922 J。此为代表stage功率梯形积分，非精确RK-stage功；开放边界粒子列漂移不等于封闭守恒误差。

0–60 s每0.25 s等权OLS拟合ln RMS，241样本：local ES gamma=-4.24420e-4、column ES=4.03876e-4、column EM=3.12259e-4 s^-1，含20 s feedback ramp，因此只是早期窗。EM相对column ES增长率降低约22.68%。图件已view_image：三轨均早期平滑扰动，无D>=0.25泡或Y分叉；渲染无明显接缝/缺陷。60 s成功不能证明旧晚期端点风险完全消除或成熟期收敛。

下一建议：先保留/归档60 s成果，再用同新hash从现有checkpoint延伸300 s；当前未启动延伸。旧3150/5000 s均未重跑。后续物理扫描未实施。

[本次审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T60_AUDIT_2026-10-06.json)、[严格验收](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/acceptance_t60.json)、[图件](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/postprocess_t60/em_growth_comparison.png)、[源数据](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/postprocess_t60/em_growth_source_data.npz)。

## 15. 2026-10-06 15:17：用户授权同hash从60 s续算300 s

启动前命令行核查无EPB Python进程。已复制保护20个60 s成果文件至原输出下archive_t60（包括checkpoint、final、report、诊断、全部快照和后处理）；SHA256清单已保存，checkpoint副本逐字节相同，不删除或移动原成果。用户明确授权从现有checkpoint同hash续算300 s，唯一新driver初始PID16656，resume=True。同hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1已核对；只修改EM runner的t_end_s，Li配置完整保留（其t_end_s字段参与hash，EM runner以EM配置终点为准）。先前尝试同步修改Li终点被hash断言拦截，未启动计算或改checkpoint；已改为保持Li配置的正确同hash路径。

新driver包含300 s终态数组/hash/有限正密度审计、后处理和严格验收。下一进度检查不早于15:47，此后每30分钟至多一次。当前只是启动，尚不能宣称300 s验收通过。旧f7和粗3150/5000 s成果未改，未启动新3150/5000 s或后三项物理扫描。

[启动与日志记录](<ORIGINAL_CHECKOUT>/tmp/em_unique_periodic_t300_launch_20261006.json)、[续算driver](<ORIGINAL_CHECKOUT>/tmp/run_em_unique_periodic_t300_20261006.py)、[60 s归档SHA256清单](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/archive_t60/archive_manifest.json)。

## 16. 2026-10-06 16:48：修正三轨300 s验收通过，视觉QA与旧存档对照完成

现有checkpoint同hash续算完成300 s/1200步，schema3、新hash保持；driver退出且stderr空，按命令行无Python进程。终态一致性、有限数组、正密度、零减步、残差/功率配对及采样能量门禁通过。严格验收0失败2警告（EM CFL上限、开放粒子通量预算未闭合）。新与旧细同300 s图件均view_image QA，EM无D>=0.25泡、无Y。200–300 s等权401样本拟合：修正EM gamma3.24473e-4 s^-1，对旧细变化−0.0453%；横向结构L2差0.013762%。不同计算周期/FAC截断及早期证据不能证明成熟期或严格收敛。建议归档300 s后同hash续算3150 s；当前尚未启动，后续物理扫描未实施。

[完整报告与图件/源数据](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T300_REVIEW_2026-10-06.md)、[审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T300_AUDIT_2026-10-06.json)。旧f7验收失败保持原判，所有旧成果保留。
