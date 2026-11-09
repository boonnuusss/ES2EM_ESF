# E区终端电导映射审查（2026-10-10）

结论：现有代码已提供非负标量被动电导负载，不必新增重复参数。真实E区Pedersen电导到当前模态终端的映射尚未建立。A/B/C/D采用matched负载，表示当前均匀线的无反射终端；0.795774715459 S是Alfven导纳，不是实测E区电导。改变负载是可检验的下一假设，不能据此断言Y缺失原因。

## 文献依据与适用边界

[Lysak 2011, 10.1029/2010JA016424](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JA016424)，第10段，给出电场反射系数R=(SigmaA−SigmaP)/(SigmaA+SigmaP)，SigmaA=1/(mu0 VA)。这是特征电势/电场的符号约定，不能混用磁扰动的反射符号。

[Lysak 2020, 10.1029/2020JA027866](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JA027866)，式1及其后讨论，明确简单公式适用于垂直磁场、静电电场和均匀电导；Hall电导、倾角和平行电场需要更完整处理。其高纬数值案例不作为赤道夜间电导标定。

[Bhattacharyya 2004, 10.1029/2003GL018960](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003GL018960)，第4节，讨论E区电阻与F区极化共同产生的放电时间尺度、南北E区积分Pedersen电导。简化三模态模型不提供普适Y分叉判据；文中参数示例不是本项目的观测输入。

## 源码核对

实际源码在<ORIGINAL_CHECKOUT>/src/esf_sim/。em_growth.py的EMGrowthConfig已有right_boundary_model/conductance_s，验证有限非负电导及零外加电压，resolved_remote_conductance_s/remote_reflection_coefficient实现上述公式。_configuration_hash保护负载、长度和VA；历史fixed_voltage默认路径省略新增字段以保留旧hash。modal_alfven.py中s从Li平面指向磁层，w_plus=Phi+VA A沿+s传播，远端w_minus=R w_plus。当前一支均匀VA线并非完整两半球E区终端；column闭合中的hemisphere_count=2不能解释成两个独立负载。

已有tests/test_modal_alfven.py的被动性/匹配脉冲测试及tests/test_em_growth.py的配置、hash与恢复拒绝测试可供后续变更回归。本轮仅只读审查及解析配置计算，没有重复旧测试或积分生产轨迹。

## 解析核查

tmp/audit_eregion_terminal_mapping_20261010.py已运行通过：89源码文件SHA256与A/B保存清单一致，Bhash未变，6个等效负载配置hash互异，长度/VA改变均改变hash。单程L/VA=20 s。以下仅均匀特征边界解析能量比例，不是全系统RK-stage守恒验收。

| 等效负载S | 电势反射R | 反射能量R² | 吸收1−R² |
|---:|---:|---:|---:|
| 0 | 1 | 100% | 0% |
| 0.01 | 0.975179 | 95.0974% | 4.9026% |
| 0.1 | 0.776730 | 60.3309% | 39.6691% |
| 0.795774715459 | 0 | 0% | 100% |
| 1 | −0.113725 | 1.2933% | 98.7067% |
| 10 | −0.852577 | 72.6887% | 27.3113% |

这些数值是等效负载敏感性示例，不是新生产配置或实测E区范围。低电导和高电导都能强反射，反射相位不同；不能用“电导越低吸收越强”概括整个参数域。matched在此理想边界上吸收最大。

## 下一实施门禁

1. 区分两个研究层级：现有标量等效远端负载敏感性可用已有接口；若声称真实E区映射，需推导磁通管足点面积、横向梯度/电流度量与模态投影，说明单支或两半球边界、Hall与空间变化的简化，并提供电导来源。不把当前远端直接改名为已标定E区。
2. 下一源码工作优先默认0的三轨共同背景ExB。非零值须在localES/columnES/columnEM密度输运和CFL中一致出现；默认0保持旧hash/路径，非零显式入hash。检验方向、单位、常密度零导数、周期平移及三轨相同输运，采用有针对性的物理/回归测试。
3. 长度与VA已有物理hash保护。以后改变VA时，matched会同步改变SigmaA，而固定conductance_s不会；须明确控制的是绝对负载还是SigmaP/SigmaA，分别记录传播时间和反射，避免把多个因素混成单因素。
4. 新路径通过以上检查后再准备独立短阶段配置及验收；本轮未启动新轨迹。A/B/C/D输出、旧失败原判全部保留，不重启或自动延长A/B/C5000。Y原因仍未唯一确定。
