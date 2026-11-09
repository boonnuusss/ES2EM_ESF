# conductance独立验收接口实现（2026-10-13）

仅em_growth_acceptance.py新增显式equivalent_passive_conductance_t60_v1；默认matched路径和结果保留。新分支要求请求/完成均60秒、conductance模型、有限非负S、有限正VA、零端电压、scope一致及全部诊断R符合解析公式，保留全部其它原数值与身份门槛。新报告必须显式独立路径，拒绝覆盖已有文件和原验收常用文件名。原阈值不变，未改模拟代码、hash或数组。

20项针对性检查通过，包括默认matched与修改前完整结果一致、ratio0p2默认失败保留、新profile通过、未知profile/覆盖拒绝、负/NaN电导、VA/端电压/模型/R/power错配，以及完整读取路径的负远端功率/CFL/残差、物理hash篡改和status跨文件身份拒绝。测试中发现的分支局部变量顺序和辅助函数签名问题已修正，最终完整检查通过。临时测试副本从未用于生产或恢复。

ratio0p2独立报告另存docs/EPB_LOAD_RATIO0P2_T60_CONDUCTANCE_ACCEPTANCE_2026-10-13.json，0失败2警告；原输出acceptance_t60.json仍1失败2警告，五个受保护输入SHA一致。独立通过仅表示等效被动负载60秒门禁，非原matched通过、真实E区标定或成熟Y证据。89源码新v4快照仅acceptance相对v3变化，完整备份/ZIP已留存。

本轮未归档、未启动ratio5、未重积分。下一将独立策略报告、原失败报告和审计共同纳入SHA归档清单，并更新ratio5 driver明确策略/v4源码溯源与单线程/hash/完整Li门禁；确认串行及新目录后才启动ratio5仅60秒。不延长任何完成轨迹。
