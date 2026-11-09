# B组300 s终态核验（2026-10-09）

B100×401、250 km唯一周期、FAC(8,8)152模态三轨完成300 s/1200步。独立检查checkpoint/final/快照密度一致、线变量与final一致，有限正密度、零宏步减半、hash保持。严格验收0失败2警告，view_image QA确认早期平滑三轨图件正常，无成熟期Y结论。archive_t300复制53文件全部SHA256一致、checkpoint逐字节一致，保留所有原输出。

A/B同网格同单线程：两ES密度逐字节一致；EM横向结构L2差0.1486987%，纯EM差场3.02901%。共同200–300 s、0.25 s等时logRMS插值OLS，gammaB3.2422139469299825e-4、A3.257300198434873e-4 s^-1，相对差−0.463152%。B/D同FAC同周期、共同节点EM结构差0.00935435%，gamma相对D差−0.0774122%；B单线程与D历史多线程仍混杂。早期一致不能证明成熟期/Y因果或严格收敛。

按授权从B自身300 s checkpoint同hash resume=True续3150 s，仅改EM runner终点；Li配置、三轨、反馈1、matched及单线程环境保持。3150终态仍须独立审计、严格验收、view_image QA、与A/D同3150比较及SHA256归档。不自动续5000，A/C/D不重跑，不移植checkpoint、不prepare/no-resume/手清尾部。
