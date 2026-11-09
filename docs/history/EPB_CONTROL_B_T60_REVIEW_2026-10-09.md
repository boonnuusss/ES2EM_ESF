# B组60 s终态核验（2026-10-09）

B100×401、250 km唯一周期、FAC(8,8)152模态三轨完成60 s/240步；独立数组检查确认checkpoint/final/快照密度一致、线变量与final一致，有限正密度、零宏步减半、hash保持。严格验收0失败2警告（EM CFL近上限与开放粒子通量账缺失）；view_image QA确认三轨早期平滑，图件正常，不作成熟期Y判断。

同网格、同单线程A/B早期比较：local ES和column ES密度逐字节一致；EM横向结构L2差0.0224005%。这支持当前零FAC对照的早期重复性，不能证明成熟期重复性、唯一Y原因或收敛。

archive_t60复制22文件并逐文件SHA256核验，checkpoint逐字节一致，保留原输出。已从B自身60 s checkpoint以同hash resume=True启动300 s唯一driver，仅修改EM runner终点，Li配置、单线程环境和三轨保持。300 s完成须独立终态审计/严格验收/view_image QA/归档门禁，通过后才续3150；不自动延长5000、不移植其他轨迹checkpoint。A/C/D不重跑。A/B与历史C/D线程混杂继续保留。
