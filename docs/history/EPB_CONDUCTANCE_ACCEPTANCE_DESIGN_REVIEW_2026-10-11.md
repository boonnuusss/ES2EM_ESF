# 独立conductance验收策略设计（2026-10-11）

现有验收同时包含matched_remote_boundary硬条件和passive_remote_absorption功率条件。ratio0p2的唯一失败来自前者；后者已有逐条非负功率和积分门禁。因此不能把matched条件改成自动允许任意边界，或仅删掉失败项。原生产策略及原acceptance_t60.json必须完整保留。

设计独立显式策略，例如equivalent_passive_conductance_t60_v1，仅用于预先声明的60秒、零外加端电压、conductance三轨敏感性实验。默认仍为原matched策略；未知策略和错配边界必须失败。新报告另存，明确策略名、版本、适用阶段、输入SHA、数值阈值与原matched判定；不覆盖原报告，也不把新策略通过解释为原matched验收通过。

新边界门禁要求有限非负S、有限正VA、零外加端电压；配置与scope的S/model一致；用SigmaA=1/(mu0 VA)计算R，并核对scope和每条诊断的R（沿用1e-12反射一致容差），|R|≤1。原配置hash重构、status/checkpoint/snapshot跨文件身份核验、有限正密度、三轨结构、原CFL/残差/配对/零宏步减半、反馈增益及阶段检查均须保留。新边界条件只代替新策略中的matched条件，不能过滤其它失败项。

保留现有passive_remote_absorption的逐条及积分判据，功率/能量容差仍1e-10相对尺度。负功率、NaN/Inf、缺诊断都须失败。代表stage采样不能证明精确RK-stage全局能量，端负载是等效敏感性参数，非真实E区标定。

本轮只读原型验证了边界子判据：ratio0p2输入通过，12个内存故障样本被拒绝，包括负/非有限S、scope S错配、scope或诊断R错配、非有限R、边界注能、非有限功率、非零端电压、错误模型、错误身份及缺历史。原型不执行完整hash重构或跨文件校验，不能单独充当生产验收。5个原输出输入SHA及89源码v3 SHA保持一致，无重积分、新生产、原报告修改或归档。

下一实施门禁：独立显式profile接口、默认matched报告回归一致、未知profile拒绝、完整hash篡改/跨文件身份/原数值门槛故障测试，并验证反射profile保留其它失败项。独立profile及测试通过之后才可单独生成新策略报告；原1失败2警告永远保留。ratio5 driver必须先采用明确策略并重新核查源码溯源，不能继续使用只支持matched的旧完成验收路径。当前ratio5未启动，ratio0p2未归档，所有完成轨迹保护。
