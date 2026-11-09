# 冻结密度ES求解重复性诊断（2026-10-09）

采用同一C组2500 s归档column ES密度，分别在FAC4/8模型、MKL线程1/10四个独立进程中各求解3次零FAC RHS；未积分、未改checkpoint或生产输出。

{
  "C_threads1_vs_D_threads1": {
    "array_equal": true,
    "rhs_relative_l2": 0.0
  },
  "C_threads10_vs_D_threads10": {
    "array_equal": false,
    "rhs_relative_l2": 4.149274290569737e-15
  },
  "C_threads10_vs_C_threads1": {
    "array_equal": false,
    "rhs_relative_l2": 2.7696242176207197e-14
  },
  "D_threads10_vs_D_threads1": {
    "array_equal": false,
    "rhs_relative_l2": 2.75993149922352e-14
  }
}

单线程C/D RHS逐字节一致，重复调用一致。10线程重复调用和跨模型/进程差异约浮点舍入尺度，电势残差约5e-13。这确认多线程求解可提供微小数值扰动，尚不能证明历史14.94%结构差来自它；冻结密度与新模型未复现整个轨迹或求解器缓存历史。当前零FAC RHS未显示FAC截断造成可分辨物理差异。

后续新诊断和A/B运行须记录线程/runtime、源码SHA256及初值SHA256；已有结果不重写，不靠修改阈值消除差异。先定义可重复的数值对照，再解释成熟期差场；尚无Y改进结果。
