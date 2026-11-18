"""AG-Mamba：本方法（受控混叠 Gabor 前端 + 标准选择性 SSM + 无对齐训练目标）。

子模块
------
* ``agmamba.method``  训练器与配置（原方法文件，命名已清理）
* ``agmamba.paths``   结果/权重目录规范（唯一路径入口）

注意：本文件**不**预先导入 ``method``，以免只用路径工具的分析脚本被迫加载
torch；需要训练器时显式 ``from agmamba import method``。
"""

__all__ = ['method', 'paths']
