# -*- coding: utf-8 -*-
"""数据集根路径解析（唯一入口）。

优先级：环境变量 AGM_DATA_ROOT > <项目根>/data 下的软链接。
本项目默认用软链接指向原始数据集，避免复制 ~19 GB：

    data/HUST      -> ~/FaultDia/HUSTbearing dataset
    data/PU        -> ~/FaultDia/PU Bearing
    data/PHM2009   -> ~/FaultDia/PHM2009

若要把数据集真正搬到别处，设置 AGM_DATA_ROOT 指向一个含同名子目录的目录即可。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_ROOT = os.environ.get('AGM_DATA_ROOT') or os.path.join(_ROOT, 'data')

# 三个基准各自的原始数据目录（保持发布时的内部层级）
HUST_DIR = os.path.join(DATA_ROOT, 'HUST')
PU_DIR = os.path.join(DATA_ROOT, 'PU', 'data')
PHM_DIR = os.path.join(
    DATA_ROOT, 'PHM2009', 'Supplementary-Dataset-main',
    'Supplementary-Dataset-main', 'PHM2009 Gearbox',
    'PHM_Society_2009_Competition_Expanded_txt')
# HUST 预处理缓存也放在外部数据目录；可用环境变量覆盖。
HUST_NPZ = os.environ.get(
    'AGM_HUST_NPZ', os.path.join(DATA_ROOT, 'HUST', 'hust_x1024.npz'))



# 可选第 4 个数据集（UO，来自 DIS 发行版）；默认不接入，需要时设 AGM_UO_DIR
UO_DIR = os.environ.get('AGM_UO_DIR', os.path.join(DATA_ROOT, 'UO'))


def describe():
    for k, v in [('DATA_ROOT', DATA_ROOT), ('HUST', HUST_DIR),
                 ('PU', PU_DIR), ('PHM2009', PHM_DIR),
                 ('UO(可选)', UO_DIR),
                 ('HUST_NPZ', HUST_NPZ)]:
        print('{:10s} {:<6s} {}'.format(k, 'OK' if os.path.exists(v)
                                        else 'MISS', v))


if __name__ == '__main__':
    describe()
