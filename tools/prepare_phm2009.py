#!/usr/bin/python
# -*- coding:utf-8 -*-
"""准备 PHM2009 Expanded_txt 数据集目录结构（针对 mingzhangPHD/Supplementary-Dataset 镜像）。

镜像里数据是 `spur N/spur N_工况.zip` 的形式，每个 zip 内是某个工况的信号 txt。
本脚本把所有 zip 解压成代码需要的结构:
    spur N/spur N_工况/spur N_工况.txt

用法:
    python prepare_phm2009.py [Expanded_txt 目录路径]

默认路径:
    ~/FaultDia/PHM2009/Supplementary-Dataset/PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt
"""
import os
import sys
import glob
import zipfile


def main():
    if len(sys.argv) > 1:
        base = sys.argv[1]
    else:
        base = os.path.expanduser(
            '~/FaultDia/PHM2009/Supplementary-Dataset/'
            'PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt')

    if not os.path.isdir(base):
        print('目录不存在: {}'.format(base))
        sys.exit(1)

    zips = sorted(glob.glob(os.path.join(base, 'spur *', '*.zip')))
    if not zips:
        print('在 {} 下没有找到 spur */ 目录或 *.zip 文件'.format(base))
        sys.exit(1)

    print('找到 {} 个 zip 文件'.format(len(zips)))
    for zp in zips:
        class_dir = os.path.dirname(zp)
        cond = os.path.splitext(os.path.basename(zp))[0]  # 例如 "spur 1_30hz_High_1"
        dest_dir = os.path.join(class_dir, cond)
        os.makedirs(dest_dir, exist_ok=True)

        with zipfile.ZipFile(zp) as z:
            txts = [n for n in z.namelist() if n.lower().endswith('.txt')]
            if not txts:
                print('  警告: {} 里没有 .txt'.format(zp))
                continue
            for idx, n in enumerate(txts):
                suffix = '' if len(txts) == 1 else '_{}'.format(idx + 1)
                out = os.path.join(dest_dir, cond + suffix + '.txt')
                with open(out, 'wb') as f:
                    f.write(z.read(n))
        print('  完成: {}'.format(cond))

    print('全部完成。可运行以下命令检查目录结构:')
    print('  find "{}" -name "*.txt" | head'.format(base))


if __name__ == '__main__':
    main()
