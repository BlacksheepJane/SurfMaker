import os
import re
import numpy as np

root_dir = "/data6/jialin/SurfDock/docking_result/SurfDock_eval_samples/timesplit/SurfDock_docking_result"

top1_rmsds = []
top5_rmsds = []

def extract_rank_rmsd(filename):
    try:
        parts = filename.split("_")
        rank_idx = parts.index("rank") + 1
        rmsd_idx = parts.index("rmsd") + 1
        rank = int(parts[rank_idx])
        rmsd = float(parts[rmsd_idx])
        return rank, rmsd
    except Exception:
        return None, None

for target_dir in os.listdir(root_dir):
    target_path = os.path.join(root_dir, target_dir)
    if not os.path.isdir(target_path):
        continue

    rmsds = []

    for fname in os.listdir(target_path):
        if fname.endswith(".sdf"):
            rank, rmsd = extract_rank_rmsd(fname)
            if rank is not None:
                # print(f"Parsed: {fname} -> rank={rank}, rmsd={rmsd:.2f}")
                rmsds.append((rank, rmsd))

    if not rmsds:
        continue

    # 排序
    rmsds.sort(key=lambda x: x[0])  # 按 rank 升序

    # 提取 Top1 和 Top5 RMSDs
    top1_rmsd = [r[1] for r in rmsds if r[0] == 1]
    top5_rmsd = [r[1] for r in rmsds if r[0] <= 5]

    if top1_rmsd:
        top1_rmsds.append(top1_rmsd[0])
    if top5_rmsd:
        top5_rmsds.extend(top5_rmsd)

# 转换为 numpy array 便于统计
top1_rmsds = np.array(top1_rmsds)
top5_rmsds = np.array(top5_rmsds)

def print_stats(name, arr):
    print(f"== {name} ==")
    print(f"数量: {len(arr)}")
    print(f"RMSD < 1Å: {(arr < 1.0).sum() / len(arr):.3f}")
    print(f"RMSD < 2Å: {(arr < 2.0).sum() / len(arr):.3f}")
    print(f"RMSD 中位数: {np.median(arr):.3f}")
    print()

print_stats("Top-1", top1_rmsds)
print_stats("Top-5", top5_rmsds)
