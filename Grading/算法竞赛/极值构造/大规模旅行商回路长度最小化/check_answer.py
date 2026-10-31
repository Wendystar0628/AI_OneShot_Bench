"""大规模旅行商回路长度最小化：校验 answer.md 并精确计算回路长度 L。

用法：python check_answer.py <answer.md 路径>
输出：JSON（UTF-8）。
退出码：0 有效；1 无效；2 用法错误或文件无法读取（未作判定）。
只用 Python 标准库，全程整数运算。城市坐标读题面旁的 points.txt。
"""
import json
import math
import os
import sys
from pathlib import Path

TASK = "大规模旅行商回路长度最小化"
N = 85900
OPTIMUM = 142382641
BOM = b"\xef\xbb\xbf"
POINTS = Path(r"E:\Project\AI_OneShot_Bench\算法竞赛\极值构造") / TASK / "points.txt"


class Invalid(Exception):
    pass


def load_points():
    pts = []
    for line in POINTS.read_text(encoding="ascii").splitlines():
        x, y = line.split()
        pts.append((int(x), int(y)))
    return pts


def parse(data):
    """返回从 0 起的排列；多余的空白、空行和从 0 起的编号都接受，不合格时抛出 Invalid。"""
    if data.startswith(BOM):
        data = data[len(BOM):]
    tokens = data.split()
    try:
        ids = [int(t) for t in tokens]
    except ValueError:
        bad = next(t for t in tokens if not t.lstrip(b"+-").isdigit())
        raise Invalid(f"读不出编号：{bad[:40]!r}")
    if len(ids) != N:
        raise Invalid(f"共 {len(ids)} 个编号，应为 {N}")
    if min(ids) != 0:
        ids = [i - 1 for i in ids]
    seen = bytearray(N)
    for i in ids:
        if not 0 <= i < N:
            raise Invalid(f"编号 {i + 1} 超出范围")
        if seen[i]:
            raise Invalid(f"编号 {i + 1} 重复出现")
        seen[i] = 1
    return ids


def tour_length(tour, pts):
    """L = Σ ⌈√(Δx²+Δy²)⌉，回路首尾相接。"""
    total = 0
    px, py = pts[tour[-1]]
    for i in tour:
        x, y = pts[i]
        s = (x - px) ** 2 + (y - py) ** 2
        d = math.isqrt(s)
        total += d if d * d == s else d + 1
        px, py = x, y
    return total


def check(path):
    res = {"task": TASK, "file": os.path.abspath(path),
           "valid": False, "invalid_reason": None,
           "length": None, "gap_percent": None}
    if not os.path.exists(path):
        res["invalid_reason"] = "文件不存在"
        return res
    if not os.path.isfile(path):
        res["invalid_reason"] = "路径不是普通文件"
        return res
    with open(path, "rb") as f:
        data = f.read()
    try:
        tour = parse(data)
    except Invalid as e:
        res["invalid_reason"] = str(e)
        return res
    length = tour_length(tour, load_points())
    gap = (length - OPTIMUM) * 10 ** 8 // OPTIMUM  # 百分比保留 6 位小数，向下取整
    res.update(valid=True, length=length, gap_percent=f"{gap / 10 ** 6:.6f}")
    return res


def main(argv):
    if len(argv) != 2:
        print("用法：python check_answer.py <answer.md 路径>", file=sys.stderr)
        return 2
    try:
        res = check(argv[1])
    except OSError as e:
        err = {"task": TASK, "file": os.path.abspath(argv[1]), "valid": None,
               "error": f"无法读取文件：{type(e).__name__}: {e.strerror or e}"}
        print(json.dumps(err, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty():
            stream.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv))
