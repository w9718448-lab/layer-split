#!/usr/bin/env python3
"""把拆好的元素合成分层 PSD。

图层结构（从上到下）：
  完整预览   原图，放在最上面
  元素/      文件夹，每个元素一个透明底的普通像素图层，按遮挡关系叠放（挡在前面的在上面）
  背景       去掉所有元素后的底（complete.py 生成的 background.png；没有时用原图代替）

用法
  python build_psd.py 原图.png --run 拆分输出目录 [--out 输出.psd] [--scale 2] [--no-base]
  --scale  画布相对原图的倍数，默认等于元素的放大倍数（保持元素清晰度）
需要 psd-tools：pip install --break-system-packages psd-tools
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps


def order_layers(items, vis_masks, full_masks):
    """返回从下到上的 id 列表。
    vis_masks：补全前元素的可见掩码（原图坐标）；full_masks：补全后的掩码。
    A 补全出来的区域与 B 的可见区域重叠 -> B 挡着 A -> B 在 A 上面。"""
    ids = [it["id"] for it in items]
    bottom = {it["id"]: it["bbox"][3] for it in items}
    above = {i: set() for i in ids}        # above[a] = 必须在 a 上面的图层
    for a in ids:
        fa = full_masks[a]
        if fa is None:
            continue
        extra = fa & ~vis_masks[a]          # 补全新增的区域
        if not extra.any():
            continue
        for b in ids:
            if b == a:
                continue
            if (extra & vis_masks[b]).sum() > 20:
                above[a].add(b)
    # 拓扑排序，同级按底边 y 从小到大（靠上的先放，靠下的盖在上面）
    indeg = {i: 0 for i in ids}
    for a, bs in above.items():
        for b in bs:
            indeg[b] += 0
    rest = set(ids)
    out = []
    while rest:
        # 可以放下一层的 = 不需要任何“尚未放置的层”在它下面
        under = {i: [a for a in rest if i in above[a]] for i in rest}   # i 必须在这些层上面
        ready = [i for i in rest if not under[i]]
        if not ready:                        # 有环，按位置强行打破
            ready = [min(rest, key=lambda i: bottom[i])]
        nxt = min(ready, key=lambda i: (bottom[i], i))
        out.append(nxt)
        rest.discard(nxt)
    return out


def canvas_mask(path, bbox, W, H):
    if not os.path.exists(path):
        return None
    el = Image.open(path).convert("RGBA")
    x0, y0, x1, y1 = bbox
    a = el.split()[3].resize((max(x1 - x0, 1), max(y1 - y0, 1)), Image.BICUBIC)
    m = np.zeros((H, W), bool)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x1, W), min(y1, H)
    arr = np.asarray(a) > 127
    m[ya:yb, xa:xb] = arr[ya - y0:yb - y0, xa - x0:xb - x0]
    return m


def main():
    ap = argparse.ArgumentParser(description="元素合成分层 PSD")
    ap.add_argument("image")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    ap.add_argument("--scale", type=float)
    ap.add_argument("--no-base", action="store_true", help="不放最上层的完整预览")
    args = ap.parse_args()

    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    W, H = img.size
    man = json.load(open(os.path.join(args.run, "manifest.json"), encoding="utf-8"))
    items = [it for it in man["items"] if os.path.exists(os.path.join(args.run, it["file"]))]

    fs = [it["size"][0] / max(it["bbox"][2] - it["bbox"][0], 1) for it in items]
    scale = args.scale or max(1.0, float(np.median(fs)))
    CW, CH = int(round(W * scale)), int(round(H * scale))

    # 遮挡顺序
    orig_dir = os.path.join(args.run, "elements_original")
    vis, full = {}, {}
    for it in items:
        fn = os.path.basename(it["file"])
        po = os.path.join(orig_dir, fn)
        pf = os.path.join(args.run, it["file"])
        # 补全前的 bbox 未保存，用原目录元素的尺寸按同一缩放推回（补全只会向外扩）
        full[it["id"]] = canvas_mask(pf, it["bbox"], W, H)
        if os.path.exists(po) and it.get("completed"):
            e = Image.open(po)
            # 原元素在原图中的位置：manifest 里记录的是补全后的 bbox，这里用 original_bbox
            ob = it.get("original_bbox") or it["bbox"]
            vis[it["id"]] = canvas_mask(po, ob, W, H)
        else:
            vis[it["id"]] = full[it["id"]]
    order = order_layers(items, vis, full)
    by_id = {it["id"]: it for it in items}

    from psd_tools.api.layers import Group

    psd = PSDImage.new("RGB", (CW, CH))

    def add(parent, name, im, top=0, left=0):
        layer = PixelLayer.frompil(im, psd, "layer", top=top, left=left)
        layer.name = name                    # 经 setter 写入，中文名走 Unicode 图层名
        parent.append(layer)
        return layer

    # 1. 最底层：背景（去掉所有元素后的底），没有补好的背景时用原图代替
    bg_path = os.path.join(args.run, "background.png")
    bg = Image.open(bg_path).convert("RGB") if os.path.exists(bg_path) else img
    add(psd, "背景", bg.resize((CW, CH), Image.LANCZOS).convert("RGBA"))

    # 2. 中间：元素文件夹，每个元素一个普通像素图层，按遮挡关系从下到上
    grp = Group.new(psd, name="group", open_folder=True)
    grp.name = "元素"
    if grp not in list(psd):
        psd.append(grp)
    for i in order:
        it = by_id[i]
        el = Image.open(os.path.join(args.run, it["file"])).convert("RGBA")
        x0, y0, x1, y1 = it["bbox"]
        tw, th = int(round((x1 - x0) * scale)), int(round((y1 - y0) * scale))
        if (tw, th) != el.size and tw > 0 and th > 0:
            el = el.resize((tw, th), Image.LANCZOS)
        add(grp, f"{it['id']:03d}_{it['name']}", el, int(round(y0 * scale)), int(round(x0 * scale)))

    # 3. 最上层：完整图预览
    if not args.no_base:
        add(psd, "完整预览", img.resize((CW, CH), Image.LANCZOS).convert("RGBA"))

    out = args.out or args.run.rstrip("/\\") + ".psd"
    psd.save(out)
    print(json.dumps({"psd": out, "canvas": [CW, CH], "elements": len(order),
                      "background": os.path.exists(bg_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
