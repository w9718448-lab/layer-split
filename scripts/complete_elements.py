#!/usr/bin/env python3
"""补全被遮挡的元素（需要 torch 和 LaMa 修复模型）。

split_layers.py 拆出来的元素只有原图里可见的部分。被前面的物体挡住时会带缺口。
这个脚本对指定元素：
  1. 把元素以外的画面全部涂成页面白色，只保留元素自己的像素
  2. 在 fills.json 指定的遮挡区域里，用 LaMa 按周围内容把元素缺的那部分补出来
  3. 补完之后取非白色部分作为新的完整轮廓，再按原来的倍数超分
补全是模型按上下文推测出来的，不是原图里真实存在的内容：缺口小而规整时（栏杆桩、围巾、细柱子挡住的碗、柜子）
效果可以用；缺的是一大块时只会得到糊状的猜测，建议改用外部图像生成工具。

用法
  python complete_elements.py 原图.png --run 拆分输出目录 --fills fills.json [--out 输出目录]

fills.json（坐标是原图像素，name 要和 manifest.json 里的名字一致）
  [{"name": "五斗柜", "fill": [[740, 822, 800, 935]]},
   {"name": "面粉碗", "fill": [[466, 620, 486, 672], [428, 616, 446, 650]]}]
  fill 是“被挡住、需要补出来”的矩形区域，可以写多个，框住遮挡物就行，不要框到别的元素。

输出：<out>/elements/ 里是补全后的元素（文件名和原来相同），<out>/compare.png 是补全前后对比。
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_layers as sl  # noqa: E402

LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
_lama = None


def get_lama():
    global _lama
    if _lama is not None:
        return _lama
    import torch
    path = None
    for d in (os.environ.get("LAYER_SPLIT_MODELS", ""), "/home/claude/models",
              os.path.expanduser("~/.u2net")):
        if d and os.path.exists(os.path.join(d, "big-lama.pt")):
            path = os.path.join(d, "big-lama.pt")
            break
    if path is None:
        d = os.path.expanduser("~/.u2net")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "big-lama.pt")
        print("下载 LaMa 修复模型 ...", file=sys.stderr)
        import urllib.request
        urllib.request.urlretrieve(LAMA_URL, path)
    _lama = torch.jit.load(path, map_location="cpu").eval()
    return _lama


def lama_fill(rgb, hole, target=512):
    """rgb: HxWx3 uint8；hole: HxW bool。返回修复后的 HxWx3 uint8。"""
    import torch
    h, w = hole.shape
    up = float(np.clip(target / max(h, w), 1.0, 6.0))
    nh, nw = int(round(h * up)), int(round(w * up))
    im = Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC)
    mk = Image.fromarray((hole * 255).astype(np.uint8)).resize((nw, nh), Image.NEAREST)
    ph, pw = (8 - nh % 8) % 8, (8 - nw % 8) % 8
    a = np.pad(np.asarray(im, np.float32) / 255.0, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m = np.pad((np.asarray(mk) > 127).astype(np.float32), ((0, ph), (0, pw)), mode="reflect")
    with torch.inference_mode():
        out = get_lama()(torch.from_numpy(a.transpose(2, 0, 1))[None],
                         torch.from_numpy(m)[None, None])
    out = np.clip(out[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)[:nh, :nw]
    return np.asarray(Image.fromarray(out).resize((w, h), Image.LANCZOS))


def complete_one(img, item, rects, scale_hint=None):
    """返回 (补全后的 RGBA（原图 1 倍尺寸）, 在原图中的 bbox)。"""
    W, H = img.size
    bx0, by0, bx1, by1 = item["bbox"]
    el = Image.open(item["_path"]).convert("RGBA")
    f = el.width / max(bx1 - bx0, 1)                       # 元素当前相对原图的放大倍数
    alpha1 = el.split()[3].resize((bx1 - bx0, by1 - by0), Image.BICUBIC)
    M = np.zeros((H, W), bool)
    M[by0:by1, bx0:bx1] = np.asarray(alpha1) > 127

    xs = [bx0, bx1] + [v for r in rects for v in (r[0], r[2])]
    ys = [by0, by1] + [v for r in rects for v in (r[1], r[3])]
    pad = max(int(0.2 * max(max(xs) - min(xs), max(ys) - min(ys))), 12)
    cx0, cy0 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    cx1, cy1 = min(max(xs) + pad, W), min(max(ys) + pad, H)

    rgb = np.asarray(img.convert("RGB"))[cy0:cy1, cx0:cx1].copy()
    m = M[cy0:cy1, cx0:cx1]
    keep = ndi.binary_dilation(m, iterations=2)
    ctx = np.where(keep[..., None], rgb, 255).astype(np.uint8)   # 元素以外全部涂成白色

    hole = np.zeros(m.shape, bool)
    for x0, y0, x1, y1 in rects:
        hole[max(y0 - cy0, 0):max(y1 - cy0, 0), max(x0 - cx0, 0):max(x1 - cx0, 0)] = True
    hole &= ~keep
    hole = ndi.binary_dilation(hole, iterations=2) & ~m
    if not hole.any():
        return None
    filled = lama_fill(ctx, hole)

    # 补出来的新内容里，非白色的部分并入元素轮廓
    nonwhite = filled.min(axis=2) < 232
    add = hole & nonwhite
    add = ndi.binary_opening(add, iterations=1)
    newm = m | add
    newm = ndi.binary_closing(newm, iterations=2)
    newm = ndi.binary_fill_holes(newm)
    newm = sl._main_parts(newm)

    out_rgb = np.where(hole[..., None] & ~m[..., None], filled, rgb).astype(np.uint8)
    out_rgb = np.where(m[..., None], rgb, out_rgb).astype(np.uint8)
    sm = np.asarray(Image.fromarray((newm * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.2))) > 127
    alpha = np.asarray(Image.fromarray((sm * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)))
    rgba = Image.fromarray(out_rgb).convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    res = sl.crop_to_alpha(rgba)
    if res is None:
        return None
    e, (ox, oy) = res
    return e, (cx0 + ox, cy0 + oy, cx0 + ox + e.width, cy0 + oy + e.height), f


def main():
    ap = argparse.ArgumentParser(description="补全被遮挡的元素（LaMa）")
    ap.add_argument("image")
    ap.add_argument("--run", required=True, help="split_layers.py 的输出目录（里面有 manifest.json）")
    ap.add_argument("--fills", required=True)
    ap.add_argument("--out", help="输出目录，默认 <run>_completed")
    args = ap.parse_args()

    img = ImageOps.exif_transpose(Image.open(args.image))
    man = json.load(open(os.path.join(args.run, "manifest.json"), encoding="utf-8"))
    by_name = {}
    for it in man["items"]:
        it["_path"] = os.path.join(args.run, it.get("file") or it["transparent"])
        by_name[it["name"]] = it
    fills = json.load(open(args.fills, encoding="utf-8"))
    out = args.out or args.run.rstrip("/\\") + "_completed"
    os.makedirs(os.path.join(out, "elements"), exist_ok=True)

    done, f_list = [], []
    for fl in fills:
        it = by_name.get(fl["name"])
        if it is None:
            print(f"找不到元素 {fl['name']}，跳过", file=sys.stderr)
            continue
        r = complete_one(img, it, [tuple(int(v) for v in x) for x in fl["fill"]])
        if r is None:
            print(f"{fl['name']}：没有可补的区域，跳过", file=sys.stderr)
            continue
        e, bbox, f = r
        done.append((it, e, bbox))
        f_list.append(f)
        print(f"  补全 {fl['name']}", file=sys.stderr, flush=True)

    if done:
        up = int(round(f_list[0]))
        if up > 1:
            ups, method = sl.superres([d[1] for d in done], up)
        else:
            ups, method = [d[1] for d in done], "none"
        before, after = [], []
        for (it, e1, bbox), e2 in zip(done, ups):
            fn = os.path.basename(it["_path"])
            e2.save(os.path.join(out, "elements", fn))
            before.append((it["name"] + " 补全前", Image.open(it["_path"]).convert("RGBA")))
            after.append((it["name"] + " 补全后", e2))
            it["bbox"] = [int(v) for v in bbox]
            it["size"] = list(e2.size)
        pairs = []
        for b, a in zip(before, after):
            pairs += [b, a]
        sl.contact_sheet(pairs, os.path.join(out, "compare.png"), cell=220, cols=4)
        print(json.dumps({"completed": [d[0]["name"] for d in done], "upscale": method,
                          "out": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
