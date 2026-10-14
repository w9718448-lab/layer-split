#!/usr/bin/env python3
"""被遮挡元素的补全（流程里的必选步骤）。

元素从整张图里抠出来后，被前面物体挡住的部分是缺的。补全分三步：

  1. prepare   自动找出缺口，给每个需要补的元素准备 输入图 / 缺口蒙版 / 提示词
  2. 生成      优先用宿主环境自带的生图（图像编辑）模型补，逐个元素处理，结果存成 <编号>_<名称>.png
               没有生图模型时，用 --backend lama 在本地补（需要 torch 和 big-lama.pt，效果一般）
  3. apply     把补全结果合并回元素：原来可见的像素原样保留，只新增补出来的部分，写回 elements/

用法
  python complete.py prepare 原图.png --run 拆分输出目录
      生成 <run>/completion/tasks.json 和每个元素一个子目录（input.png、mask.png、prompt.txt）
  python complete.py apply 原图.png --run 拆分输出目录 --gen 生成结果目录
      生成结果目录里放 <编号>_<名称>.png（与 input.png 同尺寸，纯白底或透明底均可）
  python complete.py lama 原图.png --run 拆分输出目录
      不用外部生图，直接用本地 LaMa 补（等于 prepare + 本地生成 + apply）

补全前的元素会保留在 <run>/elements_original/，PSD 排序时要用。
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_layers as sl  # noqa: E402

PROMPT = (
    "这是一张插画里的单个物体（{name}），画风是{style}。物体的一部分被别的东西挡住了，"
    "蒙版里白色的区域是被挡住、需要补出来的位置。请把这个物体缺失的部分补完整，"
    "保持同样的线稿粗细、颜色、笔触和光影，结构要合理。"
    "只补这一个物体，不要添加新物体，背景保持纯白，不要阴影，不要白边。"
    "非蒙版区域保持原样不动。"
)


def load_run(run):
    man = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
    return man


def full_masks(man, run, W, H):
    """每个元素在原图坐标系下的可见掩码（1 倍）。"""
    res = {}
    for it in man["items"]:
        p = os.path.join(run, it["file"])
        if not os.path.exists(p):
            continue
        el = Image.open(p).convert("RGBA")
        x0, y0, x1, y1 = it["bbox"]
        a = el.split()[3].resize((x1 - x0, y1 - y0), Image.BICUBIC)
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = np.asarray(a) > 127
        res[it["id"]] = m
    return res


def find_hole(m, others, min_ratio=0.012):
    """缺口 = 元素的凸包（再略收紧）里，被别的元素占着的那部分。"""
    import cv2
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cs))], 1)
    hull = hull.astype(bool)
    # 凸包会把凹形元素的正常凹处也算进去，所以只取“紧挨着元素、且被别的元素占着”的部分
    near = ndi.binary_dilation(m, iterations=max(3, int(0.06 * max(m.sum() ** 0.5, 1))))
    hole = hull & near & others & ~m
    hole = ndi.binary_opening(hole, iterations=1)
    if hole.sum() < min_ratio * m.sum():
        return None
    return hole


def style_text(img):
    return "手绘线稿加平涂上色的插画（造景贴纸风格）"


def prepare(args):
    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    W, H = img.size
    man = load_run(args.run)
    masks = full_masks(man, args.run, W, H)
    allm = np.zeros((H, W), bool)
    for m in masks.values():
        allm |= m
    rgb = np.asarray(img)
    outd = os.path.join(args.run, "completion")
    os.makedirs(outd, exist_ok=True)
    tasks = []
    for it in man["items"]:
        m = masks.get(it["id"])
        if m is None:
            continue
        others = allm & ~m
        hole = find_hole(m, others, args.min_hole)
        if hole is None:
            continue
        ys, xs = np.where(m | hole)
        pad = max(int(0.12 * max(xs.max() - xs.min(), ys.max() - ys.min())), 8)
        x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
        x1, y1 = min(xs.max() + pad + 1, W), min(ys.max() + pad + 1, H)
        keep = ndi.binary_dilation(m, iterations=1)
        ctx = np.where(keep[..., None], rgb, 255).astype(np.uint8)[y0:y1, x0:x1]
        mk = (ndi.binary_dilation(hole, iterations=2) & ~m)[y0:y1, x0:x1]
        scale = float(np.clip(768 / max(x1 - x0, y1 - y0), 1.0, 4.0))
        size = (int((x1 - x0) * scale), int((y1 - y0) * scale))
        d = os.path.join(outd, f"{it['id']:03d}_{it['name']}")
        os.makedirs(d, exist_ok=True)
        Image.fromarray(ctx).resize(size, Image.BICUBIC).save(os.path.join(d, "input.png"))
        Image.fromarray((mk * 255).astype(np.uint8)).resize(size, Image.NEAREST).save(
            os.path.join(d, "mask.png"))
        open(os.path.join(d, "prompt.txt"), "w", encoding="utf-8").write(
            PROMPT.format(name=it["name"], style=style_text(img)))
        tasks.append({"id": it["id"], "name": it["name"], "dir": os.path.relpath(d, args.run),
                      "crop": [int(x0), int(y0), int(x1), int(y1)], "scale": scale,
                      "hole_ratio": round(float(hole.sum() / max(m.sum(), 1)), 3),
                      "result": f"{it['id']:03d}_{it['name']}.png"})
    sheets = pack_sheets(tasks, args.run, outd, args.pack) if args.pack > 1 else []
    bg = prepare_background(img, masks, outd)
    json.dump({"count": len(tasks), "tasks": tasks, "sheets": sheets, "background": bg},
              open(os.path.join(outd, "tasks.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({"need_completion": len(tasks), "of": len(man["items"]),
                      "gen_calls": (len(sheets) or len(tasks)) + 1, "dir": outd},
                     ensure_ascii=False))


SHEET_PROMPT = (
    "这张图是 {n} 个格子拼成的，每个格子里是同一幅插画里抠出来的一个独立物体，依次是：{names}。"
    "每个物体都有一部分被挡住了，蒙版里白色的区域是需要补出来的位置。"
    "请在每个格子里把该物体缺失的部分补完整，保持原来的线稿粗细、颜色和笔触，结构合理。"
    "每个格子只补它自己的物体，不要跨格子，不要添加新物体，背景保持纯白，不要阴影和白边，非蒙版区域保持不动。"
)
BG_PROMPT = (
    "这是一幅插画去掉所有物体之后剩下的场景，蒙版里白色的区域原来放着物体。"
    "请把白色区域补成干净的背景：室内就补地板、墙面、天花板、窗外等建筑本身的结构，延续原来的透视、纹理和颜色；"
    "室外就补地面、草地、天空等。不要画出任何家具、器物、人物或动物，只要空的场景。"
)


def pack_sheets(tasks, run, outd, n, cell=512):
    """把多个补全任务拼进一张图，一次生图调用补 n 个元素，减少调用次数。"""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sheets = []
    for k in range(0, len(tasks), n):
        group = tasks[k:k + n]
        sh_in = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
        sh_mk = Image.new("L", (cols * cell, rows * cell), 0)
        for j, t in enumerate(group):
            d = os.path.join(run, t["dir"])
            inp = Image.open(os.path.join(d, "input.png")).convert("RGB")
            mk = Image.open(os.path.join(d, "mask.png")).convert("L")
            f = (cell - 16) / max(inp.size)
            w, h = max(int(inp.width * f), 1), max(int(inp.height * f), 1)
            cx = (j % cols) * cell + (cell - w) // 2
            cy = (j // cols) * cell + (cell - h) // 2
            sh_in.paste(inp.resize((w, h), Image.BICUBIC), (cx, cy))
            sh_mk.paste(mk.resize((w, h), Image.NEAREST), (cx, cy))
            t["sheet"] = len(sheets)
            t["cell"] = [cx, cy, w, h]
        name = f"sheet_{len(sheets):02d}"
        sh_in.save(os.path.join(outd, name + "_input.png"))
        sh_mk.save(os.path.join(outd, name + "_mask.png"))
        open(os.path.join(outd, name + "_prompt.txt"), "w", encoding="utf-8").write(
            SHEET_PROMPT.format(n=len(group), names="、".join(t["name"] for t in group)))
        sheets.append({"name": name, "result": name + ".png", "ids": [t["id"] for t in group]})
    return sheets


def prepare_background(img, masks, outd):
    """背景 = 去掉所有元素后的底。元素区域涂白作为待补区域。"""
    H, W = img.height, img.width
    allm = np.zeros((H, W), bool)
    for m in masks.values():
        allm |= m
    allm = ndi.binary_dilation(allm, iterations=3)
    rgb = np.asarray(img)
    Image.fromarray(np.where(allm[..., None], 255, rgb).astype(np.uint8)).save(
        os.path.join(outd, "background_input.png"))
    Image.fromarray((allm * 255).astype(np.uint8)).save(os.path.join(outd, "background_mask.png"))
    open(os.path.join(outd, "background_prompt.txt"), "w", encoding="utf-8").write(BG_PROMPT)
    return {"input": "background_input.png", "mask": "background_mask.png",
            "prompt": "background_prompt.txt", "result": "background.png"}


def _merge(img, it, m, task, gen_rgba, W, H):
    """把生成结果并回元素。可见像素原样保留，只新增缺口里非白的部分。"""
    x0, y0, x1, y1 = task["crop"]
    s = task["scale"]
    cw, ch = x1 - x0, y1 - y0
    g = gen_rgba.convert("RGBA").resize((cw, ch), Image.LANCZOS)
    ga = np.asarray(g)
    if ga[..., 3].min() < 250:                    # 生成结果自带透明通道
        gen_fg = ga[..., 3] > 127
    else:                                          # 纯白底：非白即前景
        gen_fg = ga[..., :3].min(axis=2) < 235
    rgb = np.asarray(img)[y0:y1, x0:x1]
    mc = m[y0:y1, x0:x1]
    mask_png = Image.open(os.path.join(task["_run"], task["dir"], "mask.png")).convert("L").resize(
        (cw, ch), Image.NEAREST)
    hole = np.asarray(mask_png) > 127
    add = hole & gen_fg & ~mc
    add = ndi.binary_opening(add, iterations=1)
    new = mc | add
    new = ndi.binary_closing(new, iterations=2) & (mc | hole)
    new = new | (ndi.binary_fill_holes(new) & hole)   # 只在缺口范围内补洞，镂空的地方保持镂空
    new = sl._main_parts(new, keep=0.01)
    out_rgb = np.where(mc[..., None], rgb, ga[..., :3]).astype(np.uint8)
    sm = np.asarray(Image.fromarray((new * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.0))) > 127
    alpha = np.asarray(Image.fromarray((sm * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)))
    rgba = Image.fromarray(out_rgb).convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    res = sl.crop_to_alpha(rgba)
    if res is None:
        return None
    e, (ox, oy) = res
    return e, (x0 + ox, y0 + oy, x0 + ox + e.width, y0 + oy + e.height)


def apply(args, gen_fn=None):
    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    W, H = img.size
    man = load_run(args.run)
    masks = full_masks(man, args.run, W, H)
    tj = json.load(open(os.path.join(args.run, "completion", "tasks.json"), encoding="utf-8"))
    tasks, sheets = tj["tasks"], tj.get("sheets", [])
    apply_background(img, args, tj.get("background"), gen_fn is not None)
    orig_dir = os.path.join(args.run, "elements_original")
    if not os.path.exists(orig_dir):
        shutil.copytree(os.path.join(args.run, "elements"), orig_dir)
    by_id = {it["id"]: it for it in man["items"]}
    pending, ok = [], 0
    for t in tasks:
        t["_run"] = args.run
        it = by_id[t["id"]]
        if gen_fn is not None:
            gen = gen_fn(img, it, masks[t["id"]], t)
        else:
            gen = None
            gp = os.path.join(args.gen, t["result"])
            if os.path.exists(gp):
                gen = Image.open(gp).convert("RGBA")
            elif "sheet" in t:
                sp = os.path.join(args.gen, sheets[t["sheet"]]["result"])
                if os.path.exists(sp):
                    sh = Image.open(sp).convert("RGBA")
                    ref = Image.open(os.path.join(args.run, "completion",
                                                  sheets[t["sheet"]]["name"] + "_input.png"))
                    if sh.size != ref.size:
                        sh = sh.resize(ref.size, Image.LANCZOS)
                    cx, cy, w, h = t["cell"]
                    inp = Image.open(os.path.join(args.run, t["dir"], "input.png"))
                    gen = sh.crop((cx, cy, cx + w, cy + h)).resize(inp.size, Image.LANCZOS)
        if gen is None:
            continue
        r = _merge(img, it, masks[t["id"]], t, gen, W, H)
        if r is None:
            continue
        pending.append((it, r))
    if not pending:
        print("没有可合并的补全结果")
        return
    fs = []
    for it in man["items"]:
        bw = it["bbox"][2] - it["bbox"][0]
        if bw > 0 and it.get("size"):
            fs.append(it["size"][0] / bw)
    f = int(round(float(np.median(fs)))) if fs else 1
    ups = [p[1][0] for p in pending]
    if f > 1:
        ups, method = sl.superres(ups, f)
    for (it, (e, bbox)), e2 in zip(pending, ups):
        e2.save(os.path.join(args.run, it["file"]))
        it.setdefault("original_bbox", list(it["bbox"]))
        it["bbox"] = [int(v) for v in bbox]
        it["size"] = list(e2.size)
        it["completed"] = True
        ok += 1
    man["completed_count"] = ok
    json.dump(man, open(os.path.join(args.run, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({"completed": ok}, ensure_ascii=False))


def apply_background(img, args, bg, use_lama):
    if not bg:
        return
    outd = os.path.join(args.run, "completion")
    dst = os.path.join(args.run, "background.png")
    if use_lama:
        import complete_elements as ce
        inp = np.asarray(img)
        mk = np.asarray(Image.open(os.path.join(outd, bg["mask"])).convert("L")) > 127
        out = ce.lama_fill(inp, mk, target=min(max(inp.shape[:2]), 1280))
        Image.fromarray(out).save(dst)
    else:
        gp = os.path.join(args.gen, bg["result"])
        if not os.path.exists(gp):
            print("没有背景生成结果 background.png，PSD 背景层会用原图代替", file=sys.stderr)
            return
        g = Image.open(gp).convert("RGB").resize(img.size, Image.LANCZOS)
        # 元素以外的地方用原图，生成结果只填元素原来的位置
        mk = np.asarray(Image.open(os.path.join(outd, bg["mask"])).convert("L")) > 127
        out = np.where(mk[..., None], np.asarray(g), np.asarray(img)).astype(np.uint8)
        Image.fromarray(out).save(dst)
    print(json.dumps({"background": dst}, ensure_ascii=False))


def lama_gen(img, it, m, t):
    """本地 LaMa 兜底：用 input/mask 补，返回与 input 同尺寸的 RGBA。"""
    import complete_elements as ce
    d = os.path.join(t["_run"], t["dir"])
    inp = np.asarray(Image.open(os.path.join(d, "input.png")).convert("RGB"))
    mk = np.asarray(Image.open(os.path.join(d, "mask.png")).convert("L")) > 127
    filled = ce.lama_fill(inp, mk, target=max(inp.shape[:2]))
    return Image.fromarray(filled).convert("RGBA")


def main():
    ap = argparse.ArgumentParser(description="补全被遮挡的元素")
    ap.add_argument("cmd", choices=["prepare", "apply", "lama"])
    ap.add_argument("image")
    ap.add_argument("--run", required=True)
    ap.add_argument("--gen", help="apply：生成结果目录")
    ap.add_argument("--pack", type=int, default=4,
                    help="prepare：几个元素拼成一张图交给生图模型，默认 4（调用次数减为约 1/4），1 表示不拼")
    ap.add_argument("--min-hole", type=float, default=0.03,
                    help="prepare：缺口面积占元素面积低于这个比例的不补，默认 0.03")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "apply":
        apply(args)
    else:
        if not os.path.exists(os.path.join(args.run, "completion", "tasks.json")):
            prepare(args)
        apply(args, gen_fn=lama_gen)


if __name__ == "__main__":
    main()
