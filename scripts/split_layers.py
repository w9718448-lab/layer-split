#!/usr/bin/env python3
"""图层拆分：把一张复杂图拆成大量透明底元素（可做造景贴纸）。

两种模式
  auto   自动：整图前景分割 -> 连通域 -> 每个元素一张透明PNG
  boxes  按框：由 Claude 看图给出每个元素的框（JSON），脚本逐个抠图

用法
  python split_layers.py IMG --out OUT --mode auto
  python split_layers.py IMG --out OUT --mode boxes --boxes boxes.json

boxes.json 格式（坐标为原图像素，x0,y0,x1,y1；name 可选）
  [{"name": "mushroom", "box": [120, 80, 340, 300]}, ...]
  也支持归一化坐标：{"box": [0.1,0.2,0.3,0.4], "norm": true}
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage as ndi

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/{}.onnx"
MODEL_CFG = {
    # name: (input size, mean, std)
    "isnet-general-use": (1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "u2net": (320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # BiRefNet：边缘（细线、毛边、发丝）比 isnet 准，输出是 logits，要过 sigmoid
    "BiRefNet-general-epoch_244": (1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "BiRefNet-general-bb_swin_v1_tiny-epoch_232": (1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
# --refine 的简写 -> 模型文件名
REFINE_MODELS = {
    "birefnet": "BiRefNet-general-epoch_244",                     # 约 970MB，CPU 每次约十几秒
    "birefnet-lite": "BiRefNet-general-bb_swin_v1_tiny-epoch_232",  # 约 220MB，快很多
}
MODEL_DIRS = [
    os.environ.get("LAYER_SPLIT_MODELS", ""),
    "/home/claude/models",
    os.path.expanduser("~/.u2net"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
]

_sessions = {}


def get_session(name):
    if name in _sessions:
        return _sessions[name]
    import onnxruntime as ort

    path = None
    for d in MODEL_DIRS:
        if d and os.path.exists(os.path.join(d, name + ".onnx")):
            path = os.path.join(d, name + ".onnx")
            break
    if path is None:
        d = os.path.expanduser("~/.u2net")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name + ".onnx")
        print(f"下载模型 {name} ...", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL.format(name), path)
    opts = ort.SessionOptions()
    if name.startswith("BiRefNet"):
        # 默认的内存池会让 BiRefNet 第二次推理涨到 12GB 以上，关掉后稳定在 7GB 左右
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(path, opts, providers=providers)
    _sessions[name] = sess
    return sess


def matte(img_rgb, model="isnet-general-use"):
    """返回与 img_rgb 同尺寸的前景概率图 float32 [0,1]。"""
    size, mean, std = MODEL_CFG[model]
    sess = get_session(model)
    w, h = img_rgb.size
    x = np.asarray(img_rgb.resize((size, size), Image.LANCZOS), dtype=np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    if model.startswith("BiRefNet"):
        # 不做 min-max 拉伸：元素不显著时拉伸会把噪声放大成假前景
        out = 1.0 / (1.0 + np.exp(-out))
    else:
        out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    m = Image.fromarray((out * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    return np.asarray(m, dtype=np.float32) / 255.0


class Sam:
    """SAM (vit_b, ONNX) 框/点提示分割。

    hires 模式下每个元素单独裁切、放大后编码，掩码精度比整图编码高很多；
    整图编码只在 --no-hires 的快速模式用。
    """

    ENC, DEC = "vit_b-encoder-quant", "vit_b-decoder-quant"

    def __init__(self):
        self.enc = get_session(self.ENC)
        self.dec = get_session(self.DEC)
        self.emb = None

    def set_image(self, img_rgb, max_up=6.0):
        W, H = img_rgb.size
        self.scale = min(1024.0 / max(W, H), max_up)
        nw, nh = max(int(W * self.scale + 0.5), 1), max(int(H * self.scale + 0.5), 1)
        x = np.asarray(img_rgb.resize((nw, nh), Image.BICUBIC), dtype=np.float32)
        x = (x - np.array([123.675, 116.28, 103.53], np.float32)) / np.array(
            [58.395, 57.12, 57.375], np.float32)
        x = np.pad(x, ((0, 1024 - nh), (0, 1024 - nw), (0, 0)))
        self.emb = self.enc.run(None, {"x": x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        self.size = (H, W)
        self.new = (nw, nh)

    @staticmethod
    def prompts(box, pos=(), neg=()):
        pts, lab = [], []
        for p in pos:
            pts.append(list(p)); lab.append(1)
        for p in neg:
            pts.append(list(p)); lab.append(0)
        if box is not None:
            pts += [[box[0], box[1]], [box[2], box[3]]]
            lab += [2, 3]
        else:
            pts.append([0, 0]); lab.append(-1)
        return pts, lab

    def run(self, pts, lab, prev=None):
        """返回 (256x256 低分辨率 logits, 预测 IoU)。prev 是上一轮的 logits，用于迭代细化。"""
        coords = (np.array(pts, np.float32) * self.scale)[None]
        has = prev is not None
        _, iou, low = self.dec.run(None, {
            "image_embeddings": self.emb,
            "point_coords": coords,
            "point_labels": np.array(lab, np.float32)[None],
            "mask_input": (prev[None, None].astype(np.float32) if has
                           else np.zeros((1, 1, 256, 256), np.float32)),
            "has_mask_input": np.ones(1, np.float32) if has else np.zeros(1, np.float32),
            "orig_im_size": np.array(self.size, np.float32),
        })
        return low[0, 0].astype(np.float32), float(np.ravel(iou)[0])

    def to_mask(self, low, thr=0.0):
        """这个量化解码器自带的整图掩码输出不可用（数值整体偏负），
        所以用 256x256 低分辨率 logits 自己放大、去补边、缩回原尺寸。"""
        H, W = self.size
        nw, nh = self.new
        up = Image.fromarray(low, "F").resize((1024, 1024), Image.BICUBIC)
        up = up.crop((0, 0, nw, nh)).resize((W, H), Image.BICUBIC)
        return np.asarray(up) > thr

    def predict(self, box=None, pos=(), neg=()):
        pts, lab = self.prompts(box, pos, neg)
        low, _ = self.run(pts, lab)
        return self.to_mask(low)


def exclusive_ownership(found, W, H):
    """同一个像素只属于一个元素：重叠部分归面积小的元素（通常在前面）。
    家具、柜子框的是整体，SAM 会把里面的瓶罐也算进来，扣掉后柜子留下缺口，交给补全步骤补。"""
    masks = []
    for name, el, bb, q in found:
        a = np.asarray(el.split()[3]) > 127
        m = np.zeros((H, W), bool)
        x0, y0 = bb[0], bb[1]
        x1, y1 = min(x0 + a.shape[1], W), min(y0 + a.shape[0], H)
        m[y0:y1, x0:x1] = a[:y1 - y0, :x1 - x0]
        masks.append(m)
    order = sorted(range(len(found)), key=lambda i: masks[i].sum())
    taken = np.zeros((H, W), bool)
    out = list(found)
    for i in order:
        m = masks[i]
        lost = m & taken
        if lost.any() and lost.sum() < 0.97 * m.sum():
            name, el, bb, q = found[i]
            x0, y0 = bb[0], bb[1]
            keep = ~lost[y0:y0 + el.height, x0:x0 + el.width]
            keep = ndi.binary_opening(keep, iterations=1) | ~m[y0:y0 + el.height, x0:x0 + el.width]
            a = np.asarray(el.split()[3]).astype(np.float32) * keep
            m2 = a > 127
            m2 = _main_parts(m2, keep=0.01)
            a = np.where(m2, a, 0).astype(np.uint8)
            el2 = el.copy()
            el2.putalpha(Image.fromarray(a))
            res = crop_to_alpha(el2)
            if res is not None:
                e3, (ox, oy) = res
                out[i] = (name, e3, (x0 + ox, y0 + oy, x0 + ox + e3.width, y0 + oy + e3.height), q)
                m = np.zeros((H, W), bool)
                m[y0 + oy:y0 + oy + e3.height, x0 + ox:x0 + ox + e3.width] = np.asarray(e3.split()[3]) > 127
        taken |= m
    return out


def _bbox_of(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _box_iou(a, b):
    ix = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
    iy = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _main_parts(m, keep=0.02):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= sizes.max() * keep])


def snap_edges(rgb, m, band=None):
    """GrabCut 只在掩码边缘窄带内重新判定前景/背景，让边缘吸附到画面里真实的轮廓线上。"""
    try:
        import cv2
    except ImportError:
        return m
    h, w = m.shape
    r = band or int(np.clip(round(0.012 * max(h, w)), 2, 5))
    sure_fg = ndi.binary_erosion(m, iterations=r)
    maybe = ndi.binary_dilation(m, iterations=r)
    if sure_fg.sum() < 30 or (~maybe).sum() < 30:
        return m
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[maybe] = cv2.GC_PR_BGD
    gc[m] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(rgb[..., ::-1]), gc, None, bgd, fgd, 3,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    # 变化太大说明 GrabCut 跑偏了，保留 SAM 的结果
    if not (0.8 * m.sum() <= out.sum() <= 1.25 * m.sum()):
        return m
    return out


def _solidity(m):
    try:
        import cv2
    except ImportError:
        return None
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    pts = np.concatenate(cs)
    hull = cv2.contourArea(cv2.convexHull(pts))
    return float(m.sum() / hull) if hull > 0 else None


def sam_extract(img, sam, box, extra=None, hires=True, snap=True, hole_ratio=0.03, refine=None):
    """用 SAM 在框内分割一个元素。

    1. hires：裁出框周围区域放大后单独编码
    2. 生成多组候选（框+中心点 / 只有框 / 框+沿长边两点），按“掩码外接框与用户框的吻合度 + 预测 IoU”择优
    3. 用上一轮掩码再迭代一次细化
    4. 补小洞、找回黑色描边、GrabCut 窄带吸附边缘、平滑
    5. refine 给了模型名时，用 BiRefNet 重新判定边缘窄带（软边缘）
    返回 (RGBA, 原图中的 bbox, 质量指标)。
    """
    extra = extra or {}
    W, H = img.size
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return None
    rgb_full = img.convert("RGB")
    pad = max(int(0.35 * max(bw, bh)), 16)
    cx0, cy0, cx1, cy1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    crop = rgb_full.crop((cx0, cy0, cx1, cy1))
    ch, cw = crop.height, crop.width
    if hires:
        sam.set_image(crop)
        off = (cx0, cy0)
    else:
        off = (0, 0)

    def sh(pts):  # 原图坐标 -> SAM 输入坐标
        return [[p[0] - off[0], p[1] - off[1]] for p in pts]

    sbox = [x0 - off[0], y0 - off[1], x1 - off[0], y1 - off[1]]
    neg = sh(extra.get("neg") or [])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if extra.get("points"):
        specs = [sh(extra["points"])]
    else:
        specs = [sh([[cx, cy]]), [],
                 sh([[x0 + bw / 3, cy], [x0 + 2 * bw / 3, cy]] if bw >= bh
                    else [[cx, y0 + bh / 3], [cx, y0 + 2 * bh / 3]])]
    ubox = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)  # 用户框在裁切图坐标系里

    lim = np.zeros((ch, cw), bool)
    p2 = int(0.04 * max(bw, bh)) + 2
    lim[max(ubox[1] - p2, 0):ubox[3] + p2, max(ubox[0] - p2, 0):ubox[2] + p2] = True

    def to_crop_mask(low):
        full = sam.to_mask(low)
        m = full if hires else full[cy0:cy1, cx0:cx1]
        return _main_parts(m & lim)

    def score(m, iou_pred):
        bb = _bbox_of(m)
        return (_box_iou(bb, ubox) + 0.5 * iou_pred) if bb else -1.0

    best = None
    for pos in specs:
        pts, lab = sam.prompts(sbox, pos, neg)
        low, ip = sam.run(pts, lab)
        m = to_crop_mask(low)
        s = score(m, ip)
        if best is None or s > best[0]:
            best = (s, low, ip, pts, lab, m)
    s, low, ip, pts, lab, m = best
    low2, ip2 = sam.run(pts, lab, prev=low)  # 迭代细化
    m2 = to_crop_mask(low2)
    if score(m2, ip2) >= s * 0.98 and m2.any():
        m, ip = m2, ip2
    if not m.any():
        return None

    # 只补小洞；被前面的物体挡出来的大洞保持镂空
    filled = ndi.binary_fill_holes(m)
    holes, hn = ndi.label(filled & ~m)
    area = m.sum()
    for i in range(1, hn + 1):
        hm = holes == i
        if hm.sum() <= area * hole_ratio:
            m |= hm
    rgb = np.asarray(crop, dtype=np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    m = m | (ndi.binary_dilation(m, iterations=1) & ~m & (lum < 85))  # 找回被削掉的黑描边
    m = ndi.binary_closing(m, iterations=2)
    if snap:
        m = snap_edges(rgb.astype(np.uint8), m)
    sm = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.2))) > 127
    sm = _main_parts(sm, keep=0.08)   # 去掉连带进来的邻居碎片，元素只保留自己
    alpha = np.asarray(Image.fromarray((sm * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)))
    refined = refine_edges(crop, sm, refine) if refine else None
    if refined is not None:
        alpha = (refined * 255).astype(np.uint8)

    rgba = crop.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    out = crop_to_alpha(rgba)
    if out is None:
        return None
    el, (ox, oy) = out
    fb = _bbox_of(sm)
    q = {"box_iou": round(_box_iou(fb, ubox), 3), "pred_iou": round(ip, 3),
         "solidity": None}
    if refine:
        q["refined"] = refined is not None
    sol = _solidity(sm)
    if sol is not None:
        q["solidity"] = round(sol, 3)
    return el, (cx0 + ox, cy0 + oy, cx0 + ox + el.width, cy0 + oy + el.height), q


def refine_edges(crop, m, model, band=None, agree=0.6):
    """用 BiRefNet 重新判定 SAM 掩码边缘一圈的像素，返回 float alpha [0,1]；不可信时返回 None。

    只动边缘窄带：内部保持 SAM 的实心结果，窄带外一律透明，所以 BiRefNet 看中邻居也带不进来。
    裁切图里这个元素不够显著时（BiRefNet 在 SAM 掩码内的认同度低于 agree），放弃精修，用回 SAM。
    """
    if m.sum() < 64:
        return None
    prob = matte(crop, model)
    if float((prob[m] > 0.5).mean()) < agree:
        return None
    h, w = m.shape
    r = band or int(np.clip(round(0.015 * max(h, w)), 2, 6))
    core = ndi.binary_erosion(m, iterations=r)
    outer = ndi.binary_dilation(m, iterations=r)
    alpha = np.where(core, 1.0, np.where(outer, prob, 0.0)).astype(np.float32)
    keep = _main_parts(alpha > 0.5, keep=0.08)
    keep = ndi.binary_dilation(keep, iterations=1)
    return np.where(keep, alpha, 0.0).astype(np.float32)


def estimate_bg(arr, thr=28):
    """从四周边缘估计背景色；若边缘颜色足够一致，返回 (颜色, True)。"""
    h, w, _ = arr.shape
    b = max(2, min(h, w) // 100)
    edge = np.concatenate([
        arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
        arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)]).astype(np.float32)
    med = np.median(edge, axis=0)
    dist = np.linalg.norm(edge - med, axis=1)
    return med, float((dist < thr).mean()) > 0.9


def fit_gradient_bg(arr, thr):
    """用二次曲面拟合渐变背景（只用图像边缘的像素），返回 (背景图, 边缘残差合格)。"""
    h, w, _ = arr.shape
    b = max(3, min(h, w) // 40)
    yy, xx = np.mgrid[0:h, 0:w]
    ring = np.zeros((h, w), bool)
    ring[:b], ring[-b:], ring[:, :b], ring[:, -b:] = True, True, True, True
    ys, xs = yy[ring], xx[ring]
    if len(ys) > 30000:
        sel = np.random.default_rng(0).choice(len(ys), 30000, replace=False)
        ys, xs = ys[sel], xs[sel]

    def feats(x, y):
        u, v = x / max(w - 1, 1) * 2 - 1, y / max(h - 1, 1) * 2 - 1
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)

    A = feats(xs.astype(np.float32), ys.astype(np.float32))
    full = feats(xx.ravel().astype(np.float32), yy.ravel().astype(np.float32))
    bgimg = np.zeros((h, w, 3), np.float32)
    resid = []
    for c in range(3):
        y = arr[ys, xs, c].astype(np.float32)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid.append(np.abs(A @ coef - y))
        bgimg[..., c] = (full @ coef).reshape(h, w)
    resid = np.sqrt(sum(r * r for r in resid))
    return bgimg, float((resid < thr).mean()) > 0.9


def matte_tiled(img, model, grids=(1, 2, 3), overlap=0.15):
    """整图 + 多档网格切块分别抠图再取并集，避免显著性模型只挑出最大的几个目标。"""
    W, H = img.size
    out = np.zeros((H, W), np.float32)
    for g in grids:
        if g == 1:
            out = np.maximum(out, matte(img, model))
            continue
        tw, th = int(W / g * (1 + overlap)), int(H / g * (1 + overlap))
        for i in range(g):
            for j in range(g):
                x0 = min(int(j * W / g - W / g * overlap / 2), W - tw)
                y0 = min(int(i * H / g - H / g * overlap / 2), H - th)
                x0, y0 = max(x0, 0), max(y0, 0)
                m = matte(img.crop((x0, y0, x0 + tw, y0 + th)), model)
                out[y0:y0 + th, x0:x0 + tw] = np.maximum(out[y0:y0 + th, x0:x0 + tw], m)
    return out


def foreground_mask(img, bg="auto", thr=28, model="isnet-general-use"):
    """整图前景掩码 (bool)。纯色背景用色差；渐变背景用曲面拟合；其余用分块抠图模型。"""
    rgba = img.convert("RGBA")
    a = np.asarray(rgba)
    if a[..., 3].min() < 250:  # 已有透明通道
        return a[..., 3] > 16, "alpha"
    arr = a[..., :3]
    med, uniform = estimate_bg(arr)
    if bg in ("auto", "color") and uniform:
        dist = np.linalg.norm(arr.astype(np.float32) - med, axis=2)
        return ndi.binary_opening(dist > thr, iterations=1), "color"
    if bg in ("auto", "gradient"):
        bgimg, ok = fit_gradient_bg(arr, thr)
        if ok or bg == "gradient":
            dist = np.linalg.norm(arr.astype(np.float32) - bgimg, axis=2)
            return ndi.binary_opening(dist > thr, iterations=1), "gradient"
    return matte_tiled(img.convert("RGB"), model) > 0.5, "model"


def components(mask, merge, min_area):
    """膨胀合并相邻碎片后做连通域，返回 [(bbox, 组掩码)]，按阅读顺序排序。"""
    if merge > 0:
        st = ndi.generate_binary_structure(2, 2)
        grown = ndi.binary_dilation(mask, structure=st, iterations=merge)
    else:
        grown = mask
    lab, n = ndi.label(grown)
    res = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        grp = (lab[sl] == i) & mask[sl]
        if grp.sum() < min_area:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        res.append(((x0, y0, x1, y1), grp))
    # 阅读顺序：先按行带再按 x
    if res:
        hs = [b[3] - b[1] for b, _ in res]
        band = max(40, int(np.median(hs)))
        res.sort(key=lambda r: (r[0][1] // band, r[0][0]))
    return res


def keep_main_parts(alpha, keep_ratio=0.02):
    """去掉远小于主体的杂点，保留主体及较大的附件。"""
    mask = alpha > 24
    lab, n = ndi.label(mask)
    if n <= 1:
        return alpha
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    big = sizes.max()
    keep = np.zeros_like(mask)
    for i, s in enumerate(sizes, start=1):
        if s >= big * keep_ratio:
            keep |= lab == i
    keep = ndi.binary_dilation(keep, iterations=2)
    return np.where(keep, alpha, 0)


def crop_to_alpha(rgba, pad=4):
    a = np.asarray(rgba.split()[3])
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return None
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, rgba.width)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, rgba.height)
    return rgba.crop((x0, y0, x1, y1)), (x0, y0)


def defringe(rgba, bg_color=None):
    """轻微收边，去掉抠图后残留的底色光晕。"""
    a = np.asarray(rgba).copy()
    alpha = a[..., 3].astype(np.float32)
    eroded = ndi.grey_erosion(alpha, size=(2, 2))
    a[..., 3] = np.clip((alpha * 0.5 + eroded * 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def extract_element(img, box, group_mask=None, model="isnet-general-use",
                    use_model=True, pad_ratio=0.12, bg_color=None):
    """从原图抠出一个元素，返回 (RGBA, (x0,y0,x1,y1) 在原图中的位置)。"""
    W, H = img.size
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    p = int(max(x1 - x0, y1 - y0) * pad_ratio)
    cx0, cy0, cx1, cy1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    crop = img.convert("RGB").crop((cx0, cy0, cx1, cy1))

    if group_mask is not None:
        # auto 模式：先用整图掩码得到粗 alpha，再用模型细化边缘
        full = np.zeros((cy1 - cy0, cx1 - cx0), dtype=bool)
        full[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = group_mask
        coarse = ndi.binary_dilation(full, iterations=3)
        alpha = coarse.astype(np.float32)
        # 内部直接取实心（填洞后内缩），只在边缘一圈用模型/羽化，避免模型把浅色内部抠半透明
        core = ndi.binary_erosion(ndi.binary_fill_holes(full), iterations=2)
        soft = np.asarray(Image.fromarray((full * 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(0.8))) / 255.0
        alpha = np.where(core, 1.0, soft).astype(np.float32)
        if use_model:
            m = matte(crop, model)
            band = coarse & ~core
            # 边缘带里模型与色差掩码都认为是前景才取模型值，否则用羽化值
            alpha = np.where(band, np.where(m > 0.3, np.maximum(m, soft), soft), alpha)
            alpha = np.where(core, 1.0, alpha).astype(np.float32)
    else:
        m = matte(crop, model)
        bm = m > 0.5
        # 背景接近纯色/渐变时，用色差补上模型漏掉的浅色内部（树干、白色茎等）
        carr = np.asarray(crop)
        if carr.shape[0] > 12 and carr.shape[1] > 12:
            med, uniform = estimate_bg(carr)
            if uniform:
                cm = np.linalg.norm(carr.astype(np.float32) - med, axis=2) > 28
            else:
                bgimg, ok = fit_gradient_bg(carr, 28)
                cm = (np.linalg.norm(carr.astype(np.float32) - bgimg, axis=2) > 28) if ok else None
            if cm is not None:
                cm = ndi.binary_opening(cm, iterations=1)
                lab_c, _ = ndi.label(cm)
                touch = np.unique(lab_c[ndi.binary_dilation(bm, iterations=2) & cm])
                bm = bm | np.isin(lab_c, touch[touch > 0])
        # 框内只保留与框相交的主体，避免把邻居带进来
        lab, n = ndi.label(bm)
        if n > 1:
            bx0, by0, bx1, by1 = x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0
            inner = lab[by0:by1, bx0:bx1]
            ids, cnt = np.unique(inner[inner > 0], return_counts=True)
            if len(ids):
                bm = np.isin(lab, ids[cnt >= cnt.max() * 0.05])
        solid = ndi.binary_fill_holes(bm)
        core = ndi.binary_erosion(solid, iterations=2)
        soft = np.asarray(Image.fromarray((solid * 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(0.8))) / 255.0
        edge = ndi.binary_dilation(solid, iterations=3) & ~core
        alpha = np.where(core, 1.0, np.where(edge, np.maximum(np.where(m > 0.3, m, 0), soft), 0.0))
        alpha = alpha.astype(np.float32)

    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    a8 = keep_main_parts(a8)
    rgba = crop.convert("RGBA")
    rgba.putalpha(Image.fromarray(a8))
    rgba = defringe(rgba, bg_color)
    out = crop_to_alpha(rgba)
    if out is None:
        return None
    el, (ox, oy) = out
    return el, (cx0 + ox, cy0 + oy, cx0 + ox + el.width, cy0 + oy + el.height)


def make_sticker(rgba, border=10, color=(255, 255, 255), shadow=True):
    """加白边（模切贴纸效果）和轻微阴影。

    白边用距离变换生成，宽度均匀、转角圆润；先把元素之间不到两倍边宽的缝隙闭合，
    再填内部空洞，这样不会出现锯齿或断开的白边。
    """
    border = max(int(border), 1)
    pad = border * 2 + 10
    canvas = Image.new("RGBA", (rgba.width + pad * 2, rgba.height + pad * 2), (0, 0, 0, 0))
    canvas.paste(rgba, (pad, pad), rgba)
    alpha = np.asarray(canvas.split()[3]) > 8
    d_out = ndi.distance_transform_edt(~alpha)
    grown = d_out <= border + border * 0.6          # 先多长一点，用来闭合缝隙
    grown = ndi.distance_transform_edt(grown) > border * 0.6   # 再收回去，整体外扩 = border
    grown = grown | alpha
    grown = ndi.binary_fill_holes(grown)
    edge = Image.fromarray((grown * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    edge = edge.point(lambda v: 255 if v > 200 else (0 if v < 55 else int((v - 55) * 255 / 145)))
    out = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sh.putalpha(edge.point(lambda v: int(v * 0.28)))
        sh = sh.filter(ImageFilter.GaussianBlur(4))
        out.alpha_composite(sh, (2, 4))
    white = Image.new("RGBA", canvas.size, color + (255,))
    white.putalpha(edge)
    out.alpha_composite(white)
    out.alpha_composite(canvas)
    return out


def _font(size=13):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "C:/Windows/Fonts/msyh.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def contact_sheet(items, path, cell=220, cols=6, bg=(236, 236, 240), start=0):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 24)), bg)
    # 棋盘格底，方便看透明
    d = ImageDraw.Draw(sheet)
    font = _font(13)
    for i, (name, im) in enumerate(items):
        r, c = divmod(i, cols)
        ox, oy = c * cell, r * (cell + 24)
        for yy in range(0, cell, 16):
            for xx in range(0, cell, 16):
                shade = 250 if ((xx + yy) // 16) % 2 == 0 else 226
                d.rectangle([ox + xx, oy + yy, ox + xx + 15, oy + yy + 15], fill=(shade,) * 3)
        t = im.copy()
        t.thumbnail((cell - 16, cell - 16), Image.LANCZOS)
        sheet.paste(t, (ox + (cell - t.width) // 2, oy + (cell - t.height) // 2), t)
        d.text((ox + 6, oy + cell + 3), f"{start + i + 1:03d} {name}"[:22], fill=(60, 60, 70), font=font)
    sheet.save(path)


def paged_sheets(items, out_dir, prefix, per_page=24):
    """元素不多时出一张总览图；超过 per_page 个时分页（编号连续），避免图太长看不清。"""
    if len(items) <= per_page:
        contact_sheet(items, os.path.join(out_dir, prefix + ".png"))
        return
    for p in range(0, len(items), per_page):
        chunk = items[p:p + per_page]
        tmp = [(n, im) for n, im in chunk]
        contact_sheet(tmp, os.path.join(out_dir, f"{prefix}_{p // per_page + 1}.png"),
                      cell=200, cols=6, start=p)


def annotate(img, records, path):
    """在原图上画出编号框，用来检查有没有漏拆或框错。"""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, r in enumerate(records, start=1):
        x0, y0, x1, y1 = r["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=3)
        d.rectangle([x0, y0, x0 + 34, y0 + 18], fill=(255, 60, 60))
        d.text((x0 + 4, y0 + 3), f"{i:03d}", fill=(255, 255, 255))
    im.save(path)


SR_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
          "realesrgan-ncnn-vulkan-20220424-ubuntu.zip")
# 紧凑的动漫/插画专用模型：线稿会变清晰，速度比 x4plus 版快约 9 倍（2 核 CPU 上每个元素约 1.5 秒）
SR_MODEL = "realesr-animevideov3-x{}"


def find_sr():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("LAYER_SPLIT_SR", ""), "/home/claude/models/resr",
              os.path.expanduser("~/.u2net/resr"), os.path.join(here, "..", "models", "resr")):
        if d and os.path.exists(os.path.join(d, "realesrgan-ncnn-vulkan")):
            return d
    return None


def ensure_sr():
    d = find_sr()
    if d:
        return d
    import zipfile
    d = os.path.expanduser("~/.u2net/resr")
    os.makedirs(d, exist_ok=True)
    zp = os.path.join(d, "resr.zip")
    print("下载超分模型 Real-ESRGAN(ncnn) ...", file=sys.stderr)
    urllib.request.urlretrieve(SR_URL, zp)
    zipfile.ZipFile(zp).extractall(d)
    os.chmod(os.path.join(d, "realesrgan-ncnn-vulkan"), 0o755)
    return d


def _run_sr(d, indir, outdir, scale):
    import subprocess
    env = dict(os.environ)
    lvp = "/usr/share/vulkan/icd.d/lvp_icd.json"
    if "VK_ICD_FILENAMES" not in env and os.path.exists(lvp):
        env["VK_ICD_FILENAMES"] = lvp          # 没有显卡时用 CPU 软件 Vulkan
    return subprocess.run(
        [os.path.join(d, "realesrgan-ncnn-vulkan"), "-i", indir, "-o", outdir, "-n", SR_MODEL.format(scale),
         "-s", str(scale), "-f", "png", "-m", os.path.join(d, "models")],
        env=env, capture_output=True, text=True, timeout=3600)


def superres(els, scale):
    """把一批 RGBA 元素放大 scale 倍。颜色走 Real-ESRGAN，边缘 alpha 单独放大再锐化，避免光晕。
    超分不可用时回退为 Lanczos 放大 + 轻锐化。返回 (放大后的列表, 方法名)。"""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="lsplit_sr_")
    ups = None
    method = "lanczos"
    try:
        d = ensure_sr()
        indir, outdir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)
        for i, el in enumerate(els):
            el.convert("RGB").save(os.path.join(indir, f"{i:04d}.png"))
        r = _run_sr(d, indir, outdir, scale)
        if "vkCreateInstance failed" in (r.stderr + r.stdout) and os.geteuid() == 0:
            import subprocess
            print("没有可用的 Vulkan 驱动，尝试安装 mesa-vulkan-drivers ...", file=sys.stderr)
            subprocess.run(["apt-get", "install", "-y", "-q", "mesa-vulkan-drivers"],
                           capture_output=True, timeout=600)
            r = _run_sr(d, indir, outdir, scale)
        outs = sorted(os.listdir(outdir))
        if len(outs) == len(els):
            ups = [Image.open(os.path.join(outdir, f)).convert("RGB") for f in outs]
            method = "realesrgan"
        else:
            print("超分不可用，改用 Lanczos 放大：" + (r.stderr or "")[-200:], file=sys.stderr)
    except Exception as e:  # 网络/驱动/二进制问题都回退
        print(f"超分不可用，改用 Lanczos 放大：{e}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res = []
    for i, el in enumerate(els):
        w, h = el.size
        tw, th = int(w * scale), int(h * scale)
        if ups is not None:
            rgb = ups[i].resize((tw, th), Image.LANCZOS)
        else:
            rgb = el.convert("RGB").resize((tw, th), Image.LANCZOS).filter(
                ImageFilter.UnsharpMask(radius=1.4, percent=90, threshold=2))
        a = np.asarray(el.split()[3].resize((tw, th), Image.BICUBIC), dtype=np.float32) / 255.0
        a = np.clip((a - 0.56) * 2.6 + 0.5, 0, 1)      # 锐化 alpha 边缘，边界略收进黑色描边里
        out = rgb.convert("RGBA")
        out.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
        res.append(out)
    return res, method


# 每个元素的耗时（秒），2 核/4 核 CPU 无显卡实测；有显卡时明显更快，按这个估算算上限
SEC_SPLIT = 6.0      # SAM 分割
SEC_SR = 1.5         # Real-ESRGAN 超分（--upscale > 1 时）
SEC_REFINE = {"birefnet-lite": 30.0, "birefnet": 60.0}


def _fmt_min(sec):
    m = sec / 60
    return f"约 {m:.0f} 分钟" if m >= 1.5 else "约 1 分钟"


def estimate(n, args):
    """估算两种拆法的耗时，供拆分前让用户选择。"""
    try:
        import onnxruntime as ort
        gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        gpu = False
    base = n * (SEC_SPLIT + (SEC_SR if args.upscale > 1 else 0))
    refine = "birefnet-lite"
    fine = base + n * SEC_REFINE[refine]
    note = "检测到显卡，实际会比估算快很多" if gpu else "没有显卡，按 CPU 估算"
    return {
        "elements": n, "gpu": gpu, "note": note,
        "options": [
            {"name": "标准", "args": "", "minutes": round(base / 60, 1), "time": _fmt_min(base),
             "desc": "SAM 分割，速度快；平涂插画的轮廓外可能带 1 像素左右的浅色毛边"},
            {"name": "精修", "args": f"--refine {refine}", "minutes": round(fine / 60, 1),
             "time": _fmt_min(fine),
             "desc": "在标准基础上用 BiRefNet 重新判定边缘一圈，毛边和底色光晕更少；"
                     f"每个元素多约 {SEC_REFINE[refine]:.0f} 秒，内存需要约 7GB"},
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="图层拆分 / 造景贴纸")
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["auto", "boxes"], default="auto")
    ap.add_argument("--boxes", help="boxes 模式用的 JSON")
    ap.add_argument("--bg", choices=["auto", "color", "gradient", "model"], default="auto",
                    help="auto 模式的背景判断：纯色 color / 渐变 gradient / 复杂背景 model（分块抠图）")
    ap.add_argument("--thr", type=int, default=28, help="色差阈值")
    ap.add_argument("--merge", type=int, default=6, help="auto：合并相邻碎片的膨胀像素")
    ap.add_argument("--min-area", type=int, default=400, help="auto：忽略小于该面积的元素")
    ap.add_argument("--engine", choices=["sam", "matte"], default="sam",
                    help="boxes 模式的分割引擎：sam 适合元素密集、互相叠压的图；matte 适合单个主体+简单背景")
    ap.add_argument("--no-hires", action="store_true",
                    help="boxes+sam：整图只编码一次（快，但小元素边缘更粗）。默认每个元素单独放大编码")
    ap.add_argument("--no-exclusive", action="store_true",
                    help="不做像素归属：默认重叠像素归面积小的元素（柜子里的瓶罐从柜子图层里扣掉，柜子留缺口待补全）")
    ap.add_argument("--snap", action="store_true",
                    help="boxes+sam：用 GrabCut 吸附边缘（实测在遮挡处会带进邻居颜色，默认关闭）")
    ap.add_argument("--upscale", type=int, default=2, choices=[1, 2, 3, 4],
                    help="元素放大倍数，用 Real-ESRGAN 把线稿变清晰，1 表示不放大（默认 2）")
    ap.add_argument("--refine", choices=["none"] + list(REFINE_MODELS), default="none",
                    help="boxes+sam：用 BiRefNet 精修元素边缘（细线、毛边更干净）。birefnet-lite 快，birefnet 更准")
    ap.add_argument("--model", default="isnet-general-use", choices=list(MODEL_CFG))
    ap.add_argument("--no-refine", action="store_true", help="auto：不用模型细化边缘")
    ap.add_argument("--sticker", action="store_true",
                    help="另外生成带白边的贴纸版 stickers/（默认只输出元素本身，不加白边）")
    ap.add_argument("--border", type=int, default=0,
                    help="仅配合 --sticker：白边宽度(px)，0 表示自动（默认）")
    ap.add_argument("--min-size", type=int, default=24, help="输出元素的最小边长")
    ap.add_argument("--zip", action="store_true", help="结果打包成 <out>.zip")
    ap.add_argument("--estimate", action="store_true",
                    help="不拆，只估算“标准”和“精修”两种拆法的耗时，给用户选")
    args = ap.parse_args()

    img = Image.open(args.image)
    img = ImageOps.exif_transpose(img)
    W, H = img.size
    out = args.out
    tr_dir = os.path.join(out, "elements")      # 元素本身：透明底，无白边（主输出）
    st_dir = os.path.join(out, "stickers")      # 可选：--sticker 才生成的白边贴纸版
    os.makedirs(tr_dir, exist_ok=True)
    if args.sticker:
        os.makedirs(st_dir, exist_ok=True)
    base_border = args.border if args.border > 0 else int(np.clip(round(0.009 * max(W, H)), 5, 24))

    jobs = []  # (name, bbox, group_mask, extra)
    info = {}
    if args.mode == "auto":
        mask, how = foreground_mask(img, args.bg, args.thr, args.model)
        info["foreground_by"] = how
        for i, (bb, grp) in enumerate(components(mask, args.merge, args.min_area), start=1):
            jobs.append((f"item{i:03d}", bb, grp, None))
    else:
        if not args.boxes:
            sys.exit("boxes 模式需要 --boxes")
        with open(args.boxes, encoding="utf-8") as f:
            data = json.load(f)
        for i, d in enumerate(data, start=1):
            bx = list(d["box"])
            if d.get("norm") or max(bx) <= 1.0:
                bx = [bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H]
            name = "".join(c for c in str(d.get("name") or f"item{i:03d}")
                           if c.isalnum() or c in "-_ ").strip().replace(" ", "_") or f"item{i:03d}"
            jobs.append((name, bx, None, {"points": d.get("points"), "neg": d.get("neg")}))

    if args.estimate:
        est = estimate(len(jobs), args)
        # 前两行就是给用户看的原话，后面是给 AI 用的详细数据
        for i, o in enumerate(est["options"], start=1):
            print(f"{i}. {o['name']}版：{o['time']}，共 {est['elements']} 个元素")
        print(json.dumps(est, ensure_ascii=False))
        return

    sam = None
    hires = not args.no_hires
    if args.mode == "boxes" and args.engine == "sam":
        sam = Sam()
        if not hires:
            sam.set_image(img.convert("RGB"))
        info["engine"] = "sam" + ("" if hires else "(整图编码)")
        if args.refine != "none":
            info["refine"] = args.refine

    found = []   # (name, el, bbox, quality)
    for idx, (name, bb, grp, extra) in enumerate(jobs, start=1):
        q = None
        if sam is not None:
            res = sam_extract(img, sam, bb, extra, hires=hires, snap=args.snap,
                              refine=REFINE_MODELS.get(args.refine))
            if res is not None:
                el, bbox, q = res
        else:
            res = extract_element(img, bb, grp, args.model, use_model=not args.no_refine)
            if res is not None:
                el, bbox = res
        if res is None or min(el.size) < args.min_size:
            continue
        found.append((name, el, bbox, q))
        if idx % 10 == 0:
            print(f"  分割 {idx}/{len(jobs)}", file=sys.stderr, flush=True)

    if found and not args.no_exclusive:
        found = exclusive_ownership(found, W, H)

    up = args.upscale
    if up > 1 and found:
        print(f"  超分放大 x{up}（{len(found)} 个元素）...", file=sys.stderr, flush=True)
        ups, method = superres([f[1] for f in found], up)
        info["upscale"] = f"x{up} {method}"
        found = [(n, e2, bb, q) for (n, _, bb, q), e2 in zip(found, ups)]

    # 编号按 Z 字形阅读顺序：从左上到右下，同一行带内从左到右
    if found:
        band = max(60, int(np.median([f[2][3] - f[2][1] for f in found]) * 0.6))
        found.sort(key=lambda f: (f[2][1] // band, f[2][0]))

    records, sheet_items, sticker_items, flagged = [], [], [], []
    for name, el, bbox, q in found:
        fn = f"{len(records) + 1:03d}_{name}.png"
        el.save(os.path.join(tr_dir, fn))
        rec = {"id": len(records) + 1, "name": name, "file": f"elements/{fn}",
               "bbox": [int(v) for v in bbox], "size": [int(v) for v in el.size]}
        if q:
            rec["quality"] = q
            # 外接框和用户框吻合度低 = 可能缺了一块或带进了邻居；这些元素需要人工看一眼
            if q["box_iou"] < 0.72:
                rec["review"] = True
                flagged.append((name, el))
        if args.sticker:
            b = int(np.clip(round(min(el.size) * 0.12), 3, base_border * up))
            stk = make_sticker(el, b)
            stk.save(os.path.join(st_dir, fn))
            rec["sticker"] = f"stickers/{fn}"
            sticker_items.append((name, stk))
        records.append(rec)
        sheet_items.append((name, el))

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(args.image), "image_size": [W, H],
                   "mode": args.mode, **info, "count": len(records), "items": records},
                  f, ensure_ascii=False, indent=2)
    paged_sheets(sheet_items, out, "sheet_elements")
    if sticker_items:
        paged_sheets(sticker_items, out, "sheet_stickers")
    if flagged:
        contact_sheet(flagged[:48], os.path.join(out, "review_these.png"), cell=220, cols=6)
    annotate(img, records, os.path.join(out, "annotated.png"))

    summary = {"count": len(records), "out": out, **info,
               "review": [r["name"] for r in records if r.get("review")]}
    if args.zip:
        import zipfile
        zpath = out.rstrip("/\\") + ".zip"
        keep_dirs = ("elements", "stickers")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, dn, fn_ in os.walk(out):
                for f_ in sorted(fn_):
                    p = os.path.join(dp, f_)
                    rel = os.path.relpath(p, out)
                    if rel.split(os.sep)[0] in keep_dirs or rel in ("manifest.json", "annotated.png") \
                            or rel.startswith("sheet_"):
                        z.write(p, os.path.join(os.path.basename(out.rstrip("/\\")), rel))
        summary["zip"] = zpath
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
