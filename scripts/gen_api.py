#!/usr/bin/env python3
"""补全步骤的生图 API 后端：读 complete.py prepare 生成的 tasks.json，逐个调用生图 API，
把结果按 tasks.json 里的 result 文件名存进一个目录，之后用 complete.py apply --gen 合并。

不需要宿主环境自带生图模型，适合整批无人值守地跑。只用标准库发 HTTP 请求，不加新依赖。

后端（API Key 从环境变量读，不要写进文件）
  gemini   Google Gemini 图像模型（Nano Banana 系列）        GEMINI_API_KEY
  openai   OpenAI 图像编辑（gpt-image 系列，支持蒙版）         OPENAI_API_KEY
  fal      fal.ai 上的局部重绘模型（默认 FLUX Fill，支持蒙版）  FAL_KEY

用法
  python gen_api.py 拆分输出目录 --backend gemini [--model 模型名] [--workers 4]
  结果默认存到 <run>/completion/gen/，已存在的结果跳过（中断后重跑会接着做）

各后端对输入尺寸比例有限制，脚本会先把输入补白到最接近的支持比例，拿到结果后再裁回原比例。
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

DEFAULT_MODEL = {
    "gemini": "gemini-nano-banana-2.1",
    "openai": "gpt-image-2",
    "fal": "fal-ai/flux-pro/v1/fill",
}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "fal": "FAL_KEY"}
# 各后端支持的宽高比；输入先补白到最接近的一档
RATIOS = {
    "gemini": [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9)],
    "openai": [(1, 1), (2, 3), (3, 2)],
    "fal": None,  # 不限比例
}
MASK_NOTE = (
    "\n\n第一张图是要修改的画面，第二张图是蒙版：蒙版里白色的位置是需要补画的区域，黑色的位置必须保持原样。"
    "输出一张和第一张图同样构图、同样尺寸比例的完整图片。"
)


def _png_b64(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _post(url, body, headers, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get_bytes(url, timeout=300):
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def pad_to_ratio(img, mask, ratios):
    """补白到最接近的支持比例，返回 (新图, 新蒙版, 原图在新图中的框)。蒙版补黑（不改）。"""
    w, h = img.size
    if not ratios:
        return img, mask, (0, 0, w, h)
    rw, rh = min(ratios, key=lambda r: abs(r[0] / r[1] - w / h))
    if w / h > rw / rh:
        nw, nh = w, int(round(w * rh / rw))
    else:
        nw, nh = int(round(h * rw / rh)), h
    ox, oy = (nw - w) // 2, (nh - h) // 2
    pi = Image.new("RGB", (nw, nh), (255, 255, 255))
    pi.paste(img.convert("RGB"), (ox, oy))
    pm = Image.new("L", (nw, nh), 0)
    pm.paste(mask.convert("L"), (ox, oy))
    return pi, pm, (ox, oy, ox + w, oy + h)


def call_gemini(img, mask, prompt, model, key):
    body = {
        "contents": [{"parts": [
            {"text": prompt + MASK_NOTE},
            {"inline_data": {"mime_type": "image/png", "data": _png_b64(img)}},
            {"inline_data": {"mime_type": "image/png", "data": _png_b64(mask)}},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    res = _post(url, body, {"x-goog-api-key": key})
    for c in res.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            d = p.get("inlineData") or p.get("inline_data")
            if d and d.get("data"):
                return Image.open(io.BytesIO(base64.b64decode(d["data"])))
    raise RuntimeError("Gemini 没有返回图片：" + json.dumps(res, ensure_ascii=False)[:300])


def call_openai(img, mask, prompt, model, key):
    # OpenAI 的蒙版约定：透明（alpha=0）的位置是要改的区域，尺寸和输入图一致
    m = mask.convert("L").point(lambda v: 0 if v > 127 else 255)
    mrgba = Image.new("RGBA", img.size, (0, 0, 0, 255))
    mrgba.putalpha(m)
    body = {
        "model": model,
        "prompt": prompt,
        "images": [{"image_url": "data:image/png;base64," + _png_b64(img)}],
        "mask": {"image_url": "data:image/png;base64," + _png_b64(mrgba)},
        "output_format": "png",
    }
    res = _post("https://api.openai.com/v1/images/edits", body, {"Authorization": "Bearer " + key})
    data = res.get("data") or []
    if not data or not data[0].get("b64_json"):
        raise RuntimeError("OpenAI 没有返回图片：" + json.dumps(res, ensure_ascii=False)[:300])
    return Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"])))


def call_fal(img, mask, prompt, model, key):
    body = {
        "prompt": prompt,
        "image_url": "data:image/png;base64," + _png_b64(img),
        "mask_url": "data:image/png;base64," + _png_b64(mask.convert("L")),
        "sync_mode": True,
    }
    res = _post("https://fal.run/" + model, body, {"Authorization": "Key " + key})
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("url"):
        raise RuntimeError("fal 没有返回图片：" + json.dumps(res, ensure_ascii=False)[:300])
    return Image.open(io.BytesIO(_get_bytes(imgs[0]["url"])))


BACKENDS = {"gemini": call_gemini, "openai": call_openai, "fal": call_fal}


def generate(backend, inp, mask, prompt, model, key, retries=3):
    """补白 -> 调用（失败重试）-> 缩放回补白尺寸 -> 裁回原图范围。"""
    pi, pm, box = pad_to_ratio(inp, mask, RATIOS[backend])
    last = None
    for k in range(retries):
        try:
            out = BACKENDS[backend](pi, pm, prompt, model, key)
            out = out.convert("RGBA").resize(pi.size, Image.LANCZOS)
            return out.crop(box)
        except urllib.error.HTTPError as e:
            last = RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
            if e.code in (400, 401, 403, 404):  # 参数或权限错误，重试没用
                break
        except Exception as e:  # noqa: BLE001  网络抖动、超时等
            last = e
        time.sleep(2 ** (k + 1))
    raise last


def jobs_from_tasks(run, tj, include_bg=True):
    """返回 [(结果文件名, 输入图路径, 蒙版路径, 提示词)]，拼图时按拼图出任务，否则按元素。"""
    comp = os.path.join(run, "completion")
    jobs = []
    if tj.get("sheets"):
        for s in tj["sheets"]:
            jobs.append((s["result"], os.path.join(comp, s["name"] + "_input.png"),
                         os.path.join(comp, s["name"] + "_mask.png"),
                         os.path.join(comp, s["name"] + "_prompt.txt")))
    else:
        for t in tj["tasks"]:
            d = os.path.join(run, t["dir"])
            jobs.append((t["result"], os.path.join(d, "input.png"), os.path.join(d, "mask.png"),
                         os.path.join(d, "prompt.txt")))
    bg = tj.get("background")
    if include_bg and bg:
        jobs.append((bg["result"], os.path.join(comp, bg["input"]), os.path.join(comp, bg["mask"]),
                     os.path.join(comp, bg["prompt"])))
    return jobs


def run_all(run, backend, model=None, out=None, workers=4, include_bg=True):
    key = os.environ.get(KEY_ENV[backend])
    if not key:
        sys.exit(f"缺少环境变量 {KEY_ENV[backend]}")
    model = model or DEFAULT_MODEL[backend]
    tj = json.load(open(os.path.join(run, "completion", "tasks.json"), encoding="utf-8"))
    out = out or os.path.join(run, "completion", "gen")
    os.makedirs(out, exist_ok=True)
    jobs = [j for j in jobs_from_tasks(run, tj, include_bg)
            if not os.path.exists(os.path.join(out, j[0]))]
    print(f"后端 {backend}（{model}），待生成 {len(jobs)} 张", file=sys.stderr, flush=True)

    def one(job):
        name, ip, mp, pp = job
        prompt = open(pp, encoding="utf-8").read()
        res = generate(backend, Image.open(ip), Image.open(mp), prompt, model, key)
        res.save(os.path.join(out, name))
        return name

    done, failed = [], []
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        futs = {ex.submit(one, j): j[0] for j in jobs}
        for f in as_completed(futs):
            try:
                done.append(f.result())
                print(f"  完成 {len(done)}/{len(jobs)} {futs[f]}", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append({"result": futs[f], "error": str(e)[:300]})
                print(f"  失败 {futs[f]}：{e}", file=sys.stderr, flush=True)
    summary = {"backend": backend, "model": model, "gen": out, "generated": len(done),
               "failed": failed}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description="补全步骤的生图 API 后端")
    ap.add_argument("run", help="split_layers.py 的输出目录（已跑过 complete.py prepare）")
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--model", help="模型名，默认：" + "，".join(f"{k} {v}" for k, v in DEFAULT_MODEL.items()))
    ap.add_argument("--out", help="结果目录，默认 <run>/completion/gen")
    ap.add_argument("--workers", type=int, default=4, help="并发请求数，默认 4")
    ap.add_argument("--no-background", action="store_true", help="不生成背景")
    a = ap.parse_args()
    run_all(a.run, a.backend, a.model, a.out, a.workers, not a.no_background)


if __name__ == "__main__":
    main()
