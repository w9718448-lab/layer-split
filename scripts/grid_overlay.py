#!/usr/bin/env python3
"""在图上叠加带坐标标注的网格，方便读出元素的像素框；支持只看某一块区域并放大。

用法
  python grid_overlay.py IMG OUT.png [--step 100] [--region x0,y0,x1,y1] [--scale 2]

--region 指定后，网格标注的仍是原图坐标，所以读出来的数可以直接写进 boxes.json。
"""
import argparse

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size):
    import os
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--region", help="x0,y0,x1,y1（原图坐标）")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()

    im = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = 0, 0, W, H
    if a.region:
        x0, y0, x1, y1 = [int(float(v)) for v in a.region.split(",")]
    im = im.crop((x0, y0, x1, y1))
    s = a.scale
    if s != 1.0:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    f = font(max(11, int(12 * max(s, 1))))
    st = a.step

    gx = (x0 // st + 1) * st if x0 % st else x0
    while gx < x1:
        px = (gx - x0) * s
        d.line([(px, 0), (px, im.height)], fill=(255, 0, 80, 110), width=1)
        d.text((px + 3, 2), str(gx), fill=(255, 0, 80, 255), font=f)
        gx += st
    gy = (y0 // st + 1) * st if y0 % st else y0
    while gy < y1:
        py = (gy - y0) * s
        d.line([(0, py), (im.width, py)], fill=(0, 90, 255, 110), width=1)
        d.text((3, py + 2), str(gy), fill=(0, 90, 255, 255), font=f)
        gy += st
    im.save(a.out)
    print(f"{a.out} 原图 {W}x{H}，区域 ({x0},{y0})-({x1},{y1})，输出 {im.width}x{im.height}")


if __name__ == "__main__":
    main()
