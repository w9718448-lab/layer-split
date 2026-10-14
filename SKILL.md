---
name: layer-split
description: 把一整张画好的插画（造景贴纸、室内/室外场景、素材页）拆成分层素材：每个元素一张透明底 PNG，被挡住的部分补完整，另出一个分层 PSD（最上完整预览、中间元素文件夹、最下去掉所有元素的背景）。用户说"拆图层""拆元素""分图层""出 PSD""做造景贴纸""把图里的东西都单独抠出来"时使用。
---

# 图层拆分 / 造景贴纸

画师画完一整张图，再用这个技能拆成图层。交付三样东西：

| 输出 | 说明 |
|---|---|
| `elements/编号_名称.png` | 每个元素一张，透明底，只有这个元素本身，不加白边、不套模板，被挡住的部分已补全 |
| `结果.psd` | 三部分，从上到下：`完整预览`（原图）→ `元素/` 文件夹（每个元素一个普通像素图层）→ `背景`（去掉所有元素后的底） |
| `sheet_elements_N.png` | 元素总览图，交付前自查和给用户预览用 |

编号按 Z 字形，从左上到右下；名称是识别出的元素名（中文）。

## 什么算元素，什么算背景

能拆的都拆，尽量拆细：
- 室内：背景只留建筑本身，即地板、墙面、天花板、梁、窗洞、窗外景色。家具（柜子、书架、桌椅、搁板、
  柱子、门、门帘、地毯）全部是元素，家具上面和里面的东西（瓶罐、书、布袋、挂饰）也各自是元素。
- 室外：背景只留地面、草地、水面、天空、远景；树、花丛、石头、建筑、桥、栏杆、人物动物都是元素。
- 一组完全相同、紧挨着的小东西（一排药瓶、一串小铃铛）可以算一个元素。
- 框住家具时连同上面的东西一起框，脚本会把重叠像素归给更小的元素（瓶罐归瓶罐），家具上留下的缺口在补全步骤补上。

## 依赖

```
pip install --break-system-packages onnxruntime pillow numpy scipy opencv-python-headless psd-tools
```

模型第一次运行时自动下载（GitHub releases），也可以提前放到 `/home/claude/models/` 或 `LAYER_SPLIT_MODELS` 指定的目录：
- SAM `vit_b-encoder-quant` + `vit_b-decoder-quant`（约 108MB），分割
- Real-ESRGAN ncnn `realesr-animevideov3`（约 47MB），元素放大变清晰。没有显卡要装 `mesa-vulkan-drivers`，不可用时自动退回普通放大
- 只有宿主没有生图模型时才需要：`torch` + `big-lama.pt`（约 200MB），本地补全兜底
- 只有加 `--refine` 时才需要：BiRefNet（`birefnet-lite` 约 220MB，`birefnet` 约 970MB），边缘精修

不要装 rembg（依赖 numba，会卡住）。所有脚本用 `python3 -I` 运行。

## 流程

### 1. 列元素框

用网格工具局部放大读坐标（网格上标的就是原图坐标）：
```
python3 -I scripts/grid_overlay.py 原图.png grid.png --step 50 --region x0,y0,x1,y1 --scale 3
```
密集图按 3×3 分块读。写 `boxes.json`，每个元素一个框，框略大于元素：
```json
[{"name": "书桌", "box": [1030, 822, 1343, 1066]},
 {"name": "青花瓷瓶", "box": [1100, 395, 1146, 436], "neg": [[1120, 440]]}]
```
`points` 是落在目标上的正点，`neg` 是落在不要的东西上的负点（原图坐标），用来纠正选错、粘连。名称不要重复。

### 2. 拆分（元素多时放后台跑）
```
nohup python3 -I scripts/split_layers.py 原图.png --out 输出目录 --mode boxes --boxes boxes.json > run.log 2>&1 &
```
看 `review_these.png` 和 `sheet_elements_N.png`，选错的元素加 `neg`/`points` 重跑；只剩碎片的元素删掉。
框都对了以后，想要更干净的边缘（细线、毛边、半透明边不带底色光晕）可以加 `--refine birefnet-lite` 正式拆一遍。
它只重新判定 SAM 掩码边缘一圈的像素，内部和框外不动；BiRefNet 认不出这个元素时自动退回 SAM 结果
（`manifest.json` 里 `quality.refined` 记录是否精修过）。CPU 上每个元素多 30 秒左右、内存约 7GB，有显卡会快很多。
背景简单、元素分得开的图（纯色或渐变底）可以直接 `--mode auto`，不用列框。

### 3. 补全（必做）

```
python3 -I scripts/complete.py prepare 原图.png --run 输出目录
```
自动找出有缺口的元素（缺口小于元素面积 3% 的不补），在 `输出目录/completion/` 下生成：
- `sheet_NN_input.png` / `sheet_NN_mask.png` / `sheet_NN_prompt.txt`：每 4 个元素拼成一张，蒙版白色是要补的地方
- `background_input.png` / `background_mask.png` / `background_prompt.txt`：去掉所有元素后的背景
- `tasks.json`：清单，`gen_calls` 是需要的生图次数

**有生图模型时（Codex、ChatGPT 等自带的图像生成/编辑）**：每张 sheet 和背景各调用一次局部重绘
（输入图 + 蒙版 + 提示词），结果按 `tasks.json` 里的 `result` 文件名（`sheet_NN.png`、`background.png`）存进一个目录，然后：
```
python3 -I scripts/complete.py apply 原图.png --run 输出目录 --gen 生成结果目录
```
**宿主没有生图模型、或者要整批无人值守跑时**，让脚本直接调生图 API（Key 放环境变量）：
```
GEMINI_API_KEY=... python3 -I scripts/complete.py api 原图.png --run 输出目录 --backend gemini
```
一条命令做完 prepare → 生成 → apply。`--backend` 可选：
- `gemini`：Gemini 图像模型，默认 `gemini-nano-banana-2.1`，`GEMINI_API_KEY`
- `openai`：gpt-image 图像编辑（原生支持蒙版），默认 `gpt-image-2`，`OPENAI_API_KEY`
- `fal`：fal.ai 局部重绘，默认 `fal-ai/flux-pro/v1/fill`，`FAL_KEY`

`--model` 换模型，`--workers` 设并发（默认 4）。结果存在 `输出目录/completion/gen/`，已有的结果会跳过，
中途失败重跑同一条命令只补没生成的。也可以只跑生成这一步：`python3 -I scripts/gen_api.py 输出目录 --backend gemini`，
再用上面的 `apply --gen 输出目录/completion/gen` 合并。

**都没有时**用本地 LaMa 兜底（只补得好小缺口，大块缺失会糊）：
```
python3 -I scripts/complete.py lama 原图.png --run 输出目录
```
合并时原来看得见的像素原样保留，只加入补出来的部分；补全前的元素留在 `elements_original/`。

### 4. 出 PSD
```
python3 -I scripts/build_psd.py 原图.png --run 输出目录 --out 结果.psd
```
画布默认是元素放大倍数（2 倍）。文件太大要发送时加 `--scale 1`。

### 5. 交付
打包 `elements/`、PSD、总览图成 zip 发给用户，再单独发一两页 `sheet_elements_N.png` 预览。

## 轻量化（生图次数多时）

- 只补真有缺口的元素（`--min-hole`，默认 0.03），没被挡住的元素不进生图
- 默认 4 个元素拼一张图一次生成（`--pack 4`），200 个元素、40 个要补时只需 10 次 + 背景 1 次；
  生图模型支持更大图时可以 `--pack 9`。拼图效果不好再用 `--pack 1` 逐个补
- 背景只生成一次
- 先用 `--no-hires` 快速拆一遍看框对不对，确认后再正式拆

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `split_layers.py --upscale` | 2 | 元素放大倍数 1 到 4 |
| `split_layers.py --no-hires` | 关 | 整图只编码一次，快很多，边缘稍粗，用于预览 |
| `split_layers.py --no-exclusive` | 关 | 不做像素归属（默认重叠像素归更小的元素） |
| `split_layers.py --sticker` | 关 | 额外出白边贴纸版，用户明确要才加 |
| `split_layers.py --refine` | none | `birefnet-lite` / `birefnet`：用 BiRefNet 精修边缘 |
| `complete.py api --backend` | — | `gemini` / `openai` / `fal`：脚本直接调生图 API 补全 |
| `complete.py --pack` | 4 | 几个元素拼一张图交给生图模型 |
| `complete.py --min-hole` | 0.03 | 缺口占比低于此值不补 |
| `build_psd.py --scale` | 元素倍数 | PSD 画布相对原图的倍数 |
| `build_psd.py --no-base` | 关 | 不放最上层完整预览 |

## 耗时参考（2 核 CPU，无显卡）

- 拆分：每个元素约 6 秒（200 个约 25 分钟），超分每个约 1.5 秒
- LaMa 兜底补全：每个元素约 1 秒，背景约 1 分钟
- `--refine birefnet-lite`：每个元素多约 30 秒（BiRefNet 固定按 1024 输入推理）
- PSD：200 层约 1 分钟，2 倍画布约 50MB

## 已知限制

- 补全是模型推测，不是原图内容。生图模型效果远好于本地 LaMa；LaMa 只适合小缺口，背景大面积会糊。
- SAM 在颜色接近的遮挡边界会带进一点邻居，用负点修。半透明物体（玻璃、水）可能被当成实心。
- 量化 SAM 解码器的整图 `masks` 输出不可用，脚本用 256×256 低分辨率输出自己放大，不要改回。
- 删除命令可能被安全检查拦截，输出目录用新名字。
