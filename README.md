# layer-split 图层拆分

把一整张画好的插画（造景贴纸、室内/室外场景）拆成分层素材：

- 每个元素一张透明底 PNG，文件名是 `编号_名称.png`，编号按 Z 字形从左上到右下
- 被挡住的部分补完整（优先用宿主 AI 自带的生图模型，没有时用本地 LaMa 兜底）
- 一个分层 PSD：最上 `完整预览`，中间 `元素/` 文件夹（每个元素一个图层），最下 `背景`（去掉所有元素后的底）

完整说明见 [SKILL.md](SKILL.md)。

## 安装

```bash
git clone https://github.com/w9718448-lab/layer-split.git
# 仓库是私有的，需要先登录 GitHub（gh auth login 或配置 token）
pip install -r layer-split/requirements.txt
# 没有生图模型、需要本地补全时再装：pip install torch
```

模型（SAM、Real-ESRGAN、LaMa）第一次运行时自动下载，也可以放到环境变量 `LAYER_SPLIT_MODELS` 指定的目录。

### 在不同 AI 环境里用

| 环境 | 用法 |
|---|---|
| Claude Code | 把整个文件夹放到 `~/.claude/skills/layer-split/`（个人）或项目里的 `.claude/skills/layer-split/` |
| Claude 网页/App | 设置里上传技能：把这个文件夹打包成 zip 上传 |
| Codex | 放到 `~/.codex/skills/layer-split/`；如果你的 Codex 版本不支持技能目录，就在 AGENTS.md 里写一句"拆图层时按 layer-split/SKILL.md 执行" |
| ChatGPT 等其他 AI | 把仓库链接或 SKILL.md 内容发给它，让它按 SKILL.md 的流程执行（需要能运行 Python 的环境） |

在有生图模型的环境（Codex、ChatGPT）里，补全步骤会把 `completion/` 下的拼图和蒙版交给生图模型，
效果比本地 LaMa 好很多。

## 命令速查

```bash
# 1 拆分（boxes.json 由 AI 看图列出，格式见 SKILL.md）
python3 -I scripts/split_layers.py 原图.png --out out --mode boxes --boxes boxes.json
# 2 补全：准备任务 → 生图模型生成 → 合并；或者直接本地 LaMa
python3 -I scripts/complete.py prepare 原图.png --run out
python3 -I scripts/complete.py apply   原图.png --run out --gen 生成结果目录
python3 -I scripts/complete.py lama    原图.png --run out
# 3 出 PSD
python3 -I scripts/build_psd.py 原图.png --run out --out 结果.psd
```
