# LongDocFACTScore 中文使用说明

本说明文档介绍如何在完全离线的环境中，使用 `ladfacts.py`（即 `src/longdocfactscore/ldfacts.py`）对中文摘要进行事实一致性打分。

## 1. 环境准备

1. 创建并激活 Python 环境（Python ≥ 3.8）。
2. 安装项目依赖：

   ```bash
   pip install -e .
   ```

   如果不能联网，请提前将 `nltk`、`numpy`、`sentence-transformers`、`torch`、`transformers`、`pandas` 等依赖包下载到本地后再离线安装。
   若要在命令行中查看进度条，请同时准备 `tqdm` 包。

## 2. 准备本地模型

`ldfacts.py` 需要两个模型：

- **句向量模型**：用于检索与摘要句子最相似的原文句子。推荐使用 `uer/sbert-base-chinese-nli`。
- **BART 文本生成模型**：用于计算 BARTScore。推荐使用 `fnlp/bart-large-chinese`。

请将模型提前下载到本地目录，例如：

```
models/
├── chinese-sbert/            # 解压后的 uer/sbert-base-chinese-nli
└── bart-large-chinese/       # 解压后的 fnlp/bart-large-chinese
```

如果模型目录中含有 `tokenizer_config.json` 等文件，可以直接作为 `tokenizer_name_or_path` 传入。若未单独提供分词器，可以让代码复用模型目录。

## 3. 运行示例

以下示例展示如何在 Python 脚本中调用 `LongDocFACTScore` 对中文摘要进行打分：

```python
from src.longdocfactscore.ldfacts import LongDocFACTScore

src_docs = ["……这里是原文，全是中文……"]
hyp_summaries = ["……这里是你的摘要……"]

scorer = LongDocFACTScore(
    device=None,  # 自动选择 GPU 或 CPU
    sent_model_name_or_path="models/chinese-sbert",
    bart_model_name_or_path="models/bart-large-chinese",
    bart_tokenizer_name_or_path="models/bart-large-chinese"  # 若分词器单独存放，改成对应目录
)

scores = scorer.score_src_hyp_long(src_docs, hyp_summaries)
print(scores)
```

### 参数说明

- `device`: 传入 `"cuda:0"` 可强制使用 GPU；留空或设为 `None` 时，代码会自动判断是否可用 CUDA，否则回退到 CPU。
- `sent_model_name_or_path`: 本地句向量模型目录路径。
- `bart_model_name_or_path`: 本地 BART 模型目录路径。
- `bart_tokenizer_name_or_path`: 本地分词器目录路径；若与模型共用同一目录，可保持与 `bart_model_name_or_path` 一致或留空。

## 4. 输入格式要求

- 请提供 **列表** 形式的原文 (`srcs`) 和摘要 (`hyps`)。
- 两个列表长度必须一致，每个元素为一个字符串。
- 文本可以包含中文或中英文混排，代码会自动按 `。！？` 等终止符号切分句子。

## 5. 使用 JSON 脚本批量打分

如果您的原文与缩写分别存放在两个 JSON 文件中，并且章节名完全相同（例如
`{"章节1": "原文1"}` 与 `{"章节1": "缩写1"}`），可以直接使用
`score_json.py` 脚本批量打分：

```bash
python -m src.longdocfactscore.score_json 原文.json 缩写.json \
  --sent-model models/chinese-sbert \
  --bart-model models/bart-large-chinese \
  --bart-tokenizer models/bart-large-chinese \
  --output scores.json
```

- `--device`：可选，指定运行设备，例如 `cuda:0`。缺省时自动选择。
- `--sent-model`：句向量模型目录或名称，默认 `uer/sbert-base-chinese-nli`。
- `--bart-model`：BART 模型目录或名称，默认 `fnlp/bart-large-chinese`。
- `--bart-tokenizer`：可选的分词器目录或名称，默认复用 BART 模型目录。
- `--output`：可选，将打分结果保存到 JSON 文件，包含各章节得分与平均得分。
- `--no-progress`：可选，添加该参数可关闭命令行进度条。

脚本会显示评分进度、逐个章节的得分，并在最后打印所有章节平均得分、原文平均字数以及缩写平均字数。输出到 JSON 的文件也会包含这些汇总信息。

## 6. 常见问题

1. **报错提示无法切分句子**：请确认输入文本非空，并包含中文或英文句号/叹号/问号等句子终止符。
2. **显存不足**：可将 `bart_score` 调用时的 `batch_size` 降低（默认为 4，可手动修改源代码或在调用前设置）。
3. **无法加载模型**：请检查传入路径是否正确，目录中需包含 `config.json`、`pytorch_model.bin`（或 safetensors）等必要文件。

按照以上流程配置后，即可在离线环境中对中文摘要进行 LongDocFACTScore 评估。
