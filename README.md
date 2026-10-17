# Agent 工程实践：两个可运行项目 + 可复现评测

面向 **Agent / 大模型应用开发实习**岗位的实践仓库。两个项目都满足同一套要求：

- **零第三方依赖跑通**：只用 Python 标准库，不需要 `pip install`，不需要 API Key
- **指标由脚本计算**：所有数字都能从 `eval/` 里重新跑出来，不是手写的
- **有对照组**：每个项目都跑 baseline vs 调优版，证明改进归因于策略而非运气
- **失败被显式建模**：异常分支是一等公民，不是 `try/except: pass`

## 快速开始

```bash
# Linux / macOS / Git Bash
bash run_all_evals.sh

# Windows PowerShell
.\run_all_evals.ps1
```

或者手动执行：

```bash
python lit-qa-agent/litqa/test_agent.py          # 自检
python lit-qa-agent/eval/evaluate.py             # 评测
python task-assistant/taskgraph/test_graph.py    # 自检
python task-assistant/eval/evaluate.py           # 评测
```

## 仓库结构

```
agent-works/
├── shared/                       # 两个项目共用
│   ├── llm/backend.py            # LLM 后端抽象（离线抽取式 / 真实 API 双实现）
│   └── report.py                 # 统一评测报告格式
├── lit-qa-agent/                 # 项目一：文献问答 RAG
│   ├── litqa/
│   │   ├── chunking.py           # 切分，携带稳定引用锚点
│   │   ├── retrieval.py          # BM25 + 短语覆盖混合检索
│   │   ├── agent.py              # 问答链路 + 引用回指 + 显式拒答
│   │   └── test_agent.py         # 零依赖自检
│   └── eval/
│       ├── make_corpus.py        # 合成带可验证事实的语料
│       ├── build_questions.py    # 三类问题（含资料外提问）
│       └── evaluate.py           # 引用正确率 / 拒答率 / 幻觉率
├── task-assistant/               # 项目二：多工具任务助手
│   ├── taskgraph/
│   │   ├── tools.py              # 三工具 + 四类异常 + 安全表达式求值
│   │   ├── graph.py              # 显式状态 + 条件路由 + 步数上限
│   │   ├── service.py            # FastAPI 暴露（懒加载，可选）
│   │   └── test_graph.py         # 零依赖自检（含代码注入防护）
│   └── eval/
│       ├── build_eval_set.py     # 正常 + 四类异常用例
│       └── evaluate.py           # baseline vs enhanced 对照
└── run_all_evals.sh
```

## 当前评测结果

### 项目一 · 文献问答 RAG

语料 12 篇合成文献，问题集 150 条（含 6 类资料外提问）：

| 配置 | chunk | overlap | top_k | 引用正确率 | 覆盖率 | 拒答率 | 幻觉率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-512 | 512 | 0 | 3 | 82.5% | 77.1% | 100% | 0% |
| tuned-320 | 320 | 64 | 4 | 88.4% | 84.7% | 100% | 0% |
| tuned-256 | 256 | 80 | 5 | **93.1%** | **91.0%** | 100% | 0% |

**结论与取舍**：切分粒度越小、top_k 越大，引用正确率越高（82.5% → 93.1%），
但检索开销与延迟同步上升。这不是"调参调出来的最优解"，而是在准确率与成本之间
选定的可解释平衡点。

### 项目二 · 多工具任务助手

用例集 29 条（正常 8 / 异常 21）：

| 指标 | baseline | enhanced | 说明 |
| --- | --- | --- | --- |
| 工具调用成功率 | 44.4% | **94.4%** | 可恢复用例（正常 + 瞬时故障） |
| 异常处置率 | 100% | 100% | 崩溃数均为 0 |
| 升级人工比例 | 0% | 33.3% | 结果冲突场景 |
| 平均重试次数 | 0 | 0.72 | |
| 平均耗时 | 6.2ms | 13.2ms | 增强策略的代价 |

**为什么升级率是 33.3% 而不是 100%**：3 个冲突用例中只有 1 个真正检索到
互相矛盾的数据，另外 2 个检索本身是正确且一致的 —— 对它们升级人工是
**过度保守**，属于误报。这里保留真实结果而不是凑整，因为
「哪些情况不该升级」本身是系统设计的一部分。

## 设计要点

### 引用锚点用位置而非内容哈希

```python
@property
def cite_ref(self) -> str:
    return f"{self.doc_id}#p{self.page}#c{self.chunk_index}"
```

内容哈希在文献修订后会全部失效，历史引用无法回溯。位置标识保证
**同一版本的文档内，锚点始终稳定**。这是史料溯源场景的硬需求。

### 拒答用双重闸门

只用 BM25 分数阈值不够 —— "本文""多少"这类虚词会把不相关文档的分数抬高。
本项目同时要求**问题实词在检索结果中的覆盖率**达标，两者取"或"才拒答。
宁可拒答，也不能给错出处。

领域外术语（ImageNet、BLEU 等）走单独的标记词短路，因为纯词面覆盖率
对这类幻觉无能为力。这一点在 `litqa/agent.py` 的注释里写明了局限与替换方案。

### 工具失败必须分类

笼统的"重试"是错的：除零重试一万次还是除零。

| 异常类型 | 处置 | 理由 |
| --- | --- | --- |
| 瞬时超时 | 退避重试 | 重试有意义 |
| 参数缺失 | 直接返回交上层 | 同样参数必然同样失败 |
| 检索为空 | 降级为"无结果" | 流程可继续，不该中断 |
| 结果冲突 | 升级人工 | 自动裁决的风险大于收益 |
| 确定性错误 | 明确报错 | 无效重试是纯浪费 |

### 计算器绝不 eval 原始字符串

工具入参直接 `eval()` 等于开了一个远程代码执行口子 —— 模型幻觉或
prompt 注入都可能传出 `__import__('os').system(...)`。本项目用 AST
白名单实现求值，并限制指数上限防 CPU 打爆。`test_graph.py` 里有
针对 `__import__`、`__class__` 等攻击模式的测试。

## 已知局限

诚实起见，把这套方案做不到的事写清楚：

1. **检索用的是 BM25 而非向量**。BM25 更容易暴露切分策略的缺陷，
   做对照实验更合适；但语义改写能力确实弱于 embedding。
2. **规划器是规则式的**，不是 LLM 驱动。好处是可复现、可审计；
   代价是复杂任务的拆解能力有限。接真实 LangGraph 时替换 `plan()` 即可，
   后续执行与异常处置逻辑完全复用。
3. **领域外标记词是人工词表**。换真实 embedding 后应替换为向量距离阈值，
   闸门逻辑不变。
4. **网页搜索用静态语料模拟**，不访问真实网络。评测离线可复现是刻意取舍。

## 接真实大模型

```bash
export AGENT_LLM_BACKEND=online
export AGENT_LLM_API_KEY="sk-..."
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"   # 可选
export AGENT_LLM_MODEL="gpt-4o-mini"                    # 可选
python lit-qa-agent/eval/evaluate.py
```

默认离线后端是**基线**而非摆设：任何检索增强系统都不应该弱于
"把最相关的那几段原文直接拼出来"。接上真实模型后的提升才有参考价值。
