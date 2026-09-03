# Multi-Agent Deep Research System(多智能体深度研究系统)

[English](README.md) | **中文**

一个基于 LangGraph 的多智能体流水线:接收一个研究主题,把它拆解成若干子问题,并行研究每个子问题,汇总研究结果,再通过自动化的 Writer-Critic(写作-审核)修订循环产出报告。

```
用户输入
  -> Planner(把主题拆成 3-5 个子问题)
  -> Send API(并行的 Search Agent,每个子问题一个)
       每个 agent:先用 Tavily 做网页搜索 -> 再用 LLM 提炼要点,并以 [n] 标注来源
  -> RAG Retriever(把引用重新映射为全局统一编号,把结果嵌入
       Chroma 向量库,检索出跟主题最相关的 top-k 个 chunk)
  -> Writer(撰写报告草稿,保留 [n] 引用标记 + 一份 References 参考文献列表)
  -> Critic(给草稿打分,如果不通过就打回 Writer 重写,最多 3 次;
       通过后压缩引用编号并生成最终报告)
  -> 最终报告
```

LLM 后端:[DeepSeek](https://api-docs.deepseek.com/)(`deepseek-chat`),通过兼容 OpenAI 接口的 `langchain-openai` 客户端调用。网页搜索:[Tavily](https://tavily.com/)。向量数据库:[Chroma](https://www.trychroma.com/)(每次研究运行时的内存态实例)。可选的调用链路追踪:[LangSmith](https://smith.langchain.com/)。

## 项目结构

```
research_system/
├── main.py                     # CLI 命令行入口
├── app.py                      # Streamlit 网页界面
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── research_system/
│   ├── config.py                # 环境变量加载 + get_llm() / get_search_client() / get_embeddings()
│   ├── state.py                 # ResearchState(LangGraph 的 state schema)
│   ├── graph.py                 # 构建并编译 StateGraph
│   ├── system.py                # DeepResearchSystem:运行 + 打印/保存报告的便捷封装
│   └── nodes/
│       ├── planner.py           # planner_node、dispatch_search(Send API 并行分发)
│       ├── search.py            # search_agent:Tavily 网页搜索 + LLM 提炼,标注 [n]
│       ├── rag.py                # rag_retriever_node:引用重映射 + Chroma 向量检索
│       ├── writer.py             # writer_node:撰写报告草稿,保留 [n] 引用
│       └── critic.py             # critic_node、should_revise、renumber_citations
├── eval/                        # 可选的评测系统,跑真实流水线(见 ## 评测)
│   ├── cases.py                  # 固定的评测 topic 集合
│   ├── metrics.py                # 确定性、不调 API 的引用/长度指标
│   ├── judge.py                  # LLM-as-judge 打分(覆盖度/忠实度/连贯性/引用质量)
│   ├── run_eval.py                # CLI:跑真实流水线 + 打分 + 落盘到 eval/results/
│   └── compare.py                 # CLI:两次评测结果之间的回归对比
└── tests/
    ├── conftest.py               # FakeLLM/FakeTavilySearch/FakeEmbeddings,测试不碰网络
    ├── test_critic.py            # 修订循环路由逻辑的单元测试
    ├── test_pipeline.py          # 用 fake 数据跑通整个图的端到端测试
    └── test_eval_metrics.py      # eval/metrics.py 里纯函数的单元测试
```

## 关键设计点

- **用 Send API 做并行 fan-out。** `dispatch_search` 返回一个 `list[Send]`,每个子问题对应一个;LangGraph 会并发执行它们全部。`search_results` 用 `operator.add` 作为 reducer,让每个并行分支都是往同一个列表里追加,而不是互相覆盖。
- **带硬停止的 Writer-Critic 修订循环。** Critic 返回一个结构化(Pydantic)的判定结果;如果不通过,`revision_count` 加一,控制流转回 Writer。`revision_count >= MAX_REVISIONS`(3)时强制批准,保证流程一定会终止。
- **带检查点的运行。** 图是用 `MemorySaver` 编译的,所以每次调用 `DeepResearchSystem.research()` 都会拿到自己独立的 `thread_id`,可以独立地恢复/查看运行状态。
- **有真实依据的搜索。** 每个并行的 `search_agent` 调用都会先针对自己的子问题做一次真实的 [Tavily](https://tavily.com/) 网页搜索,再让 LLM 只用这些搜索结果去提炼要点——LLM 不是凭自己的参数记忆在回答。每条事实都用带括号的数字(`[n]`)标注,指回它的来源。
- **真正的向量检索。** `rag_retriever_node` 把每条搜索结果嵌入(设了 `OPENAI_API_KEY` 就用 OpenAI 的 embedding,没设就用本地的 HuggingFace 模型)进这次研究专属的内存态 Chroma 索引,再只检索出跟主题最相关的 top-k 个 chunk——Writer 看不到全部原始结果,只看到被检索出来的那一部分。
- **引用在整条流水线里保持一致。** 各个 search agent 并行运行,各自用局部编号 `[1][2][3]` 标注来源;RAG 这一步把所有结果重新映射到一套全流水线统一的编号(`numbered_sources`)上再做嵌入,这样下游任何地方的 `[n]` 指的都是同一个来源。Writer 保留这些标记,并加上一段 `## References`;一旦 Critic 批准(或者被修订上限强制批准),`renumber_citations` 会压缩编号里的空隙,重建 `## References`,只列出最终正文里真正被引用过的来源。
- **可选的 LangSmith 调用链路追踪。** 设置 `LANGCHAIN_API_KEY` 就能给整条流水线打开完整追踪,不用改任何其他地方的代码——`config.py` 在 import 时就把相关环境变量配好了。

## 环境搭建

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt   # 包含 requirements.txt + pytest
cp .env.example .env                  # 然后填入 DEEPSEEK_API_KEY 和 TAVILY_API_KEY
```

## 使用方法

```bash
python main.py "AI Agent技术发展趋势"
# 或者用默认主题:
python main.py
```

```python
from research_system.system import DeepResearchSystem

system = DeepResearchSystem()
result = system.research("AI Agent技术发展趋势")
system.print_report(result)
system.save_report(result, "report.md")
```

或者启动 Streamlit 网页界面:

```bash
streamlit run app.py
```

## 测试

测试对 LLM 层(`research_system.config.get_llm`)、搜索层(`research_system.config.get_search_client`)、embedding 层(`research_system.config.get_embeddings`)都做了 mock,所以能离线、确定性地跑,不需要任何 API key:

```bash
pytest
```

## 评测

上面的 `pytest` 只是拿假数据检查流水线的*逻辑*——路由对不对、state 形状对不对。它没法告诉你,用真实模型、真实搜索结果跑出来的报告到底好不好、引用是否忠实,因为 `FakeLLM` 不管收到什么 prompt,永远只返回同一段固定字符串。`eval/` 是一套独立的、手动触发的评测系统,会对着一批固定的 topic(`eval/cases.py`)跑真实流水线(真实的 DeepSeek + Tavily 调用),从两个维度给每次运行打分:

**基于规则的(确定性,不调 API —— `eval/metrics.py`,离线单测覆盖在 `tests/test_eval_metrics.py`):**
- 引用合法性——报告里每个 `[n]` 都能对应到一个真实存在、编号没有越界的来源
- 参考文献列表一致性——正文引用和 `## References` 里的条目完全对得上(这同时也是对 `renumber_citations()` 在真实 LLM 输出上的一道回归检查,是纯 fake 数据的测试触及不到的)
- 引用覆盖率——检索到的来源里,有多大比例真的被引用了(这个比例有一个跟 RAG 的 top-k chunk 检索绑定的结构性天花板——见 `eval/metrics.py` 里的警示注释;更适合做前后对比,不适合当绝对分数看)
- 报告长度(字符数 + token 数,用 `tiktoken` 的 `cl100k_base` 编码当作一把统一的长度尺子,不是 DeepSeek 精确的计费 token 数)
- 修订次数——Writer-Critic 循环有没有撞到 `MAX_REVISIONS` 这个硬上限
- 墙钟延迟

**LLM-as-judge(`eval/judge.py`,结构化输出,每个维度 1-5 分):** 覆盖度(是不是回答了计划里的每个子问题)、忠实度(核对真实检索到的来源摘录,不只是看报告文本本身)、连贯性、引用是否得当。设了 `OPENAI_API_KEY` 的话,打分会用 `gpt-4o-mini` 而不是 DeepSeek,避免流水线自己的模型给自己批卷子。

### 怎么跑

```bash
python -m eval.run_eval --label baseline
python -m eval.run_eval --label baseline --cases tech-agentic-ai-zh,history-silk-road-zh
python -m eval.run_eval --label baseline --no-judge   # 跳过裁判模型这一步
```

会写出 `eval/results/<timestamp>-<label>.json`(完整数据,中文内容是可读的,没有被转义)和对应的 `.md` 汇总表格。`eval/results/` 已经加进 `.gitignore`。

### 回归对比

```bash
python -m eval.compare eval/results/<改动前>.json eval/results/<改动后>.json
```

打印一张聚合指标的差值表,加上逐 case 的裁判分差值——这是用来判断"某次 prompt/模型/RAG 的 `k` 值改动到底有没有真的起作用",而不是单凭感觉的标准工作流。警示:好几个节点的 temperature 都不是 0(writer=0.7、search=0.3、rag=0.3),所以单独一次前后对比,反映的是"改动本身的效果"加上"跑动带来的随机噪声"混在一起——如果需要一个有把握的前后对比结论,建议改动前后各跑两次。

### 迭代记录:一份真实的调试轨迹,不是一个孤立的数字

单次评测跑出来的绝对分数本身意义有限(参考上面引用覆盖率那条警示)——真正有意义的是:某一次具体的、单独隔离出来的改动,拿同样的 2 个 case 去验证,到底有没有真的让情况变好。下面是完整的迭代轨迹,包括一次事后证明是错误的尝试:

| 运行 | 改动内容 | 裁判均分(1-5) | 裁判通过率 | 撞修订上限比例 | 平均延迟(秒) |
|---|---|---|---|---|---|
| `baseline` | —— | 2.75 | 0% | 0% | 61.9 |
| `critic-rag-fix` | Critic 开始核对 `rag_context`(之前完全没有任何检索资料可核对);RAG 的 `k` 从 3 调到 6,合成上下文上限从 500 字调到 2000 字 | 3.13 | 0% | 0% | 79.5 |
| `writer-factrule` | 明确告诉 Writer 不要编造数字/案例,没数据就诚实模糊表述 | 3.50 | 50% | 0% | 52.7 |
| `critic-per-citation` | Critic 改成逐条核对 `[n]` 引用对应的真实来源摘录,不再只看一段压缩过的总结 | 3.63 | 0% | 50% | 197.8 |
| `max-revisions-5`(已撤销) | 修订上限从 3 调到 5,指望多给点机会能让分歧收敛 | 3.50 | 0% | **100%** | **386.7** |
| `graded-severity` | Critic 区分"硬编造"和"诚实的模糊表述"两种严重程度;上限改回 3 | 3.50 | 0% | 50% | 129.1 |

每一行实际发现了什么:
- **`critic-rag-fix`**:Critic 最初压根没有任何检索资料可以拿来核对草稿——它其实是在给结构和语气打分,不是在核实真实性。给它 `rag_context`,同时放大 RAG 的预算让 Writer 手里有足够真实素材可用,让引用覆盖率几乎翻倍(46%→94%),裁判分也涨了——但没能单独解决编造问题。
- **`writer-factrule`**:直接告诉 Writer 别编具体内容——没数据的时候用诚实但模糊的说法,好过编一个精确但假的——是这一整个系列里性价比最高的一次改动(第一次有 case 真正过关;裁判分单次涨幅最大)。
- **`critic-per-citation`**:Writer 的编造行为这时候已经从"凭空捏造数字"变成了"把真实数字揉在一起"(比如来源写的是 79%,报告写成了 85%)——太细微了,一段压缩过的上下文根本抓不出来。重建"引用编号→真实摘录"的对照(这份实现现在放在 `critic.py` 里,`eval/judge.py` 直接导入复用,不再各写一份)确实抓出了这个问题,但配套的"零容忍拒绝"规则造成了一个**死锁**:Critic 打分在每一轮修订里都在 4-6 分之间来回震荡,从没到过批准线,因为"专业报告"这种文体本身就会不断把 Writer 拉向写具体内容,而事实核查又会不断把这些具体内容打回去。
- **`max-revisions-5`——这次是失败的尝试**:调大修订上限,指望能给这个死锁多一点空间去收敛,听起来很合理。但结果不是:**两个** case 这次全部撞到了新的上限、被强制批准放行(这一步完全跳过了事实核查),延迟几乎翻倍,裁判分也没有任何提升。当批准门槛本身就是结构性地满足不了时,给再多次尝试也没用——这才是真正的 bug 所在。已撤销。
- **`graded-severity`**:把事实核查的问题拆成"硬伤"(编造的事实、无法追溯或被混淆的数字——依然会拦下批准)和"软伤"(诚实的模糊表述——只记下来,不拦批准)两种严重程度,打破了这个死锁:有一个 case 这次是被 Critic 真心批准的,不是撞上限强制放行的,延迟比上一轮跑快了 3 倍。但值得注意的是——这同一个 case,**独立裁判**给出的分数完全没有变化(改动前后都是 3.5)——这次改动修好的是 Critic 自身判断的一致性,不是报告本身的客观质量。"流水线自己的 Critic 认可它"和"一个独立裁判认为它足够好"之间的鸿沟依然存在——而这正是这套评测系统存在的意义:作为一道独立于流水线自身判断的检查,而不是直接相信 Critic 自己的说法。

每次运行完整的 JSON/MD(topic、草稿全文、每个裁判逐条给出的 `reasoning`)都存在 `eval/results/` 里,这个目录已经加进 `.gitignore`——git 提交历史,以及上面每次改动对应的 commit message,才是"尝试过什么、顺序是什么、为什么这么做"(包括那次撤销)的持久记录。

### 成本提示

`eval/run_eval.py` 会对每个 case 都发起真实的 DeepSeek + Tavily API 调用(设了 `OPENAI_API_KEY` 的话还会调用 OpenAI)——这会花真金白银,也要跑上几分钟,所以特意**没有**放进 `pytest`/CI 里。那些基于规则的纯函数指标本身是完全离线单测覆盖的。

## 简历文案

```
Multi-Agent Deep Research System — LangGraph, DeepSeek

- 架构设计了一套多智能体研究流水线:Planner -> 并行 Search Agent
  (LangGraph Send API fan-out,基于真实 Tavily 网页搜索)-> Chroma
  向量检索 -> Writer -> Critic 修订循环。
- 用 operator.add reducer 实现了 Send API 并行 fan-out,支持 N 个并行
  search agent 且能自动合并结果。
- 搭建了一套能扛住并行 fan-out 的引用体系:各 agent 的局部 [n] 标记
  被重新映射为全局统一编号,嵌入 Chroma 做 top-k 检索,报告定稿时
  再压缩/去重成一份 References 列表。
- 用 Pydantic 结构化输出实现了 Writer-Critic 修订循环,并设置了硬性的
  修订次数上限以保证流程一定终止。
- 所有 agent 都接入 DeepSeek 兼容 OpenAI 的 API,相比 GPT-4 级别模型
  LLM 成本降低约 95%;另外加了可选的 LangSmith 追踪和 Streamlit 界面。
- 搭建了一套离线评测系统(确定性的引用/参考文献完整性检查 + 跨供应商
  的 LLM-as-judge,覆盖度/忠实度/连贯性三个维度打分),能跑通真实流水线
  端到端,并支持前后版本的回归对比;用它驱动了 6 轮有数据支撑的迭代
  (其中一次在数据证明它让情况变差后被主动撤销),把裁判均分从 2.75
  拉到 3.50(满分 5),并让第一个 case 真正过关;过程中还挖出了流水线
  自身 Critic 和纯 fake 数据测试都没能发现的几种失败模式(编造案例、
  数字被混淆、Critic 批准死锁)。
```
