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
  -> Edit Review(暂停,等待用户提交可选的后续修改指令;每一轮先用关键词
       匹配、匹配不到再用 RAG 检索定位到相关的段落,只改这几段,
       再打回 Writer/Critic,直到用户提交空指令结束)
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
│   ├── report_sections.py       # 编辑循环用:把报告切段、RAG 检索出相关段落、
│   │                             #   把 LLM 改完的内容拼回完整报告
│   └── nodes/
│       ├── planner.py           # planner_node、dispatch_search(Send API 并行分发)
│       ├── search.py            # search_agent:Tavily 网页搜索 + LLM 提炼,标注 [n]
│       ├── rag.py                # rag_retriever_node:引用重映射 + Chroma 向量检索
│       ├── writer.py             # writer_node:撰写/修订报告,保留 [n] 引用
│       ├── critic.py             # critic_node、should_revise、renumber_citations
│       └── edit_review.py        # edit_review_node、should_continue_editing(编辑循环)
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
    ├── test_edit_review.py       # 编辑循环路由逻辑的单元测试
    ├── test_report_sections.py   # 段落切分/检索/拼回的单元测试
    ├── test_writer.py            # writer_node 段落级编辑路径的测试(用可控的假 LLM)
    └── test_eval_metrics.py      # eval/metrics.py 里纯函数的单元测试
```

## 关键设计点

- **用 Send API 做并行 fan-out。** `dispatch_search` 返回一个 `list[Send]`,每个子问题对应一个;LangGraph 会并发执行它们全部。`search_results` 用 `operator.add` 作为 reducer,让每个并行分支都是往同一个列表里追加,而不是互相覆盖。
- **带硬停止的 Writer-Critic 修订循环。** Critic 返回一个结构化(Pydantic)的判定结果;如果不通过,`revision_count` 加一,控制流转回 Writer。`revision_count >= MAX_REVISIONS`(3)时强制批准,保证流程一定会终止。
- **带检查点的运行,结束后会自动清理。** 图是用 `MemorySaver` 编译的,所以每次调用
  `DeepResearchSystem.research()` 都会拿到自己独立的 `thread_id`,可以独立地恢复/查看运行状态。
  `MemorySaver` 不会自动过期任何数据,而一个 `DeepResearchSystem`(以及它的 checkpointer)
  会存活一整个进程的生命周期——比如 `app.py` 里每个 Streamlit `session_state` 只建一个实例——
  所以如果不管的话,每一轮运行的完整 state 会在整个会话期间一直堆在内存里。现在
  `start_research`/`resume_research` 会在某个线程不再处于中断状态(也就是没有可恢复的内容)时
  立刻删掉它的 checkpoint,已经跑完的运行不会再残留。
- **有真实依据的搜索。** 每个并行的 `search_agent` 调用都会先针对自己的子问题做一次真实的 [Tavily](https://tavily.com/) 网页搜索,再让 LLM 只用这些搜索结果去提炼要点——LLM 不是凭自己的参数记忆在回答。每条事实都用带括号的数字(`[n]`)标注,指回它的来源。
- **真正的向量检索。** `rag_retriever_node` 把每条搜索结果嵌入(设了 `OPENAI_API_KEY` 就用 OpenAI 的 embedding,没设就用本地的 HuggingFace 模型)进这次研究专属的内存态 Chroma 索引,再只检索出跟主题最相关的 top-k 个 chunk——Writer 看不到全部原始结果,只看到被检索出来的那一部分。
- **引用在整条流水线里保持一致。** 各个 search agent 并行运行,各自用局部编号 `[1][2][3]` 标注来源;RAG 这一步把所有结果重新映射到一套全流水线统一的编号(`numbered_sources`)上再做嵌入,这样下游任何地方的 `[n]` 指的都是同一个来源。Writer 保留这些标记,并加上一段 `## References`;一旦 Critic 批准(或者被修订上限强制批准),`renumber_citations` 会压缩编号里的空隙,重建 `## References`,只列出最终正文里真正被引用过的来源。
- **可选的 LangSmith 调用链路追踪。** 设置 `LANGCHAIN_API_KEY` 就能给整条流水线打开完整追踪,不用改任何其他地方的代码——`config.py` 在 import 时就把相关环境变量配好了。
- **报告通过后可以继续编辑,基于 checkpoint 状态而不是对话历史。** Critic 通过之后,图会在 `edit_review_node` 暂停(和 `human_review` 一样用 `interrupt()`/`Command(resume=...)`),等待一条可选的后续修改指令。每一轮都是从 `MemorySaver` 持久化的 state 里读当前的 `final_report`,不是靠累积一份对话记录;`edit_history` 也只保留最近 5 条原始指令,不会无限增长——这是刻意设计成"有界上下文",不是聊天日志。
- **段落级检索编辑,不是每轮重发全文。** `report_sections.py` 把报告按 `## ` 标题切段,先用一张标题同义词表做关键词匹配(修了一个手动测试中发现的真实跨语言检索失误:中文指令里提到"结论部分",单靠向量相似度检索会选错段落),匹配不到再退回 Chroma 向量检索,只把匹配到的段落交给 Writer 修改。没被选中的段落完全不经过 LLM,原样拼回,不是靠"其余部分保持原样"这句 prompt 指令碰运气。如果报告切不出段落,或者 LLM 没按格式回,会退回全文编辑兜底。

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

报告审核通过之后,CLI 和 Streamlit 界面都会提示输入一条可选的后续修改指令(以及这次编辑要涉及几段,默认 2),直接留空就结束编辑。每条指令都是基于当前报告应用修改,再交给 Critic 重新审核一遍,然后才会问下一条。

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
| `unified-rubric` | Critic 和裁判现在用完全同一套共享的 `QualityVerdict` schema 和 `build_quality_prompt` 打分(之前 Critic 用的是自己一套 1-10 分的整体打分和批准逻辑,裁判用的是这套 4 维度 1-5 分的量表) | 3.25 | 0% | 50% | 124.7 |
| `excerpt-bugfix` | 修复了一个引用编号索引 bug:摘录查找用的是 `numbered_sources[n-1]`,这是流水线**压缩编号之前**原始来源列表的下标——一旦 `renumber_citations()` 压缩/重排了 `[n]` 编号(几乎每次真实运行都会发生),这个下标对应关系就错了 | 3.25 | 0% | 50% | 104.5 |

每一行实际发现了什么:
- **`critic-rag-fix`**:Critic 最初压根没有任何检索资料可以拿来核对草稿——它其实是在给结构和语气打分,不是在核实真实性。给它 `rag_context`,同时放大 RAG 的预算让 Writer 手里有足够真实素材可用,让引用覆盖率几乎翻倍(46%→94%),裁判分也涨了——但没能单独解决编造问题。
- **`writer-factrule`**:直接告诉 Writer 别编具体内容——没数据的时候用诚实但模糊的说法,好过编一个精确但假的——是这一整个系列里性价比最高的一次改动(第一次有 case 真正过关;裁判分单次涨幅最大)。
- **`critic-per-citation`**:Writer 的编造行为这时候已经从"凭空捏造数字"变成了"把真实数字揉在一起"(比如来源写的是 79%,报告写成了 85%)——太细微了,一段压缩过的上下文根本抓不出来。重建"引用编号→真实摘录"的对照(这份实现现在放在 `critic.py` 里,`eval/judge.py` 直接导入复用,不再各写一份)确实抓出了这个问题,但配套的"零容忍拒绝"规则造成了一个**死锁**:Critic 打分在每一轮修订里都在 4-6 分之间来回震荡,从没到过批准线,因为"专业报告"这种文体本身就会不断把 Writer 拉向写具体内容,而事实核查又会不断把这些具体内容打回去。
- **`max-revisions-5`——这次是失败的尝试**:调大修订上限,指望能给这个死锁多一点空间去收敛,听起来很合理。但结果不是:**两个** case 这次全部撞到了新的上限、被强制批准放行(这一步完全跳过了事实核查),延迟几乎翻倍,裁判分也没有任何提升。当批准门槛本身就是结构性地满足不了时,给再多次尝试也没用——这才是真正的 bug 所在。已撤销。
- **`graded-severity`**:把事实核查的问题拆成"硬伤"(编造的事实、无法追溯或被混淆的数字——依然会拦下批准)和"软伤"(诚实的模糊表述——只记下来,不拦批准)两种严重程度,打破了这个死锁:有一个 case 这次是被 Critic 真心批准的,不是撞上限强制放行的,延迟比上一轮跑快了 3 倍。但值得注意的是——这同一个 case,**独立裁判**给出的分数完全没有变化(改动前后都是 3.5)——这次改动修好的是 Critic 自身判断的一致性,不是报告本身的客观质量。"流水线自己的 Critic 认可它"和"一个独立裁判认为它足够好"之间的鸿沟依然存在——而这正是这套评测系统存在的意义:作为一道独立于流水线自身判断的检查,而不是直接相信 Critic 自己的说法。
- **`unified-rubric`**:把"两边用的是不同量表"这个解释排除掉之后,鸿沟本身并没有消失。同样是那 2 个 case,Critic 这次是真心把 `history-silk-road-zh` 批准了(4.50/5 均分,不是撞上限强制放行),但独立裁判给**一模一样的最终报告**、用的还是**一模一样的 schema 和 prompt**,打出来还是 3.50/5。同一份报告、同一套打分标准,结论却不一样——这排除了"措辞/schema 不一致"这个解释,也把问题指向了更具体的方向:裁判(或者 Critic)在核对每条引用的时候,是不是压根就没在看正确的来源材料?
- **`excerpt-bugfix`**:确实如此。`build_source_excerpt_map` 把引用编号 `[n]` 对应到 `numbered_sources[n-1]`——这是流水线**原始的、压缩编号之前**的来源列表下标。`renumber_citations()` 只要报告没有引用到全部检索来源(几乎每次真实运行都是这样——`eval/results/` 里引用覆盖率从没到过 100%),就会压缩/重排 `[n]` 编号,这一压缩,原来那个下标对应关系就错了。裁判去核实最终报告里的 `[1]`,实际读到的可能是完全不相关的另一个来源的文字。修复方式是直接解析报告自己的 `## References` 段作为唯一真相源,不再信任那个可能过期的下标——在动真实评测数据之前,先用一个不调 API、纯确定性的独立复现验证过这个修复是对的。但在同样这 2 个 case 上,裁判均分这次完全没变(3.25→3.25):两个 case 的分数朝**相反方向**移动了(`history-silk-road-zh` 3.50→3.00,`tech-agentic-ai-zh` 3.00→3.50),刚好互相抵消。这次修复本身是真实的、也被独立证明是对的;只是在 n=2 的样本量下,信号还不足以让它在均值上体现出来——这也是下面这一步的直接起因。

### 把评测集从 2 个 case 扩到 6 个

连续两轮(上面的 `excerpt-bugfix`,以及一次没单独列成表格行的 `graded-severity` vs `unified-rubric` 对比)都出现了"每个 case 朝相反方向变化、互相抵消"的情况。**能不能分清"真实效果"和"跑动本身的噪声",这时候已经变成了真正的瓶颈,而不是代码本身。**于是把评测集从开发过程中一直用的那 2 个 case,扩到 `eval/cases.py` 里完整的 6 个,第一次端到端跑完整套,记作 `full6`:

| 指标 | 数值 |
|---|---|
| 裁判均分 | 3.21 |
| 裁判通过率 | **0%**(0/6) |
| 撞修订上限比例 | **100%**(6/6) |
| 平均引用覆盖率 | 56.0% |
| 平均延迟(秒) | 124.9 |

这次暴露出两件被之前 2-case 结果掩盖的事:
- **裁判通过率是 0%,而且全部 6 个都撞了修订上限。** 6 份报告没有一份是被 Critic 自己真心批准的——全都是靠撞到 `MAX_REVISIONS` 强制放行收尾的。之前 2-case 跑出来的"撞上限比例 50%",严重低估了 Critic 自己的批准标准在更广泛的话题样本上有多难被满足。
- **引用覆盖率按语言分得很干净**:3 个中文 topic 的 case 平均覆盖率 73.3%,3 个英文 topic 的 case 平均只有 38.7%,其中 `science-quantum-computing-en` 只有 12%。每一个中文 case 都比每一个英文 case 高——这是一个干净的分界,不是噪声。

根因:`planner.py`、`search.py` 的摘要合成步骤、`rag.py` 的摘要合成步骤,这三处的 prompt 全都硬编码了"用中文回答",完全不管 topic 本身是什么语言。因为 Planner 生成的子问题原文会被直接当成 Tavily 的搜索词,像"容错量子计算"这种英文 topic,实际搜索时用的还是中文查询词——检索一个主要文献都是英文的领域,召回质量自然打折扣。

**`lang-follow-topic`**:把这 4 处硬编码的"用中文回答"(Planner、search agent 摘要合成、RAG 摘要合成、Writer 最终成文格式)全部换成"用 topic 所使用的语言回答"。只重新跑了那 3 个英文 case(比跑全部 6 个便宜,也更能单独看出这个改动的效果):

| Case | 覆盖率(修复前 → 后) | 裁判分(修复前 → 后) |
|---|---|---|
| `finance-llm-investing-en` | 64% → **88%** | 3.00 → 3.50 |
| `climate-carbon-capture-en` | 40% → **64%** | 3.25 → 3.00 |
| `science-quantum-computing-en` | 12% → **60%** | 2.25 → 3.75 |
| **平均** | 38.7% → **70.7%** | 2.83 → **3.42** |

3 个 case 的覆盖率全部朝同一个方向改善——跟 `excerpt-bugfix` 里那种互相抵消的噪声模式不一样,这次是一致的、有方向性的信号。裁判通过率依然是 0%:这次改动解决的是"非中文 topic 检索质量差"这个具体问题,但没有解决 `full6` 揭示的那个更深层、依然悬而未决的问题——Critic 自己的批准标准,不管什么 topic,几乎都很难被真正满足。

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
  端到端,并支持前后版本的回归对比;用它驱动了 9 轮有数据支撑的迭代
  (其中一次在数据证明它让情况变差后被主动撤销),把裁判均分从 2.75
  拉到 3.50(满分 5),并让第一个 case 真正过关;过程中还挖出了流水线
  自身 Critic 和纯 fake 数据测试都没能发现的几种失败模式(编造案例、
  数字被混淆、Critic 批准死锁、一个把事实核查指向错误来源的引用索引 bug)。
- 在噪声较大的 2-case 回归对比开始出现"改动前后互相抵消看不出信号"的
  情况后,把评测集从 2 个扩到 6 个 case;更大的样本量揭示出一个系统性的
  语言 bug(非中文 topic 也被强制用中文检索),修复后英文 topic 的引用
  覆盖率从 38.7% 提升到 70.7%,裁判分从 2.83 提升到 3.42。
- 扩展流水线支持报告通过后的多轮人机协同编辑:复用 LangGraph 的
  interrupt()/Command(resume=...) 模式,让用户可以对已审核通过的报告
  提交后续修改指令,每一轮都基于 checkpoint 里的结构化状态、配合一个
  有界的编辑历史窗口,而不是靠累积对话记录。
- 为这套编辑功能设计了段落级 RAG 检索:把报告切成段落建 Chroma 索引,
  只对指令相关的段落做修改,没被选中的段落原样拼回、不经过 LLM,
  避免了依赖 LLM"自觉不动其他部分";手动测试中发现一次真实的跨语言
  检索失误后加了关键词同义词预判环节,过程中还顺手修复了一个
  Chroma 默认集合导致的跨调用状态污染 bug。
```
