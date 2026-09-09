import os

import streamlit as st

from research_system.system import DeepResearchSystem

st.set_page_config(
    page_title="Deep Research System",
    page_icon="\U0001F52C",
    layout="wide",
)

st.markdown(
    """
<style>
.main-title { font-size: 2rem; font-weight: 700; }
.subtitle   { margin-bottom: 2rem; }
.badge      { background: rgba(127,127,127,0.15); padding: 2px 8px;
              border-radius: 4px; font-size: 0.8rem; margin-right: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">\U0001F52C Multi-Agent Deep Research System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    '<span class="badge">LangGraph</span>'
    '<span class="badge">DeepSeek</span>'
    '<span class="badge">Tavily</span>'
    '<span class="badge">Chroma RAG</span>'
    '<span class="badge">LangSmith</span>'
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    deepseek_key = st.text_input(
        "DeepSeek API Key", type="password", value=os.getenv("DEEPSEEK_API_KEY", "")
    )
    tavily_key = st.text_input(
        "Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", "")
    )
    openai_key = st.text_input(
        "OpenAI API Key (optional, for embeddings)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Used for RAG embeddings. Without it, a local HuggingFace model is used instead.",
    )
    langsmith_key = st.text_input(
        "LangSmith Key (optional)", type="password", value=os.getenv("LANGCHAIN_API_KEY", "")
    )
    if deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "deep-research-system"

    st.markdown("---")
    st.markdown("**\U0001F5FA️ Architecture**")
    st.code(
        "Planner\n"
        "  -> Human Review (approve/edit plan)\n"
        "  -> Send API (parallel)\n"
        "     Search x N (Tavily)\n"
        "  -> RAG Retriever (Chroma)\n"
        "  -> Writer\n"
        "  -> Critic loop (max 3)\n"
        "  -> Edit Review (loop: revise / done)\n"
        "  -> Final Report",
        language="text",
    )

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("\U0001F4DD Research Topic")
    topic = st.text_area(
        "",
        placeholder="e.g. AI大模型的发展现状与未来趋势",
        height=100,
        label_visibility="collapsed",
    )

    presets = ["AI Agent技术分析", "DeepSeek技术解析", "大模型落地挑战"]
    cols = st.columns(3)
    for col, p in zip(cols, presets):
        if col.button(p, use_container_width=True):
            topic = p

    go = st.button("\U0001F680 Start Research", type="primary", use_container_width=True)

if "rs_system" not in st.session_state:
    st.session_state.rs_system = DeepResearchSystem()
if "hitl_phase" not in st.session_state:
    st.session_state.hitl_phase = "idle"  # idle | review | editing | done

with col_result:
    st.subheader("\U0001F4CA Research Report")

    if go and st.session_state.hitl_phase == "idle":
        if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            st.error("Please configure DeepSeek and Tavily API keys in the sidebar.")
        elif not topic:
            st.error("Please enter a research topic.")
        else:
            with st.spinner("\U0001F4CB Planner breaking topic into sub-questions..."):
                state = st.session_state.rs_system.start_research(topic)
            result = state["result"]
            if DeepResearchSystem.is_interrupted(result):
                payload = DeepResearchSystem.get_interrupt_payload(result)
                st.session_state.hitl_config = state["config"]
                st.session_state.hitl_plan = payload["research_plan"]
                st.session_state.hitl_topic = topic
                st.session_state.hitl_phase = "review"
            else:
                st.session_state.hitl_result = result
                st.session_state.hitl_phase = "done"
            st.rerun()

    if st.session_state.hitl_phase == "review":
        st.info("\U0001F9D1‍⚖️ Human Review: 请审核 / 编辑研究计划,再继续。")
        st.caption(f"主题: {st.session_state.hitl_topic}")
        plan_text = st.text_area(
            "每行一个子问题(可增删改)",
            value="\n".join(st.session_state.hitl_plan),
            height=160,
        )
        col_approve, col_cancel = st.columns(2)
        if col_approve.button("✅ 批准并继续", type="primary", use_container_width=True):
            edited_plan = [line.strip() for line in plan_text.split("\n") if line.strip()]
            steps = [
                "\U0001F50D Parallel search agents running (Tavily)...",
                "\U0001F4DA RAG Retriever indexing results (Chroma)...",
                "✏️ Writer drafting report...",
                "\U0001F50D Critic reviewing quality...",
            ]
            with st.spinner(" / ".join(steps)):
                state = st.session_state.rs_system.resume_research(
                    st.session_state.hitl_config, {"research_plan": edited_plan}
                )
            result = state["result"]
            if DeepResearchSystem.is_interrupted(result):
                payload = DeepResearchSystem.get_interrupt_payload(result)
                # Critic approval no longer goes straight to END -- it pauses
                # at edit_review first, so this is the next interrupt to expect.
                st.session_state.hitl_report = payload["final_report"]
                st.session_state.hitl_phase = "editing"
            else:
                st.session_state.hitl_result = result
                st.session_state.hitl_phase = "done"
            st.rerun()
        if col_cancel.button("❌ 取消", use_container_width=True):
            st.session_state.hitl_phase = "idle"
            st.rerun()

    if st.session_state.hitl_phase == "editing":
        st.success("✅ 报告已生成 / 已更新。可以继续提修改要求,或直接完成。")
        st.markdown("---")
        st.markdown(st.session_state.hitl_report)
        st.markdown("---")
        instruction = st.text_area(
            "后续修改指令(留空则完成)",
            placeholder="例如:在结论部分补充一句关于局限性的说明",
            height=100,
            key="edit_instruction_input",
        )
        edit_k = st.number_input(
            "涉及几段?", min_value=1, max_value=5, value=2, key="edit_k_input"
        )
        col_edit, col_finish = st.columns(2)
        if col_edit.button("✏️ 提交修改", type="primary", use_container_width=True):
            with st.spinner("✏️ Writer 正在根据指令修改..."):
                state = st.session_state.rs_system.resume_research(
                    st.session_state.hitl_config,
                    {"edit_instructions": instruction.strip(), "edit_k": int(edit_k)},
                )
            result = state["result"]
            if DeepResearchSystem.is_interrupted(result):
                payload = DeepResearchSystem.get_interrupt_payload(result)
                st.session_state.hitl_report = payload["final_report"]
            else:
                st.session_state.hitl_result = result
                st.session_state.hitl_phase = "done"
            st.rerun()
        if col_finish.button("✅ 完成", use_container_width=True):
            state = st.session_state.rs_system.resume_research(
                st.session_state.hitl_config, {"edit_instructions": None}
            )
            st.session_state.hitl_result = state["result"]
            st.session_state.hitl_phase = "done"
            st.rerun()

    if st.session_state.hitl_phase == "done":
        st.success("✅ Research complete!")
        result = st.session_state.hitl_result
        report = result.get("final_report", "")
        revisions = result.get("revision_count", 0)

        st.markdown(f"**Revisions:** {revisions} | **Length:** {len(report):,} chars")
        st.markdown("---")
        st.markdown(report)

        st.download_button(
            "\U0001F4E5 Download Report (Markdown)",
            data=report,
            file_name=f"research_{topic[:20].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if st.button("\U0001F504 New Research"):
            st.session_state.hitl_phase = "idle"
            st.rerun()
