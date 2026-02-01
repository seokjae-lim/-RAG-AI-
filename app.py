import streamlit as st

from agents import AgenticWorkflow


st.set_page_config(page_title="Agentic Stock Analyst", layout="wide")

st.title("Agentic AI Stock Analyst")
st.caption("티커 입력 → 수집 → 분석 → 검증 → 요약 리포트까지 자동 수행")

with st.sidebar:
    st.header("사용 방법")
    st.write(
        "티커를 입력하고 실행 버튼을 누르면, 에이전트가 계획-실행-검증 단계를 거쳐 리포트를 생성합니다."
    )
    st.info("데이터는 yfinance에서 가져오며 네트워크 상태에 따라 지연될 수 있습니다.")

workflow = AgenticWorkflow()


st.subheader("목표 입력")
with st.form("ticker-form"):
    ticker = st.text_input("종목 티커", value="AAPL")
    horizon_days = st.slider("분석 기간 (일)", min_value=180, max_value=730, value=365, step=30)
    run = st.form_submit_button("에이전트 실행")

if run:
    st.subheader("1) 계획 단계")
    result = workflow.run(ticker, horizon_days)
    plan_steps = result["plan"]
    for step in plan_steps:
        st.markdown(
            f"**{step.role}**: {step.goal}  \\n"
            f"- 입력: {step.inputs}  \\n"
            f"- 출력: {step.outputs}"
        )

    st.subheader("2) 실행 단계")
    events = result["events"]
    event_cols = st.columns(len(events))
    for col, event in zip(event_cols, events):
        status_icon = "✅" if event.status == "done" else "⚠️"
        col.metric(event.role, status_icon, event.message)

    history = result["history"]
    if history is not None and not history.empty:
        st.write("최근 가격 데이터")
        st.dataframe(history.tail(10), use_container_width=True)
        st.line_chart(history.set_index("Date")["Close"])
    else:
        st.warning("가격 데이터를 찾을 수 없습니다.")

    st.subheader("3) 분석 단계")
    analysis = result["analysis"]
    if analysis:
        metrics = {
            "1년 누적수익률": f"{analysis['total_return']:.2%}",
            "연환산 수익률": f"{analysis['annualized_return']:.2%}",
            "연간 변동성": f"{analysis['volatility']:.2%}",
            "샤프 지수": f"{analysis['sharpe']:.2f}",
            "RSI(14)": f"{analysis['rsi']:.2f}",
            "시그널": analysis["signal"],
        }
        cols = st.columns(3)
        for idx, (label, value) in enumerate(metrics.items()):
            cols[idx % 3].metric(label, value)
    else:
        st.info("분석 지표를 생성할 수 없습니다.")

    st.subheader("4) 검증 단계")
    validation = result["validation"]
    if validation.passed:
        st.success("검증 통과")
    else:
        st.error("검증 실패")
    for note in validation.notes:
        st.write(f"- {note}")

    st.subheader("5) 요약 리포트")
    st.markdown(result["report"])
