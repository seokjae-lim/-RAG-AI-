from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PlanStep:
    role: str
    goal: str
    inputs: str
    outputs: str


@dataclass
class ValidationResult:
    passed: bool
    notes: List[str]


@dataclass
class AgentEvent:
    role: str
    status: str
    message: str
    timestamp: datetime


class PlannerAgent:
    def plan(self, ticker: str, horizon_days: int) -> List[PlanStep]:
        return [
            PlanStep(
                role="Planner",
                goal=f"Define scope and workflow for {ticker}.",
                inputs=f"Ticker={ticker}, Horizon={horizon_days} days",
                outputs="Plan steps and expected artifacts",
            ),
            PlanStep(
                role="Collector",
                goal="Collect historical prices, volume, and metadata.",
                inputs="Ticker, date range",
                outputs="Price history dataframe, metadata",
            ),
            PlanStep(
                role="Analyst",
                goal="Compute returns, risk, trend, and momentum indicators.",
                inputs="Price history",
                outputs="Metrics dictionary",
            ),
            PlanStep(
                role="Validator",
                goal="Check data quality and analysis sanity.",
                inputs="History + metrics",
                outputs="Validation pass/fail with notes",
            ),
            PlanStep(
                role="Reporter",
                goal="Summarize insights, risks, and next steps.",
                inputs="Metrics + validation",
                outputs="Markdown report",
            ),
        ]


class DataCollectorAgent:
    def collect(self, ticker: str, horizon_days: int) -> Dict[str, pd.DataFrame | Dict]:
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=horizon_days)
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(start=start_date, end=end_date, interval="1d")
        info = ticker_obj.fast_info
        if history.empty:
            return {"history": history, "info": info}
        history = history.reset_index()
        return {"history": history, "info": info}


class AnalystAgent:
    def analyze(self, history: pd.DataFrame) -> Dict[str, float | str]:
        if history.empty:
            return {}
        prices = history["Close"].astype(float)
        volume = history["Volume"].astype(float)
        returns = prices.pct_change().dropna()
        if returns.empty:
            return {}

        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(prices)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility else 0.0
        drawdown = (prices / prices.cummax() - 1).min()

        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_60 = prices.rolling(60).mean().iloc[-1]
        sma_120 = prices.rolling(120).mean().iloc[-1]
        trend_score = float(np.sign(sma_20 - sma_60)) if not np.isnan(sma_60) else 0.0

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not rs.empty else float("nan")

        volume_ma = volume.rolling(20).mean().iloc[-1]
        volume_trend = float(np.sign(volume.iloc[-1] - volume_ma)) if not np.isnan(volume_ma) else 0.0

        signal = "중립"
        if trend_score > 0 and rsi >= 55:
            signal = "강세"
        elif trend_score < 0 and rsi <= 45:
            signal = "약세"

        return {
            "last_close": float(prices.iloc[-1]),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
            "sma_20": float(sma_20) if not np.isnan(sma_20) else float("nan"),
            "sma_60": float(sma_60) if not np.isnan(sma_60) else float("nan"),
            "sma_120": float(sma_120) if not np.isnan(sma_120) else float("nan"),
            "rsi": float(rsi) if not np.isnan(rsi) else float("nan"),
            "volume_trend": float(volume_trend),
            "signal": signal,
        }


class ValidatorAgent:
    def validate(self, history: pd.DataFrame, analysis: Dict[str, float | str]) -> ValidationResult:
        notes: List[str] = []
        passed = True

        if history.empty:
            passed = False
            notes.append("No price history found for the given ticker.")
        if history.shape[0] < 120:
            passed = False
            notes.append("Insufficient history (need at least 120 trading days).")
        if history["Close"].isna().any():
            passed = False
            notes.append("Missing close prices detected.")
        if history["Volume"].fillna(0).sum() == 0:
            notes.append("Volume data is missing or zero for all rows.")
        if not analysis:
            passed = False
            notes.append("Analysis could not be computed.")
        if analysis.get("volatility", 0) == 0:
            notes.append("Volatility is zero or missing; check data quality.")

        latest_date: Optional[pd.Timestamp] = None
        if not history.empty:
            latest_date = pd.to_datetime(history["Date"]).max()
        if latest_date and (datetime.utcnow().date() - latest_date.date()).days > 7:
            notes.append("Latest data is older than 7 days; market data may be stale.")

        return ValidationResult(passed=passed, notes=notes)


class ReporterAgent:
    def report(
        self,
        ticker: str,
        info: Dict,
        analysis: Dict[str, float | str],
        validation: ValidationResult,
    ) -> str:
        name = info.get("shortName") or info.get("longName") or ticker
        currency = info.get("currency") or ""
        sector = info.get("sector") or "N/A"
        signal = analysis.get("signal", "중립")
        rsi = analysis.get("rsi")
        trend_label = "상승" if analysis.get("sma_20", 0) > analysis.get("sma_60", 0) else "하락"

        lines = [
            f"# {name} ({ticker})",
            "",
            "## 요약",
            f"- 시그널: **{signal}**",
            f"- 섹터: {sector}",
            f"- 통화: {currency}",
            "",
            "## 핵심 지표",
        ]

        if analysis:
            lines += [
                f"- 최근 종가: {analysis.get('last_close', 0):,.2f}",
                f"- 1년 누적수익률: {analysis.get('total_return', 0):.2%}",
                f"- 연환산 수익률: {analysis.get('annualized_return', 0):.2%}",
                f"- 연간 변동성: {analysis.get('volatility', 0):.2%}",
                f"- 샤프 지수: {analysis.get('sharpe', 0):.2f}",
                f"- 최대 낙폭: {analysis.get('max_drawdown', 0):.2%}",
                f"- RSI(14): {rsi:.2f}" if isinstance(rsi, float) else "- RSI(14): N/A",
                f"- 단기/중기 추세: {trend_label}",
            ]

        lines.append("\n## 검증 결과")
        lines.append("- 상태: 통과" if validation.passed else "- 상태: 실패")
        if validation.notes:
            for note in validation.notes:
                lines.append(f"- {note}")

        lines.append("\n## 다음 액션")
        lines.append("- 리스크 허용 범위를 고려해 분할 매수/매도를 검토하세요.")
        lines.append("- 실적 발표 일정 및 주요 뉴스 이벤트를 확인하세요.")

        return "\n".join(lines)


class AgenticWorkflow:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.collector = DataCollectorAgent()
        self.analyst = AnalystAgent()
        self.validator = ValidatorAgent()
        self.reporter = ReporterAgent()

    def run(self, ticker: str, horizon_days: int) -> Dict[str, object]:
        events: List[AgentEvent] = []

        plan = self.planner.plan(ticker, horizon_days)
        events.append(
            AgentEvent(
                role="Planner",
                status="done",
                message="Plan created.",
                timestamp=datetime.utcnow(),
            )
        )

        data_bundle = self.collector.collect(ticker, horizon_days)
        events.append(
            AgentEvent(
                role="Collector",
                status="done",
                message="Data collected.",
                timestamp=datetime.utcnow(),
            )
        )
        history = data_bundle.get("history")
        info = data_bundle.get("info")

        analysis = self.analyst.analyze(history)
        events.append(
            AgentEvent(
                role="Analyst",
                status="done" if analysis else "warning",
                message="Analysis computed." if analysis else "Analysis incomplete.",
                timestamp=datetime.utcnow(),
            )
        )

        validation = self.validator.validate(history, analysis)
        events.append(
            AgentEvent(
                role="Validator",
                status="done" if validation.passed else "warning",
                message="Validation passed." if validation.passed else "Validation failed.",
                timestamp=datetime.utcnow(),
            )
        )

        report = self.reporter.report(ticker, info or {}, analysis, validation)
        events.append(
            AgentEvent(
                role="Reporter",
                status="done",
                message="Report generated.",
                timestamp=datetime.utcnow(),
            )
        )

        return {
            "plan": plan,
            "history": history,
            "info": info,
            "analysis": analysis,
            "validation": validation,
            "report": report,
            "events": events,
        }
