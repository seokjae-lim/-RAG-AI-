 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..08252c7aa3ab9bbfbbe2df0609f92377128adbe7
--- /dev/null
+++ b/README.md
@@ -0,0 +1,19 @@
+# Agentic AI Stock Analyst
+
+사용자가 종목(티커)만 입력하면 **데이터 수집 → 분석 → 검증 → 요약 리포트**까지 자동으로 수행하는 에이전트형 앱입니다.
+
+## 주요 특징
+- **계획 → 실행 → 검증** 단계 분리 및 에이전트 역할 표시
+- 역할 기반 에이전트(Planner/Collector/Analyst/Validator/Reporter) 구조
+- 1년 가격 데이터 기반 수익률, 변동성, RSI, 추세 지표 계산
+- 검증 단계에서 데이터 신뢰도와 최신성 체크
+
+## 실행 방법
+```bash
+pip install -r requirements.txt
+streamlit run app.py
+```
+
+## 참고
+- 데이터는 `yfinance`를 통해 수집됩니다.
+- 거래일 수가 부족하거나 데이터가 오래된 경우 검증 단계에서 경고가 표시됩니다.
 
EOF
)
