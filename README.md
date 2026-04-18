# Домашнє завдання 10: тестування мультиагентної системи

Це розширення мультиагентної системи з homework-8 (Supervisor → Planner → Researcher → Critic) — додано автоматизоване тестування через DeepEval.

**Що додалося:**
- `tests/golden_dataset.json` — 15 прикладів (5 happy path, 5 edge cases, 5 failure cases)
- `tests/test_planner.py` — тести якості плану (GEval: Plan Quality, Plan–Request Alignment)
- `tests/test_researcher.py` — тест groundedness відповіді дослідника
- `tests/test_critic.py` — тести якості критики (сильний і слабкий брифи)
- `tests/test_tools.py` — 3 тести Tool Correctness (Planner, Researcher, Supervisor)
- `tests/test_e2e.py` — end-to-end evaluation на golden dataset (Answer Relevancy, Correctness, Citation Presence)
- `tests/results/e2e_baseline.json` — збережений baseline результат першого запуску

**Як запустити:**
```bash
# Активувати venv
source .venv/bin/activate

# Запустити всі тести
./run_tests.sh

# Тільки e2e (перші 3 приклади з golden dataset за замовчуванням)
DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval test run tests/test_e2e.py

# Повний golden dataset (всі 15)
E2E_FULL=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval test run tests/test_e2e.py
```
