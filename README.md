# Домашнє завдання 10: тестування мультиагентної системи

Це розширення мультиагентної системи з homework-8 (Supervisor → Planner → Researcher → Critic) — додано автоматизоване тестування через DeepEval.

**Що реалізовано:**

| Вимога | Реалізація |
|---|---|
| Golden Dataset (15–20 прикладів) | `tests/golden_dataset.json` — 15 прикладів: 5 happy path, 5 edge cases, 5 failure cases |
| Component tests (Planner, Researcher, Critic) | `test_planner.py` — 2 тести; `test_researcher.py` — 1 тест; `test_critic.py` — 2 тести |
| Tool correctness (мін. 3) | `test_tools.py` — 3 тести: Planner викликає пошук, Researcher використовує план, Supervisor викликає `save_report` |
| End-to-end з мін. 2 метриками | `test_e2e.py` — 3 метрики: Answer Relevancy, Correctness (GEval), Citation Presence (GEval) |
| Custom GEval метрика | `test_planner.py` — **Plan–Request Alignment**: перевіряє що план відповідає запиту користувача, а не є шаблонним |
| Обґрунтовані пороги | Пороги 0.4–0.65 встановлені на основі baseline запуску; результати в `tests/results/e2e_baseline.json` |
| Тести запускаються | `./run_tests.sh` або `DEEPEVAL_TELEMETRY_OPT_OUT=YES deepeval test run tests/` |

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
