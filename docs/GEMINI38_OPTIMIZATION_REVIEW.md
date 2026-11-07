# Gemini 3.8 Flash — изолированная оптимизация CQA (на ревью)

**Статус:** только экспериментальный CLI. Существующий server.py, cqa/, .env, checklist, SQLite и UI не изменяются. Ветка не должна сливаться до проверки другом.

## Зачем

По переданным замерам один диалог p1-ru-001 занял 248,69 с AI-времени с двумя проходами и 6263 входными / 2540 выходными токенами. Это одно измерение, а не средняя скорость или качество. Ground Truth ещё нет.

Существующий код уже имеет JSON Schema, проверку цитат, расчёт N/A, метрики токенов и разрешает отключить review. Здесь добавлен **отдельный измерительный эксперимент**, чтобы не рисковать рабочим UI.

## Что можно сравнить

- Gemini 3.8 Flash с reasoning_effort=low и review=on/off. Системный промпт и схема ответа берутся из существующего evaluator без изменений.
- 1 и 2 одновременных **независимых диалога**. Это может сократить общее время пакета, но не скорость одного диалога; квоты и HTTP 429 необходимо проверять.
- Возобновление --resume валидных записей без повторных API-вызовов и раздельные JSON для сравнения с независимым Ground Truth.
- Код запуска по умолчанию только показывает план; требуется явное разрешение на сетевые вызовы и внешнюю обработку данных.

## Подготовка

Нужны Python 3.10+, приватный локальный data/cqa.sqlite3 и позже ключ Google в локальном .env (AI_API_KEY). Никаких ключей, транскриптов и предсказаний не добавлять в PR. Скрипт принудительно выбирает gemini-3.8-flash только внутри своего процесса; остальные настройки системы не меняются. Чтобы не показывать неверную стоимость, эксперимент не наследует неизвестные тарифы из .env.

## Windows PowerShell: из корня проекта

```powershell
# Без вызовов AI
py -3 scripts/gemini38_experiment.py --limit 3

# Базовый режим: 2 прохода, последовательно
py -3 scripts/gemini38_experiment.py --limit 3 --review on --workers 1 --output data/gemini38_baseline.json --execute --allow-cloud-data

# Один проход: сравнить качество и задержку
py -3 scripts/gemini38_experiment.py --limit 3 --review off --workers 1 --output data/gemini38_single.json --execute --allow-cloud-data

# 2 прохода + параллельно 2 разных диалога
py -3 scripts/gemini38_experiment.py --limit 3 --review on --workers 2 --output data/gemini38_parallel.json --execute --allow-cloud-data

# Продолжить прерванный вариант
py -3 scripts/gemini38_experiment.py --limit 3 --review on --workers 2 --output data/gemini38_parallel.json --resume --execute --allow-cloud-data
```

Не использовать --allow-cloud-data без разрешения на отправку финансовых диалогов во внешний Google API. По умолчанию ограничение --limit 3; для всего набора необходимо явно указать --all. При уже существующем файле результатов требуется --resume или другой --output.

## Перед принятием изменений

```powershell
py -3 -m unittest discover -s tests -v
py -3 manage.py benchmark data/gemini38_baseline.json data/ground_truth.json
py -3 manage.py benchmark data/gemini38_single.json data/ground_truth.json
```

Ground Truth нужен от независимых проверяющих, а не от той же модели. Сравнивать точность ответов, N/A, gross_error, MAE балла, JSON/evidence failures, число токенов, p50/p95, общую длительность пакета. Суммарные API-секунды и wall-clock при 2 workers — не одно и то же.

Для друга: проверь разрешение на обработку данных, доступные Gemini quota и качество KZ/RU/EN. Не менять production-конфигурацию или review=true до получения независимой разметки. API-ключ будет передан позже и вводится только локально.
