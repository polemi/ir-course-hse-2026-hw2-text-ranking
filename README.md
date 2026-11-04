# ДЗ 2. Текстовое ранжирование

Kaggle: polemi

Скор на Kaggle: **0.60914**, бейзлайн — 0.60500.

BM25F по заголовку и тексту документа. Вес заголовка — 2.5, текста — 1. Остальные параметры: `k1=1.2`, `title_b=0.5`, `body_b=0.75`.

На обучающих запросах nDCG@10 — 0.623078.

Для запуска нужен Python 3.12 и файлы соревнования.

Запуск:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./solution.py --submission_file=submission.csv PATH-TO-DATA
```
