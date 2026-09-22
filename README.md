# Black Red Line — AUDIO ANALYZER 1.0

Локальный сервис анализа музыкального трека для DIRECTOR.

## Что выдаёт
- BPM и confidence
- сетку битов
- STRONG_BEAT
- BASS_PEAK
- TRANSITION
- DROP (эвристический кандидат)
- `sync_points` — короткий список монтажных точек для DIRECTOR

## Быстрый запуск через Docker
```bash
docker build -t brl-audio-analyzer .
docker run --rm -p 8000:8000 brl-audio-analyzer
```

Проверка:
```bash
curl http://localhost:8000/health
```

Анализ:
```bash
curl -X POST "http://localhost:8000/analyze?max_sync_points=24" \
  -F "file=@track.mp3"
```

## Локальный CLI
После установки зависимостей:
```bash
python analyzer.py track.mp3 -o analysis.json
```

## Для DIRECTOR
DIRECTOR вызывает `POST /analyze`, получает JSON и использует `sync_points` как кандидаты для синхронизации. Приоритет: движение/читаемость упражнения → качество кадра → музыка.

## Важно
`DROP`/`TRANSITION` в версии 1.0 определяются эвристически по спектральному потоку, общей энергии и басовой энергии. Это не гарантированное семантическое распознавание структуры песни. Для production-версии 2.0 можно добавить отдельную модель музыкальной сегментации и VIDEO MOTION ANALYZER.
