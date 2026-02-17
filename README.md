# transmission_telegram-bot

Telegram-бот для управления Transmission 3 через меню-кнопки.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TG_TOKEN="<telegram-bot-token>"
python bot.py
```

## Переменные окружения

- `TG_TOKEN` — **обязательно**.
- `ALLOWED_USER_IDS` — список Telegram user id через запятую.
- `TR_URL` — полный URL подключения к Transmission RPC (если указан, перекрывает host/port/path).
- `TR_PROTOCOL` — `http` или `https` (по умолчанию `http`).
- `TR_HOST` — хост Transmission (по умолчанию `127.0.0.1`).
- `TR_PORT` — порт Transmission RPC (по умолчанию `9091`).
- `TR_PATH` — путь RPC (по умолчанию `/transmission/rpc`).
- `TR_USER` / `TR_PASS` — логин/пароль RPC.
- `TR_TIMEOUT` — таймаут RPC в секундах (по умолчанию `10`).
- `LIST_LIMIT` — сколько торрентов показывать в одном списке (по умолчанию `25`).
- `LOG_LEVEL` — уровень логирования (`INFO`, `DEBUG` и т.п.).

## Что улучшено относительно базового варианта

- Валидация конфигурации и более явные ошибки запуска.
- Безопасное HTML-экранирование названий торрентов и ошибок.
- Разбиение длинных сообщений Telegram на части (лимит 4096 символов).
- Более строгая обработка вложений `.torrent` и очистка временных файлов.
- Простая валидация magnet/http(s) ссылки перед добавлением.
