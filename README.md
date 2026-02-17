# transmission_telegram-bot

Telegram-бот для управления Transmission 3 через меню-кнопки.

## Установка в Linux (с systemd)

1) Установите Git (если ещё не установлен):

```bash
sudo apt update
sudo apt install -y git python3 python3-venv
```

2) Склонируйте репозиторий и перейдите в него:

```bash
git clone https://github.com/AlexeysM14/transmission_telegram-bot.git
cd transmission_telegram-bot
```

3) Запустите установку (создаст `.venv`, systemd-сервис и команду `transmission3-bot`):

```bash
sudo bash install.sh
```

4) Откройте меню настройки:

```bash
transmission3-bot update
```

В меню доступны пункты:
- `1` — скачать обновления бота из GitHub (`git pull` + обновление зависимостей);
- `2` — задать токен Telegram-бота (`TG_TOKEN`);
- `3` — задать Telegram user id (`ALLOWED_USER_IDS`);
- `4` — задать URL Transmission RPC (`TR_URL`).

5) После настройки запустите сервис:

```bash
sudo systemctl start transmission3-bot
sudo systemctl status transmission3-bot
```

## Быстрый запуск вручную (без systemd)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TG_TOKEN="<telegram-bot-token>"
python bot.py
```

## Как получить токен Telegram

1. Откройте Telegram и найдите `@BotFather`.
2. Отправьте команду `/newbot`.
3. Задайте имя бота (display name), затем username (должен оканчиваться на `bot`, например `my_transmission3_bot`).
4. BotFather отправит строку вида `123456789:AA...` — это и есть `TG_TOKEN`.
5. Введите этот токен в `transmission3-bot update` → пункт `2`.

## Как узнать свой Telegram user id

- `@userinfobot` — отправьте `/start`, бот покажет ваш `Id`.
- `@getmyid_bot` — отправьте любое сообщение, бот вернёт ваш user id.
- Через Telegram Bot API (если хотите без сторонних ботов):
  1. Напишите что-нибудь вашему боту.
  2. Выполните команду:

```bash
curl -s "https://api.telegram.org/bot<TG_TOKEN>/getUpdates"
```

  3. В ответе найдите поле `"from":{"id":...}` — это ваш user id.

- Полученный id впишите в `transmission3-bot update` → пункт `3`.

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
