# transmission_telegram-bot

Telegram-бот для управления Transmission 3 через меню-кнопки.

## Установка в Linux (с systemd)

Команды ниже рассчитаны на запуск от пользователя `root` (например, в Proxmox/LXC-контейнере), поэтому `sudo` не используется.
Нужен Python 3.10 или новее.

1) Установите Git (если ещё не установлен):

```bash
apt update
apt install -y git python3 python3-venv
```

2) Склонируйте репозиторий и перейдите в него:

```bash
git clone https://github.com/AlexeysM14/transmission_telegram-bot.git
cd transmission_telegram-bot
```

3) Запустите установку (создаст изолированную `.venv`, systemd-сервис и команду `transmission3-bot`):

```bash
bash install.sh
```

4) Откройте меню настройки:

```bash
transmission3-bot update
```

Проверить текущее состояние сервиса, прокси и ключевых интеграций можно командой:

```bash
transmission3-bot status
```

В меню доступны пункты:
- `1` — скачать обновления бота из GitHub, собрать новую `.venv` и перезапустить сервис;
- `2` — задать токен Telegram-бота (`TG_TOKEN`);
- `3` — задать Telegram user id (`ALLOWED_USER_IDS`);
- `4` — задать прокси для Telegram Bot API (`TG_PROXY`, например: `http://127.0.0.1:8080` или `socks5://127.0.0.1:1080`);
- `5` — задать отдельный прокси только для long polling / `getUpdates` (`TG_GET_UPDATES_PROXY`, опционально);
- `6` — задать URL Transmission RPC (`TR_URL`, например: `http://127.0.0.1:9091/transmission/rpc`);
- `7` — перезапустить systemd-сервис бота (`transmission3-bot`);
- `8` — вывести последние 10 строк файла логов ошибок бота.
- `9` — отключить основной прокси Telegram (удалить `TG_PROXY`);
- `10` — отключить отдельный прокси для `getUpdates` (удалить `TG_GET_UPDATES_PROXY`);
- `11` — разрешить доступ всем приватным чатам (`ALLOW_ALL_USERS=1`);
- `12` — снова отключить доступ всем приватным чатам (удалить `ALLOW_ALL_USERS`).
- `13` — вставить профиль `hysteria2://`, `hy2://` или `hysteria2+realm://` и настроить локальный клиент;
- `14` — остановить Hysteria 2 и удалить сохранённый профиль.

Команда `transmission3-bot status` дополнительно показывает:
- активен ли systemd-сервис `transmission3-bot`;
- настроены ли `ALLOWED_USER_IDS`, `ALLOW_ALL_USERS`, `TG_PROXY`, `TG_GET_UPDATES_PROXY` и `HYSTERIA2_SOCKS5_PROXY`;
- доступен ли Telegram Bot API через указанный прокси;
- доступен ли Transmission RPC и сколько сейчас активных/остановленных торрентов;
- где находится файл логов и когда он обновлялся в последний раз.

### Системные пути и права

При установке через `install.sh` файлы разделены по назначению:

- `/opt/transmission3-bot` — код и `.venv`, принадлежат `root:root` и недоступны сервису для записи;
- `/etc/transmission3-bot/environment` — токены и настройки, `root:transmission3-bot`, права `0640`;
- `/var/lib/transmission3-bot` — изменяемое состояние бота, принадлежит сервисному пользователю;
- `/var/log/transmission3-bot` — ротируемые файловые логи, принадлежит сервисному пользователю;
- `/usr/local/bin/transmission3-bot` — обычный root-owned launcher, а не ссылка в каталог с кодом.

systemd монтирует корневую файловую систему для процесса только для чтения и разрешает запись лишь в каталоги состояния и логов. При повторном запуске `install.sh` доверенная root-owned конфигурация `.env`, JSON-состояние и файлы логов автоматически переносятся в новые каталоги.

Конфигурация из каталога, доступного сервисному пользователю на запись, по умолчанию не импортируется: заново задайте секреты через `transmission3-bot update`. Осознанно разрешить такой одноразовый импорт можно только после проверки файла, запустив установщик с `IMPORT_UNTRUSTED_LEGACY_ENV=1`. В `/opt/transmission3-bot/.env` остаётся только root-owned совместимая ссылка на файл из `/etc`.

Для первой миграции старой установки запускайте новый `install.sh` из отдельного доверенного checkout, а не из прежнего `/opt/transmission3-bot`: в старой схеме этот каталог принадлежал сервисному пользователю.

Обновление кода требует `sudo`. CLI и установщик никогда не выполняют Git-команды, `pip`, Python или requirements из текущего `/opt/transmission3-bot`. Вместо этого создаётся новый root-owned checkout рядом с каталогом установки, Git запускается без system/global/local config и hooks, а зависимости устанавливаются только из бинарных wheels (`--only-binary=:all:`, `--no-input`) в новую `.venv`. Повторный checkout установщика сверяется с конкретным commit первого checkout; параллельные запуски установки блокируются через `flock`.

Готовый release целиком переключается через rename. Предыдущие root-owned release, CLI, unit и конфигурация сохраняются до перезапуска и проверки `systemctl is-active`; при ошибке они восстанавливаются и прежний сервис запускается снова. Если старая установка доступна сервисному пользователю на запись, она считается недоверенной: при неудачной миграции бот остаётся остановленным, а старый код не восстанавливается и не выполняется.

5) После настройки запустите сервис:

```bash
systemctl start transmission3-bot
systemctl status transmission3-bot
```

## Быстрый запуск вручную (без systemd)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Опционально: для отправки графика трафика установите `matplotlib`:

```bash
python -m pip install matplotlib
```

```bash
export TG_TOKEN="<telegram-bot-token>"
export ALLOWED_USER_IDS="<your-telegram-user-id>"
export TG_PROXY="http://proxy-login:proxy-password@127.0.0.1:8080"
# optional: separate proxy only for getUpdates long polling
# export TG_GET_UPDATES_PROXY="socks5://proxy-login:proxy-password@127.0.0.1:1080"
# optional: Hysteria 2 SOCKS5 fallback, used when the proxies above are absent
# export HYSTERIA2_SOCKS5_PROXY="socks5://127.0.0.1:1080"
export BOT_TIMEZONE="Europe/Moscow"
python bot.py
```

Если настройки Telegram-прокси и Hysteria 2 не заданы, бот работает как раньше — напрямую.

Для SOCKS-прокси зависимость уже включена в `requirements.txt`, поэтому достаточно указать URL вида `socks5://host:port`.
Если у прокси требуется авторизация, добавьте логин и пароль прямо в URL, например: `http://login:password@host:port` или `socks5://login:password@host:port`.
`mtproto://` не поддерживается: MTProto-прокси работают в Telegram-клиентах, а бот использует Telegram Bot API (HTTP/SOCKS-прокси).

### Добавление профиля Hysteria 2

Дополнительные Python-пакеты не нужны: поддержка SOCKS5 уже включена в зависимости бота. Нужен отдельный
официальный исполняемый файл `hysteria`, содержащий клиент Hysteria 2. Установите его по
[официальной инструкции](https://v2.hysteria.network/docs/getting-started/Installation/) так, чтобы он находился
в `/usr/local/bin/hysteria` или `/usr/bin/hysteria`.

После установки бинарника откройте меню:

```bash
transmission3-bot update
```

Выберите пункт `13` и вставьте полученный профиль целиком, например:

```text
hysteria2://password@example.com:443/?sni=example.com
```

CLI проверит формат ссылки, сохранит её в `/etc/transmission3-bot/hysteria2-client.json` с ограниченными
правами, создаст и включит `transmission3-bot-hysteria2.service`. Клиент поднимет SOCKS5 на
`127.0.0.1:1080`, а в окружение бота автоматически будет добавлено
`HYSTERIA2_SOCKS5_PROXY=socks5://127.0.0.1:1080`. Сам профиль в `.env` не записывается и в выводе CLI
не показывается. Если бот уже запущен, CLI перезапустит его для немедленного применения нового транспорта.

Приоритет подключения для обычных запросов: `TG_PROXY` → `HYSTERIA2_SOCKS5_PROXY` → прямое соединение.
Для long polling: `TG_GET_UPDATES_PROXY` → `TG_PROXY` → `HYSTERIA2_SOCKS5_PROXY` → прямое соединение.
Таким образом, настроенный обычный прокси всегда имеет приоритет, а Hysteria 2 используется только как fallback.
Проверить локальный порт и доступ к Telegram через него можно командой `transmission3-bot status`.

Пункт `14` останавливает и отключает созданный клиентский сервис, удаляет конфиг с профилем и убирает
fallback из окружения. Ручной режим через `HYSTERIA2_SOCKS5_PROXY` по-прежнему поддерживается, если клиент
Hysteria или Xray управляется отдельно.

## Возможности Telegram-бота

- `📊 Статус` — текущая скорость, активные скачивания, свободное место и общий трафик.
- `📈 Статистика` — трафик за день, 7 дней и месяц.
- `📚 История раздач` — список торрентов с отданным/скачанным трафиком и ratio. Записи остаются в истории после удаления торрента из Transmission.
- `📋 Торренты` — списки, поиск и быстрые действия.
- `➕ Добавить` — magnet/URL или `.torrent` файл размером до 10 MiB.
- `⚙️ Управление` — пауза, старт, удаление и уведомления.

История раздач и настройки уведомлений хранятся в SQLite в каталоге состояния. При systemd-установке это `/var/lib/transmission3-bot`; старые `traffic_anchors.json` и `torrent_history.json` импортируются автоматически. Снимок истории обновляется при обычных действиях с ботом и фоновым заданием.

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
read -rsp "Telegram bot token: " TG_TOKEN && echo
printf 'url = "https://api.telegram.org/bot%s/getUpdates"\n' "$TG_TOKEN" | curl --silent --show-error --config -
unset TG_TOKEN
```

Так токен не попадает в историю shell и аргументы процесса `curl`. После получения id отзовите токен у `@BotFather`,
если есть подозрение, что он где-либо раскрылся.

  3. В ответе найдите поле `"from":{"id":...}` — это ваш user id.

- Полученный id впишите в `transmission3-bot update` → пункт `3`.

## Переменные окружения

По умолчанию уведомления о завершении торрентов включаются автоматически для нового приватного чата с ботом (в разделе «⚙️ Управление» их можно отключить).

- `TG_TOKEN` — **обязательно**.
- `ALLOWED_USER_IDS` — список Telegram user id через запятую. По умолчанию доступ закрыт, пока не указан хотя бы один id.
- `ALLOW_ALL_USERS` — **небезопасный режим** для личных/тестовых установок: `1`, `true`, `yes` или `on` разрешает доступ любому приватному чату, если `ALLOWED_USER_IDS` пустой.
- `TG_PROXY` — **опционально**: прокси для всех запросов Telegram Bot API, например `http://127.0.0.1:8080`, `socks5://127.0.0.1:1080` или `http://login:password@127.0.0.1:8080`.
- `TG_GET_UPDATES_PROXY` — **опционально**: отдельный прокси только для long polling (`getUpdates`); если не указан, используется `TG_PROXY`, затем fallback Hysteria 2, а без них — прямое подключение. Формат тот же, включая вариант с `login:password@`.
- `HYSTERIA2_SOCKS5_PROXY` — **опционально**: URL локального SOCKS5-порта клиента Hysteria 2, например `socks5://127.0.0.1:1080`. Пункт меню `13` задаёт переменную автоматически; вручную она нужна только при внешнем управлении клиентом. Используется как fallback, когда соответствующий `TG_PROXY` не задан; поддерживаются только `socks5://` и `socks5h://`.
- `TR_URL` — полный URL подключения к Transmission RPC в явном виде, например: `http://127.0.0.1:9091/transmission/rpc` (если указан, перекрывает host/port/path).
- `TR_PROTOCOL` — `http` или `https` (по умолчанию `http`).
- `TR_HOST` — хост Transmission (по умолчанию `127.0.0.1`).
- `TR_PORT` — порт Transmission RPC (по умолчанию `9091`).
- `TR_PATH` — путь RPC (по умолчанию `/transmission/rpc`).
- `TR_USER` / `TR_PASS` — логин/пароль RPC.
- `TR_TIMEOUT` — таймаут RPC в секундах (по умолчанию `10`).
- `LIST_LIMIT` — сколько торрентов показывать в одном списке (по умолчанию `25`).
- `BOT_TIMEZONE` — часовой пояс статистики и фоновых снимков в формате IANA, например `Europe/Moscow` (по умолчанию `UTC`).
- `LOG_LEVEL` — уровень логирования в консоль (`INFO`, `DEBUG` и т.п.).
- `STATE_DIR` — каталог постоянного состояния (при systemd-установке `/var/lib/transmission3-bot`).
- `LOG_FILE` — путь к файлу логов предупреждений/ошибок (при systemd-установке `/var/log/transmission3-bot/bot-errors.log`, ротация 1 MiB × 3 файла).

## Разработка и проверки

Код поддерживает Python 3.10+. Прямые runtime-зависимости задаются в `requirements.in`, а полностью закреплённый
`requirements.txt` генерируется с SHA-256-хешами. Версии инструментов разработки находятся в `requirements-dev.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

python -m pytest -q
ruff check bot.py state_store.py transmission3-bot tests
ruff format --check bot.py state_store.py transmission3-bot tests
pyright
bandit -q -r bot.py state_store.py transmission3-bot
bash -n install.sh

# после изменения requirements.in
python -m piptools compile --generate-hashes --strip-extras requirements.in
```

CI выполняет тот же набор проверок на Python 3.10, 3.12 и 3.14, проверяет актуальность lock-файла и целостность окружения через `pip check`, а также отдельно запускает аудит известных уязвимостей runtime-зависимостей. Тесты не требуют работающего Transmission или доступа к Telegram.
