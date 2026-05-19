#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AlexeysM14/transmission_telegram-bot.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/transmission3-bot}"
INSTALL_DIR="${INSTALL_DIR%/}"
if [[ -z "$INSTALL_DIR" ]]; then
  INSTALL_DIR="/"
fi
SERVICE_NAME="transmission3-bot"
SERVICE_USER="${SERVICE_USER:-transmission3-bot}"
SERVICE_GROUP="${SERVICE_GROUP:-$SERVICE_USER}"

validate_install_dir() {
  if [[ "$INSTALL_DIR" != /* ]]; then
    echo "INSTALL_DIR must be an absolute path"
    exit 1
  fi

  if [[ "$INSTALL_DIR" =~ [[:space:]] ]]; then
    echo "INSTALL_DIR must not contain whitespace"
    exit 1
  fi

  case "$INSTALL_DIR" in
    /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/opt|/proc|/root|/run|/sbin|/sys|/tmp|/usr|/usr/local|/var)
      echo "Refusing to install directly into unsafe path: $INSTALL_DIR"
      exit 1
      ;;
  esac
}

ensure_service_user() {
  if ! [[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]]; then
    echo "SERVICE_USER has an invalid system username: $SERVICE_USER"
    exit 1
  fi

  if ! [[ "$SERVICE_GROUP" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]]; then
    echo "SERVICE_GROUP has an invalid system group name: $SERVICE_GROUP"
    exit 1
  fi

  if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    groupadd --system "$SERVICE_GROUP"
  fi

  if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd \
      --system \
      --gid "$SERVICE_GROUP" \
      --home-dir "$INSTALL_DIR" \
      --shell /usr/sbin/nologin \
      "$SERVICE_USER"
  fi
}

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run as root: sudo bash install.sh"
  exit 1
fi

validate_install_dir
ensure_service_user

if ! command -v git >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git python3 python3-venv
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "Updating existing install in $INSTALL_DIR"
  git -c "safe.directory=$INSTALL_DIR" -C "$INSTALL_DIR" pull --ff-only
else
  if [[ -e "$INSTALL_DIR" ]]; then
    if [[ -d "$INSTALL_DIR" ]] && [[ -z "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "$INSTALL_DIR"
    else
      echo "Refusing to overwrite non-empty non-git directory: $INSTALL_DIR"
      exit 1
    fi
  fi
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
chmod +x "$INSTALL_DIR/transmission3-bot"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
chmod 0755 "$INSTALL_DIR"
if [[ -f "$INSTALL_DIR/.env" ]]; then
  chmod 0600 "$INSTALL_DIR/.env"
fi

ln -sf "$INSTALL_DIR/transmission3-bot" /usr/local/bin/transmission3-bot

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Transmission Telegram Bot
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=-$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/.venv/bin/python $INSTALL_DIR/bot.py
Restart=always
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=$INSTALL_DIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "Installed. Configure token/user id via: sudo transmission3-bot update"
echo "Then start bot: systemctl start $SERVICE_NAME"
