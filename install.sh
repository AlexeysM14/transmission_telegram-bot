#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AlexeysM14/transmission_telegram-bot.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/transmission3-bot}"
SERVICE_NAME="transmission3-bot"

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run as root: sudo bash install.sh"
  exit 1
fi

if ! command -v git >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git python3 python3-venv
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "Updating existing install in $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only
else
  rm -rf "$INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
chmod +x "$INSTALL_DIR/transmission3-bot"

ln -sf "$INSTALL_DIR/transmission3-bot" /usr/local/bin/transmission3-bot

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Transmission Telegram Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=-$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/.venv/bin/python $INSTALL_DIR/bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "Installed. Configure token/user id via: transmission3-bot update"
echo "Then start bot: systemctl start $SERVICE_NAME"
