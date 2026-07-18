#!/usr/bin/env bash
set -Eeuo pipefail
umask 0022

REPO_URL="${REPO_URL:-https://github.com/AlexeysM14/transmission_telegram-bot.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/transmission3-bot}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

SERVICE_NAME="transmission3-bot"
INSTALL_MARKER_NAME=".${SERVICE_NAME}.installation"
SERVICE_USER="${SERVICE_USER:-transmission3-bot}"
SERVICE_GROUP="${SERVICE_GROUP:-$SERVICE_USER}"
START_SERVICE="${START_SERVICE:-auto}"
IMPORT_UNTRUSTED_LEGACY_ENV="${IMPORT_UNTRUSTED_LEGACY_ENV:-0}"
CONFIG_DIR="/etc/$SERVICE_NAME"
ENV_FILE="$CONFIG_DIR/environment"
INSTALL_CONFIG="$CONFIG_DIR/install.conf"
STATE_DIR="/var/lib/$SERVICE_NAME"
LOG_DIR="/var/log/$SERVICE_NAME"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CLI_PATH="/usr/local/bin/$SERVICE_NAME"
CLI_IMPL_DIR="/usr/local/libexec"
CLI_IMPL_PATH="$CLI_IMPL_DIR/$SERVICE_NAME"
INSTALL_PARENT="$(dirname "$INSTALL_DIR")"
LOCK_DIR="/run/lock"
LOCK_FILE="$LOCK_DIR/${SERVICE_NAME}.install.lock"
LOCK_FD=9

BUILD_ROOT=""
STAGED_RELEASE=""
PREVIOUS_RELEASE="$INSTALL_PARENT/.${SERVICE_NAME}.previous.$$"
FAILED_RELEASE="$INSTALL_PARENT/.${SERVICE_NAME}.failed.$$"
CLI_BACKUP="${CLI_PATH}.previous.$$"
CLI_IMPL_BACKUP="${CLI_IMPL_PATH}.previous.$$"
ENV_BACKUP="$CONFIG_DIR/.environment.previous.$$"
INSTALL_CONFIG_BACKUP="$CONFIG_DIR/.install.conf.previous.$$"
UNIT_BACKUP="/etc/systemd/system/.${SERVICE_NAME}.service.previous.$$"

CURRENT_PRESENT=0
CURRENT_TRUSTED=0
DEPLOYMENT_ASSESSED=0
SERVICE_WAS_ACTIVE=0
SERVICE_WAS_ENABLED=0
SHOULD_START=0
UNTRUSTED_SERVICE_STOPPED=0
TRANSACTION_STARTED=0
RELEASE_SWAPPED=0
CLI_HAD_PREVIOUS=0
CLI_IMPL_HAD_PREVIOUS=0
ENV_HAD_PREVIOUS=0
INSTALL_CONFIG_HAD_PREVIOUS=0
UNIT_HAD_PREVIOUS=0
CONFIG_TOUCHED=0
CLI_TOUCHED=0
UNIT_TOUCHED=0
DEPLOY_COMMITTED=0
DEPLOYMENT_LOCKED=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

canonicalize_path() {
  local path="$1"
  local ancestor="$path"
  local suffix=""
  local component
  local resolved

  while [[ ! -d "$ancestor" ]]; do
    [[ "$ancestor" != "/" ]] || return 1
    component="${ancestor##*/}"
    suffix="/$component$suffix"
    ancestor="${ancestor%/*}"
    [[ -n "$ancestor" ]] || ancestor="/"
  done

  resolved="$(cd -P -- "$ancestor" 2>/dev/null && pwd -P)" || return 1
  if [[ -z "$suffix" ]]; then
    printf '%s\n' "$resolved"
  elif [[ "$resolved" == "/" ]]; then
    printf '%s\n' "$suffix"
  else
    printf '%s%s\n' "$resolved" "$suffix"
  fi
}

is_protected_install_path() {
  case "$1" in
    /|/bin|/bin/*|/boot|/boot/*|/dev|/dev/*|/etc|/etc/*)
      return 0
      ;;
    /home|/home/*|/lib|/lib/*|/lib32|/lib32/*|/lib64|/lib64/*)
      return 0
      ;;
    /media|/mnt|/opt|/proc|/proc/*|/root|/root/*|/run|/run/*)
      return 0
      ;;
    /sbin|/sbin/*|/srv|/sys|/sys/*|/tmp|/tmp/*|/usr|/usr/*|/var|/var/*)
      return 0
      ;;
  esac
  return 1
}

has_install_marker() {
  local marker="$1/$INSTALL_MARKER_NAME"
  [[ -f "$marker" && ! -L "$marker" ]] || return 1
  [[ "$(<"$marker")" == "$SERVICE_NAME" ]]
}

looks_like_legacy_installation() {
  local root="$1"
  local path

  for path in bot.py requirements.txt transmission3-bot; do
    [[ -f "$root/$path" && ! -L "$root/$path" ]] || return 1
  done
  [[ -d "$root/.venv" || -d "$root/.git" ]] || return 1
  /usr/bin/grep -Fq 'from transmission_rpc import' "$root/bot.py" || return 1
  /usr/bin/grep -Fq 'transmission-rpc' "$root/requirements.txt" || return 1
  /usr/bin/grep -Fq 'SERVICE_NAME = "transmission3-bot"' "$root/transmission3-bot"
}

is_recognized_installation() {
  local root="$1"
  has_install_marker "$root" || looks_like_legacy_installation "$root"
}

lookup_user_id() {
  /usr/bin/id -u "$1" 2>/dev/null
}

lookup_group_id() {
  local entry
  local group_id

  if [[ -x /usr/bin/getent ]]; then
    entry="$(/usr/bin/getent group "$1" 2>/dev/null)" || return 1
    IFS=: read -r _ _ group_id _ <<< "$entry"
  elif [[ -x /usr/bin/dscl ]]; then
    entry="$(/usr/bin/dscl . -read "/Groups/$1" PrimaryGroupID 2>/dev/null)" || return 1
    group_id="${entry##* }"
  else
    return 1
  fi

  [[ "$group_id" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$group_id"
}

validate_service_identity() {
  local user_id
  local group_id

  if user_id="$(lookup_user_id "$SERVICE_USER")"; then
    [[ "$user_id" != 0 ]] || die "SERVICE_USER must not resolve to UID 0: $SERVICE_USER"
  fi
  if group_id="$(lookup_group_id "$SERVICE_GROUP")"; then
    [[ "$group_id" != 0 ]] || die "SERVICE_GROUP must not resolve to GID 0: $SERVICE_GROUP"
  fi
}

validate_inputs() {
  local repository_path
  local resolved_repository
  local resolved_install

  [[ -n "$INSTALL_DIR" ]] || die "INSTALL_DIR must not be empty"
  [[ "$INSTALL_DIR" == /* ]] || die "INSTALL_DIR must be an absolute path"
  [[ ! "$INSTALL_DIR" =~ [[:space:]] ]] || die "INSTALL_DIR must not contain whitespace"
  [[ "$INSTALL_DIR" == "/" || "$INSTALL_DIR" != */ ]] || die "INSTALL_DIR must not have a trailing slash"
  [[ "$INSTALL_DIR" != *//* ]] || die "INSTALL_DIR must not contain duplicate slashes"
  case "$INSTALL_DIR" in
    */./*|*/.|*/../*|*/..)
      die "INSTALL_DIR must not contain '.' or '..' path components"
      ;;
  esac
  [[ "$REPO_URL" != *$'\n'* && "$REPO_URL" != *$'\r'* ]] || die "REPO_URL must be one line"
  if [[ -n "$EXPECTED_COMMIT" ]]; then
    [[ "$EXPECTED_COMMIT" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] \
      || die "EXPECTED_COMMIT must be a lowercase 40- or 64-character hexadecimal Git object id"
  fi

  resolved_install="$(canonicalize_path "$INSTALL_DIR")" || die "Unable to resolve INSTALL_DIR: $INSTALL_DIR"
  [[ "$resolved_install" == "$INSTALL_DIR" ]] || die "INSTALL_DIR must be canonical: $INSTALL_DIR resolves to $resolved_install"
  is_protected_install_path "$INSTALL_DIR" && die "Refusing to install into protected path: $INSTALL_DIR"

  if [[ -e "$INSTALL_DIR" || -L "$INSTALL_DIR" ]]; then
    [[ -d "$INSTALL_DIR" && ! -L "$INSTALL_DIR" ]] || die "Existing INSTALL_DIR must be a directory, not a file or symlink"
    is_recognized_installation "$INSTALL_DIR" || die "Existing INSTALL_DIR is not a recognized $SERVICE_NAME installation"
  fi

  [[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]] || die "Invalid SERVICE_USER: $SERVICE_USER"
  [[ "$SERVICE_GROUP" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]] || die "Invalid SERVICE_GROUP: $SERVICE_GROUP"
  [[ "$START_SERVICE" == auto || "$START_SERVICE" == 0 || "$START_SERVICE" == 1 ]] || die "START_SERVICE must be auto, 0, or 1"
  [[ "$IMPORT_UNTRUSTED_LEGACY_ENV" == 0 || "$IMPORT_UNTRUSTED_LEGACY_ENV" == 1 ]] \
    || die "IMPORT_UNTRUSTED_LEGACY_ENV must be 0 or 1"
  validate_service_identity

  repository_path="$REPO_URL"
  if [[ "$repository_path" == file://* ]]; then
    repository_path="${repository_path#file://}"
  fi
  if [[ "$repository_path" == /* && -e "$repository_path" ]]; then
    resolved_repository="$(canonicalize_path "$repository_path")" || die "Unable to resolve local REPO_URL"
    case "$resolved_repository" in
      "$resolved_install"|"$resolved_install"/*)
        die "REPO_URL must not use the existing installation tree"
        ;;
    esac
  fi
}

sanitize_build_environment() {
  local variable

  while IFS='=' read -r variable _; do
    case "$variable" in
      PYTHON*|PIP_*|GIT_*|LD_*|DYLD_*|APT_*|DPKG_*|BASH_ENV|ENV|CDPATH|IFS)
        unset "$variable"
        ;;
      HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|http_proxy|https_proxy|all_proxy|no_proxy)
        unset "$variable"
        ;;
      REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|SSL_CERT_FILE|SSL_CERT_DIR|SSLKEYLOGFILE)
        unset "$variable"
        ;;
      SHELLOPTS|BASHOPTS)
        export -n "$variable" 2>/dev/null || true
        ;;
    esac
  done < <(/usr/bin/env)
  unset BASH_ENV ENV CDPATH IFS
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
  unset REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR SSLKEYLOGFILE
  unset LD_PRELOAD LD_LIBRARY_PATH DYLD_INSERT_LIBRARIES DYLD_LIBRARY_PATH
  export -n SHELLOPTS BASHOPTS 2>/dev/null || true
  export -n IMPORT_UNTRUSTED_LEGACY_ENV 2>/dev/null || true
  export PATH=/usr/sbin:/usr/bin:/sbin:/bin
  export PYTHONNOUSERSITE=1
  export PYTHONSAFEPATH=1
  export PIP_CONFIG_FILE=/dev/null
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
}

flock_is_available() {
  [[ -x /usr/bin/flock ]]
}

flock_cmd() {
  /usr/bin/flock "$@"
}

is_secure_lock_directory() {
  local path="$1"

  [[ -d "$path" && ! -L "$path" ]] || return 1
  [[ -z "$(find "$path" -maxdepth 0 ! -user root -print -quit)" ]] || return 1
  if [[ -n "$(find "$path" -maxdepth 0 -perm /022 -print -quit)" ]]; then
    [[ -n "$(find "$path" -maxdepth 0 -perm -1000 -print -quit)" ]] || return 1
  fi
}

acquire_deployment_lock() {
  [[ $DEPLOYMENT_LOCKED -eq 0 ]] || return 0
  flock_is_available || die "/usr/bin/flock is required; install the util-linux package first"
  is_secure_lock_directory "$LOCK_DIR" || die "Unsafe deployment lock directory: $LOCK_DIR"

  if [[ ! -e "$LOCK_FILE" && ! -L "$LOCK_FILE" ]]; then
    if ! (umask 0077; set -o noclobber; : > "$LOCK_FILE") 2>/dev/null; then
      [[ -e "$LOCK_FILE" || -L "$LOCK_FILE" ]] || die "Unable to create deployment lock: $LOCK_FILE"
    fi
  fi
  is_secure_regular_file "$LOCK_FILE" || die "Unsafe deployment lock file: $LOCK_FILE"

  exec 9>> "$LOCK_FILE" || die "Unable to open deployment lock: $LOCK_FILE"
  if ! flock_cmd -n "$LOCK_FD"; then
    exec 9>&-
    die "Another $SERVICE_NAME installation or update is already running"
  fi
  DEPLOYMENT_LOCKED=1
}

release_deployment_lock() {
  [[ $DEPLOYMENT_LOCKED -eq 1 ]] || return 0
  flock_cmd -u "$LOCK_FD" >/dev/null 2>&1 || true
  exec 9>&-
  DEPLOYMENT_LOCKED=0
}

ensure_system_dependencies() {
  if [[ ! -x /usr/bin/git ]] || [[ ! -x /usr/bin/python3 ]] || ! /usr/bin/python3 -I -m venv --help >/dev/null 2>&1; then
    /usr/bin/apt-get update
    /usr/bin/apt-get install -y git python3 python3-venv
  fi
  [[ -x /usr/bin/git ]] || die "/usr/bin/git is required"
  [[ -x /usr/bin/python3 ]] || die "/usr/bin/python3 is required"
  /usr/bin/python3 -I -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "Python 3.10 or newer is required"
  [[ -x /usr/bin/systemctl ]] || die "/usr/bin/systemctl is required"
}

ensure_service_user() {
  if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    groupadd --system "$SERVICE_GROUP"
  fi
  if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd \
      --system \
      --gid "$SERVICE_GROUP" \
      --home-dir "$STATE_DIR" \
      --shell /usr/sbin/nologin \
      "$SERVICE_USER"
  fi

  local user_id
  local group_id
  user_id="$(lookup_user_id "$SERVICE_USER")" || die "Unable to resolve SERVICE_USER after creation: $SERVICE_USER"
  group_id="$(lookup_group_id "$SERVICE_GROUP")" || die "Unable to resolve SERVICE_GROUP after creation: $SERVICE_GROUP"
  [[ "$user_id" != 0 ]] || die "SERVICE_USER must not resolve to UID 0: $SERVICE_USER"
  [[ "$group_id" != 0 ]] || die "SERVICE_GROUP must not resolve to GID 0: $SERVICE_GROUP"
}

systemctl_cmd() {
  /usr/bin/systemctl "$@"
}

start_service_and_check() {
  systemctl_cmd start "$SERVICE_NAME"
  sleep 2
  systemctl_cmd is-active --quiet "$SERVICE_NAME"
}

is_secure_regular_file() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || return 1
  [[ -z "$(find "$path" -maxdepth 0 \( ! -user root -o -perm /022 \) -print -quit)" ]]
}

is_trusted_tree() {
  local root="$1"
  [[ -d "$root" && ! -L "$root" ]] || return 1
  [[ -z "$(find "$root" -xdev ! -user root -print -quit)" ]] || return 1
  [[ -z "$(find "$root" -xdev ! -type l -perm /022 -print -quit)" ]]
}

is_secure_parent_directory() {
  local path="$1"
  [[ -d "$path" && ! -L "$path" ]] || return 1
  [[ -z "$(find "$path" -maxdepth 0 \( ! -user root -o -perm /022 \) -print -quit)" ]]
}

is_secure_directory_ancestry() {
  local path="$1"

  while true; do
    is_secure_parent_directory "$path" || return 1
    [[ "$path" != "/" ]] || return 0
    path="${path%/*}"
    [[ -n "$path" ]] || path="/"
  done
}

assess_current_deployment() {
  if [[ -e "$INSTALL_DIR" || -L "$INSTALL_DIR" ]]; then
    CURRENT_PRESENT=1
  else
    return
  fi

  if ! is_trusted_tree "$INSTALL_DIR"; then
    return
  fi
  if [[ -e "$CLI_PATH" || -L "$CLI_PATH" ]]; then
    is_secure_regular_file "$CLI_PATH" || return
  fi
  if [[ -e "$CLI_IMPL_PATH" || -L "$CLI_IMPL_PATH" ]]; then
    is_secure_regular_file "$CLI_IMPL_PATH" || return
  fi
  if [[ -e "$UNIT_FILE" || -L "$UNIT_FILE" ]]; then
    is_secure_regular_file "$UNIT_FILE" || return
  fi
  CURRENT_TRUSTED=1
}

run_clean_git() {
  PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG=/dev/null \
    /usr/bin/git \
      -c core.hooksPath=/dev/null \
      -c core.fsmonitor=false \
      -c protocol.ext.allow=never \
      "$@"
}

harden_release() {
  local release="$1"
  chown -hR root:root "$release"
  chmod -R u=rwX,go=rX "$release"
  if [[ -d "$release/.git" ]]; then
    chmod -R u+rwX,go-rwx "$release/.git"
  fi
}

write_install_marker() {
  local release="$1"
  local marker="$release/$INSTALL_MARKER_NAME"

  rm -f -- "$marker"
  printf '%s\n' "$SERVICE_NAME" > "$marker"
  chown root:root "$marker"
  chmod 0644 "$marker"
}

build_clean_release() {
  local actual_commit

  BUILD_ROOT="$(mktemp -d "$INSTALL_PARENT/.${SERVICE_NAME}.build.XXXXXX")"
  STAGED_RELEASE="$BUILD_ROOT/release"

  echo "Building a clean release from the configured repository"
  run_clean_git clone --quiet --depth 1 --no-hardlinks -- "$REPO_URL" "$STAGED_RELEASE"
  if [[ -n "$EXPECTED_COMMIT" ]]; then
    actual_commit="$(run_clean_git -C "$STAGED_RELEASE" rev-parse --verify 'HEAD^{commit}')" \
      || die "Unable to resolve cloned Git commit"
    [[ "$actual_commit" == "$EXPECTED_COMMIT" ]] \
      || die "Repository changed during update: expected $EXPECTED_COMMIT, cloned $actual_commit"
  fi

  /usr/bin/python3 -I -m venv "$STAGED_RELEASE/.venv"
  PIP_ONLY_BINARY=:all: \
    "$STAGED_RELEASE/.venv/bin/python" -I -m pip install \
      --disable-pip-version-check \
      --no-input \
      --only-binary=:all: \
      --require-hashes \
      -r "$STAGED_RELEASE/requirements.txt"
  "$STAGED_RELEASE/.venv/bin/python" -I -m pip check
  # Updates always build a new venv, so packaging tools are unnecessary at runtime.
  "$STAGED_RELEASE/.venv/bin/python" -I -m pip uninstall --yes setuptools wheel
  "$STAGED_RELEASE/.venv/bin/python" -I -m pip check
  "$STAGED_RELEASE/.venv/bin/python" -I -m pip uninstall --yes pip

  ln -s "$ENV_FILE" "$STAGED_RELEASE/.env"
  chown -h root:root "$STAGED_RELEASE/.env"
  write_install_marker "$STAGED_RELEASE"
  harden_release "$STAGED_RELEASE"
  is_trusted_tree "$STAGED_RELEASE" || die "Fresh release did not pass ownership/permission validation"
  is_secure_regular_file "$STAGED_RELEASE/transmission3-bot" || die "Fresh release has an unsafe CLI source"
  is_secure_regular_file "$STAGED_RELEASE/requirements.txt" || die "Fresh release has unsafe requirements"
}

ensure_secure_config_directory() {
  if [[ -e "$CONFIG_DIR" || -L "$CONFIG_DIR" ]]; then
    is_secure_parent_directory "$CONFIG_DIR" || die "Unsafe configuration directory: $CONFIG_DIR"
  else
    install -d -o root -g "$SERVICE_GROUP" -m 0750 "$CONFIG_DIR"
  fi
  chown root:"$SERVICE_GROUP" "$CONFIG_DIR"
  chmod 0750 "$CONFIG_DIR"
}

backup_transaction_files() {
  if [[ -e "$ENV_FILE" || -L "$ENV_FILE" ]]; then
    is_secure_regular_file "$ENV_FILE" || die "Unsafe environment file: $ENV_FILE"
    install -o root -g "$SERVICE_GROUP" -m 0640 "$ENV_FILE" "$ENV_BACKUP"
    ENV_HAD_PREVIOUS=1
  fi
  if [[ -e "$INSTALL_CONFIG" || -L "$INSTALL_CONFIG" ]]; then
    is_secure_regular_file "$INSTALL_CONFIG" || die "Unsafe install metadata: $INSTALL_CONFIG"
    install -o root -g root -m 0644 "$INSTALL_CONFIG" "$INSTALL_CONFIG_BACKUP"
    INSTALL_CONFIG_HAD_PREVIOUS=1
  fi
  if [[ -e "$UNIT_FILE" || -L "$UNIT_FILE" ]]; then
    is_secure_regular_file "$UNIT_FILE" || die "Unsafe systemd unit: $UNIT_FILE"
    install -o root -g root -m 0644 "$UNIT_FILE" "$UNIT_BACKUP"
    UNIT_HAD_PREVIOUS=1
  fi
}

is_allowed_app_environment_key() {
  case "$1" in
    TG_TOKEN|ALLOWED_USER_IDS|ALLOW_ALL_USERS|TG_PROXY|TG_GET_UPDATES_PROXY|HYSTERIA2_SOCKS5_PROXY)
      return 0
      ;;
    TR_URL|TR_PROTOCOL|TR_HOST|TR_PORT|TR_PATH|TR_USER|TR_PASS|TR_TIMEOUT)
      return 0
      ;;
    LIST_LIMIT|BOT_TIMEZONE|LOG_LEVEL|CONFIRM_DEL_KEEP)
      return 0
      ;;
  esac
  return 1
}

normalize_environment_file() {
  local normalized_env
  local line
  local key
  local value

  normalized_env="$(mktemp "$CONFIG_DIR/.environment.XXXXXX")"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ ! "$line" =~ ^[[:space:]]*([A-Z][A-Z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      continue
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    is_allowed_app_environment_key "$key" || continue
    printf '%s=%s\n' "$key" "$value" >> "$normalized_env"
  done < "$ENV_FILE"
  printf 'STATE_DIR=%s\n' "$STATE_DIR" >> "$normalized_env"
  printf 'LOG_FILE=%s/bot-errors.log\n' "$LOG_DIR" >> "$normalized_env"
  chown root:"$SERVICE_GROUP" "$normalized_env"
  chmod 0640 "$normalized_env"
  mv -f "$normalized_env" "$ENV_FILE"
}

migrate_configuration_by_copy() {
  local legacy_env="$INSTALL_DIR/.env"
  local metadata_tmp

  if [[ ! -e "$ENV_FILE" ]]; then
    if is_secure_regular_file "$legacy_env" && is_secure_parent_directory "$INSTALL_DIR"; then
      install -o root -g "$SERVICE_GROUP" -m 0640 "$legacy_env" "$ENV_FILE"
      echo "Copied legacy configuration to $ENV_FILE"
    elif [[ "$IMPORT_UNTRUSTED_LEGACY_ENV" == 1 && -f "$legacy_env" && ! -L "$legacy_env" ]]; then
      install -o root -g "$SERVICE_GROUP" -m 0640 "$legacy_env" "$ENV_FILE"
      echo "WARNING: imported service-writable legacy configuration by explicit request" >&2
    else
      install -o root -g "$SERVICE_GROUP" -m 0640 /dev/null "$ENV_FILE"
      if [[ -e "$legacy_env" || -L "$legacy_env" ]]; then
        echo "Skipped untrusted legacy configuration at $legacy_env; configure secrets again via $SERVICE_NAME update" >&2
      fi
    fi
  fi
  normalize_environment_file

  metadata_tmp="$(mktemp "$CONFIG_DIR/.install.conf.XXXXXX")"
  {
    printf 'INSTALL_DIR=%s\n' "$INSTALL_DIR"
    printf 'ENV_FILE=%s\n' "$ENV_FILE"
    printf 'STATE_DIR=%s\n' "$STATE_DIR"
    printf 'LOG_DIR=%s\n' "$LOG_DIR"
    printf 'SERVICE_USER=%s\n' "$SERVICE_USER"
    printf 'SERVICE_GROUP=%s\n' "$SERVICE_GROUP"
    printf 'REPO_URL=%s\n' "$REPO_URL"
    printf 'CLI_IMPL_PATH=%s\n' "$CLI_IMPL_PATH"
  } > "$metadata_tmp"
  chown root:root "$metadata_tmp"
  chmod 0644 "$metadata_tmp"
  mv -f "$metadata_tmp" "$INSTALL_CONFIG"
}

ensure_runtime_directory() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -d "$path" && ! -L "$path" ]] || die "Unsafe runtime directory: $path"
  else
    install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$path"
  fi
  chown root:root "$path"
  chmod 0700 "$path"
}

release_runtime_directories() {
  local path
  for path in "$STATE_DIR" "$LOG_DIR"; do
    if [[ -d "$path" && ! -L "$path" ]]; then
      chown "$SERVICE_USER:$SERVICE_GROUP" "$path"
      chmod 0750 "$path"
    fi
  done
}

migrate_mutable_files_by_copy() {
  local name
  local source_path
  local target_path

  ensure_runtime_directory "$STATE_DIR"
  ensure_runtime_directory "$LOG_DIR"

  if [[ $CURRENT_PRESENT -eq 1 && -d "$INSTALL_DIR" ]]; then
    for name in traffic_anchors.json torrent_history.json; do
      source_path="$INSTALL_DIR/$name"
      target_path="$STATE_DIR/$name"
      [[ ! -L "$target_path" ]] || die "Refusing legacy migration through symlink: $target_path"
      if [[ -f "$source_path" && ! -L "$source_path" && ! -e "$target_path" ]]; then
        install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 "$source_path" "$target_path"
        echo "Copied legacy state to $target_path"
      fi
    done

    for source_path in "$INSTALL_DIR"/bot-errors.log*; do
      [[ -f "$source_path" && ! -L "$source_path" ]] || continue
      name="$(basename "$source_path")"
      target_path="$LOG_DIR/$name"
      [[ ! -L "$target_path" ]] || die "Refusing log migration through symlink: $target_path"
      if [[ ! -e "$target_path" ]]; then
        install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 "$source_path" "$target_path"
        echo "Copied legacy log to $target_path"
      fi
    done
  fi
  release_runtime_directories
}

prepare_unit_file() {
  local unit_tmp
  unit_tmp="$(mktemp "/etc/systemd/system/.${SERVICE_NAME}.service.XXXXXX")"
  cat > "$unit_tmp" <<EOF
[Unit]
Description=Transmission Telegram Bot
Wants=network-online.target
After=network-online.target ${SERVICE_NAME}-hysteria2.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=-$ENV_FILE
Environment=STATE_DIR=$STATE_DIR
Environment=LOG_FILE=$LOG_DIR/bot-errors.log
ExecStart=$INSTALL_DIR/.venv/bin/python $INSTALL_DIR/bot.py
Restart=always
RestartSec=5
UMask=0077
StateDirectory=$SERVICE_NAME
StateDirectoryMode=0750
LogsDirectory=$SERVICE_NAME
LogsDirectoryMode=0750
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$STATE_DIR $LOG_DIR

[Install]
WantedBy=multi-user.target
EOF
  chown root:root "$unit_tmp"
  chmod 0644 "$unit_tmp"
  mv -f "$unit_tmp" "$UNIT_FILE"
}

backup_existing_cli_paths() {
  [[ ! -e "$CLI_BACKUP" && ! -L "$CLI_BACKUP" ]] || die "CLI rollback path already exists: $CLI_BACKUP"
  [[ ! -e "$CLI_IMPL_BACKUP" && ! -L "$CLI_IMPL_BACKUP" ]] || die "CLI rollback path already exists: $CLI_IMPL_BACKUP"
  if [[ $CURRENT_TRUSTED -eq 1 ]]; then
    if [[ -e "$CLI_PATH" || -L "$CLI_PATH" ]]; then
      install -o root -g root -m 0755 "$CLI_PATH" "$CLI_BACKUP"
      CLI_HAD_PREVIOUS=1
    fi
    if [[ -e "$CLI_IMPL_PATH" || -L "$CLI_IMPL_PATH" ]]; then
      install -o root -g root -m 0755 "$CLI_IMPL_PATH" "$CLI_IMPL_BACKUP"
      CLI_IMPL_HAD_PREVIOUS=1
    fi
  else
    if [[ -e "$CLI_PATH" || -L "$CLI_PATH" ]]; then
      mv "$CLI_PATH" "$CLI_BACKUP"
      CLI_HAD_PREVIOUS=1
    fi
    if [[ -e "$CLI_IMPL_PATH" || -L "$CLI_IMPL_PATH" ]]; then
      mv "$CLI_IMPL_PATH" "$CLI_IMPL_BACKUP"
      CLI_IMPL_HAD_PREVIOUS=1
    fi
  fi
  CLI_TOUCHED=1
}

install_cli_from_release() {
  local cli_tmp
  local wrapper_tmp

  if [[ -e "$CLI_IMPL_DIR" || -L "$CLI_IMPL_DIR" ]]; then
    is_secure_parent_directory "$CLI_IMPL_DIR" || die "Unsafe CLI implementation directory: $CLI_IMPL_DIR"
  else
    install -d -o root -g root -m 0755 "$CLI_IMPL_DIR"
  fi
  is_secure_parent_directory "$(dirname "$CLI_PATH")" || die "Unsafe CLI directory: $(dirname "$CLI_PATH")"
  backup_existing_cli_paths

  cli_tmp="$(mktemp "$CLI_IMPL_DIR/.${SERVICE_NAME}.XXXXXX")"
  install -o root -g root -m 0755 "$INSTALL_DIR/transmission3-bot" "$cli_tmp"
  mv -f "$cli_tmp" "$CLI_IMPL_PATH"

  wrapper_tmp="$(mktemp "$(dirname "$CLI_PATH")/.${SERVICE_NAME}.XXXXXX")"
  {
    printf '%s\n' '#!/bin/sh'
    printf '%s\n' 'unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS PYTHONBREAKPOINT PYTHONUSERBASE'
    printf '%s\n' 'unset LD_AUDIT LD_DEBUG LD_DEBUG_OUTPUT LD_DYNAMIC_WEAK LD_HWCAP_MASK LD_LIBRARY_PATH LD_ORIGIN_PATH LD_PRELOAD LD_PROFILE LD_SHOW_AUXV LD_USE_LOAD_BIAS'
    printf '%s\n' 'unset DYLD_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES DYLD_LIBRARY_PATH DYLD_PRINT_TO_FILE'
    printf '%s\n' 'unset BASH_ENV ENV CDPATH IFS HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy'
    printf '%s\n' 'unset REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR SSLKEYLOGFILE'
    printf '%s\n' 'export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PIP_CONFIG_FILE=/dev/null'
    printf 'exec /usr/bin/env -u SHELLOPTS -u BASHOPTS /usr/bin/python3 -I %s "$@"\n' "$CLI_IMPL_PATH"
  } > "$wrapper_tmp"
  chown root:root "$wrapper_tmp"
  chmod 0755 "$wrapper_tmp"
  mv -f "$wrapper_tmp" "$CLI_PATH"
}

swap_release() {
  [[ ! -e "$PREVIOUS_RELEASE" && ! -L "$PREVIOUS_RELEASE" ]] || die "Rollback path already exists: $PREVIOUS_RELEASE"
  if [[ $CURRENT_PRESENT -eq 1 ]]; then
    mv "$INSTALL_DIR" "$PREVIOUS_RELEASE"
  fi
  if ! mv "$STAGED_RELEASE" "$INSTALL_DIR"; then
    if [[ $CURRENT_TRUSTED -eq 1 && -e "$PREVIOUS_RELEASE" && ! -e "$INSTALL_DIR" ]]; then
      mv "$PREVIOUS_RELEASE" "$INSTALL_DIR"
    fi
    return 1
  fi
  STAGED_RELEASE=""
  RELEASE_SWAPPED=1
}

restore_backed_file() {
  local target="$1"
  local backup="$2"
  local had_previous="$3"
  rm -f -- "$target"
  if [[ "$had_previous" -eq 1 ]]; then
    [[ -e "$backup" ]] || return 1
    mv "$backup" "$target"
  fi
}

rollback_trusted_deployment() {
  local rollback_ok=1

  echo "Deployment failed; restoring the previous trusted release" >&2
  systemctl_cmd stop "$SERVICE_NAME" >/dev/null 2>&1 || true

  if [[ $RELEASE_SWAPPED -eq 1 ]]; then
    if [[ -e "$INSTALL_DIR" || -L "$INSTALL_DIR" ]]; then
      mv "$INSTALL_DIR" "$FAILED_RELEASE" || rollback_ok=0
    fi
    if [[ $rollback_ok -eq 1 && $CURRENT_PRESENT -eq 1 && -e "$PREVIOUS_RELEASE" ]]; then
      mv "$PREVIOUS_RELEASE" "$INSTALL_DIR" || rollback_ok=0
    elif [[ $CURRENT_PRESENT -eq 1 ]]; then
      rollback_ok=0
    fi
  fi

  if [[ $CLI_TOUCHED -eq 1 ]]; then
    restore_backed_file "$CLI_PATH" "$CLI_BACKUP" "$CLI_HAD_PREVIOUS" || rollback_ok=0
    restore_backed_file "$CLI_IMPL_PATH" "$CLI_IMPL_BACKUP" "$CLI_IMPL_HAD_PREVIOUS" || rollback_ok=0
  fi
  if [[ $CONFIG_TOUCHED -eq 1 ]]; then
    restore_backed_file "$ENV_FILE" "$ENV_BACKUP" "$ENV_HAD_PREVIOUS" || rollback_ok=0
    restore_backed_file "$INSTALL_CONFIG" "$INSTALL_CONFIG_BACKUP" "$INSTALL_CONFIG_HAD_PREVIOUS" || rollback_ok=0
  fi
  if [[ $UNIT_TOUCHED -eq 1 ]]; then
    restore_backed_file "$UNIT_FILE" "$UNIT_BACKUP" "$UNIT_HAD_PREVIOUS" || rollback_ok=0
  fi
  release_runtime_directories
  systemctl_cmd daemon-reload >/dev/null 2>&1 || true
  if [[ $SERVICE_WAS_ENABLED -eq 1 ]]; then
    systemctl_cmd enable "$SERVICE_NAME" >/dev/null 2>&1 || rollback_ok=0
  else
    systemctl_cmd disable "$SERVICE_NAME" >/dev/null 2>&1 || rollback_ok=0
  fi

  if [[ $SERVICE_WAS_ACTIVE -eq 1 && $rollback_ok -eq 1 ]]; then
    if start_service_and_check; then
      echo "Previous trusted release restored and active" >&2
    else
      echo "Previous release was restored but did not become active" >&2
    fi
  fi

  if [[ $rollback_ok -eq 0 ]]; then
    echo "Rollback could not be completed; the service remains stopped." >&2
  elif [[ -e "$FAILED_RELEASE" ]]; then
    rm -rf -- "$FAILED_RELEASE"
  fi
}

rollback_or_fail_closed() {
  if [[ $CURRENT_TRUSTED -eq 1 ]]; then
    rollback_trusted_deployment
    return
  fi

  systemctl_cmd stop "$SERVICE_NAME" >/dev/null 2>&1 || true
  systemctl_cmd disable "$SERVICE_NAME" >/dev/null 2>&1 || true
  release_runtime_directories
  if [[ $CURRENT_PRESENT -eq 1 ]]; then
    echo "Deployment failed and the previous release was untrusted." >&2
    echo "The service remains stopped; the untrusted release will not be restored or executed." >&2
  else
    echo "Initial deployment failed; no previous release is available and the service remains stopped." >&2
  fi
  if [[ -e "$PREVIOUS_RELEASE" || -L "$PREVIOUS_RELEASE" ]]; then
    echo "Untrusted files were preserved for inspection at $PREVIOUS_RELEASE" >&2
  fi
}

cleanup_artifacts() {
  if [[ -n "$BUILD_ROOT" && -d "$BUILD_ROOT" ]]; then
    rm -rf -- "$BUILD_ROOT"
  fi
  if [[ $DEPLOY_COMMITTED -eq 1 ]]; then
    rm -rf -- "$PREVIOUS_RELEASE" "$CLI_BACKUP" "$CLI_IMPL_BACKUP"
  fi
  rm -f -- "$ENV_BACKUP" "$INSTALL_CONFIG_BACKUP" "$UNIT_BACKUP"
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ $DEPLOYMENT_LOCKED -eq 1 ]]; then
    if [[ $exit_code -ne 0 && $TRANSACTION_STARTED -eq 1 && $DEPLOY_COMMITTED -eq 0 ]]; then
      rollback_or_fail_closed
    elif [[ $exit_code -ne 0 && $DEPLOYMENT_ASSESSED -eq 1 && $CURRENT_TRUSTED -eq 0 ]]; then
      systemctl_cmd stop "$SERVICE_NAME" >/dev/null 2>&1 || true
      systemctl_cmd disable "$SERVICE_NAME" >/dev/null 2>&1 || true
      echo "Update failed before activation; no untrusted or incomplete release will start at boot." >&2
    fi
    cleanup_artifacts
  fi
  release_deployment_lock
  exit "$exit_code"
}
trap on_exit EXIT

if [[ ${TRANSMISSION3_BOT_INSTALL_LIB_ONLY:-0} == 1 ]]; then
  trap - EXIT
  return 0 2>/dev/null || exit 0
fi

if [[ ${EUID} -ne 0 ]]; then
  die "Please run as root: sudo bash install.sh"
fi

sanitize_build_environment
acquire_deployment_lock
validate_inputs
ensure_system_dependencies
ensure_service_user

if [[ -e "$INSTALL_PARENT" || -L "$INSTALL_PARENT" ]]; then
  is_secure_directory_ancestry "$INSTALL_PARENT" \
    || die "INSTALL_DIR ancestry must be root-owned and not group/world-writable"
else
  install -d -o root -g root -m 0755 "$INSTALL_PARENT"
fi
is_secure_directory_ancestry "$INSTALL_PARENT" \
  || die "INSTALL_DIR ancestry must be root-owned and not group/world-writable"
assess_current_deployment
DEPLOYMENT_ASSESSED=1

if systemctl_cmd is-active --quiet "$SERVICE_NAME"; then
  SERVICE_WAS_ACTIVE=1
fi
if systemctl_cmd is-enabled --quiet "$SERVICE_NAME"; then
  SERVICE_WAS_ENABLED=1
fi
SHOULD_START=$SERVICE_WAS_ACTIVE
if [[ "$START_SERVICE" == 1 ]]; then
  SHOULD_START=1
elif [[ "$START_SERVICE" == 0 ]]; then
  SHOULD_START=0
fi
if [[ $CURRENT_PRESENT -eq 1 && $CURRENT_TRUSTED -eq 0 && $SERVICE_WAS_ACTIVE -eq 1 ]]; then
  systemctl_cmd stop "$SERVICE_NAME"
  UNTRUSTED_SERVICE_STOPPED=1
  echo "Stopped legacy service before building; its writable checkout will not be executed as root."
fi

build_clean_release

ensure_secure_config_directory
if [[ $SERVICE_WAS_ACTIVE -eq 1 && $UNTRUSTED_SERVICE_STOPPED -eq 0 ]]; then
  systemctl_cmd stop "$SERVICE_NAME"
fi
TRANSACTION_STARTED=1
backup_transaction_files
CONFIG_TOUCHED=1
migrate_configuration_by_copy
migrate_mutable_files_by_copy

swap_release
install_cli_from_release
UNIT_TOUCHED=1
prepare_unit_file
systemctl_cmd daemon-reload
systemctl_cmd enable "$SERVICE_NAME"

if [[ $SHOULD_START -eq 1 ]]; then
  start_service_and_check || die "New release did not remain active"
fi

DEPLOY_COMMITTED=1
cleanup_artifacts
echo "Installed a clean root-owned release in $INSTALL_DIR"
echo "Configure token/user id via: sudo transmission3-bot update"
if [[ $SHOULD_START -eq 0 ]]; then
  echo "Then start bot: systemctl start $SERVICE_NAME"
fi
