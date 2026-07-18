from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


def run_installer_functions(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    script = f"""
set -euo pipefail
export TRANSMISSION3_BOT_INSTALL_LIB_ONLY=1
source {shlex.quote(str(INSTALLER))}
unset TRANSMISSION3_BOT_INSTALL_LIB_ONLY

TEST_ROOT={shlex.quote(str(tmp_path))}
INSTALL_DIR="$TEST_ROOT/install"
PREVIOUS_RELEASE="$TEST_ROOT/previous"
FAILED_RELEASE="$TEST_ROOT/failed"
CLI_PATH="$TEST_ROOT/cli"
CLI_IMPL_PATH="$TEST_ROOT/cli-impl"
CLI_BACKUP="$TEST_ROOT/cli.previous"
CLI_IMPL_BACKUP="$TEST_ROOT/cli-impl.previous"
ENV_FILE="$TEST_ROOT/environment"
ENV_BACKUP="$TEST_ROOT/environment.previous"
INSTALL_CONFIG="$TEST_ROOT/install.conf"
INSTALL_CONFIG_BACKUP="$TEST_ROOT/install.conf.previous"
UNIT_FILE="$TEST_ROOT/service.unit"
UNIT_BACKUP="$TEST_ROOT/service.unit.previous"
STATE_DIR="$TEST_ROOT/state"
LOG_DIR="$TEST_ROOT/log"

systemctl_cmd() {{
  printf '%s\n' "$*" >> "$TEST_ROOT/systemctl.calls"
  if [[ "$1" == is-active && -f "$INSTALL_DIR/release-marker" && "$(<"$INSTALL_DIR/release-marker")" == new ]]; then
    return 1
  fi
  return 0
}}
release_runtime_directories() {{ :; }}
sleep() {{ :; }}

{body}
"""
    return subprocess.run(
        ["/bin/bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_trusted_health_failure_restores_release_cli_and_configuration(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
mkdir -p "$INSTALL_DIR" "$PREVIOUS_RELEASE"
printf new > "$INSTALL_DIR/release-marker"
printf old > "$PREVIOUS_RELEASE/release-marker"
printf new-cli > "$CLI_PATH"
printf old-cli > "$CLI_BACKUP"
printf new-impl > "$CLI_IMPL_PATH"
printf old-impl > "$CLI_IMPL_BACKUP"
printf new-env > "$ENV_FILE"
printf old-env > "$ENV_BACKUP"
printf new-config > "$INSTALL_CONFIG"
printf old-config > "$INSTALL_CONFIG_BACKUP"
printf new-unit > "$UNIT_FILE"
printf old-unit > "$UNIT_BACKUP"

CURRENT_PRESENT=1
CURRENT_TRUSTED=1
SERVICE_WAS_ACTIVE=1
SERVICE_WAS_ENABLED=1
RELEASE_SWAPPED=1
CLI_TOUCHED=1
CLI_HAD_PREVIOUS=1
CLI_IMPL_HAD_PREVIOUS=1
CONFIG_TOUCHED=1
ENV_HAD_PREVIOUS=1
INSTALL_CONFIG_HAD_PREVIOUS=1
UNIT_TOUCHED=1
UNIT_HAD_PREVIOUS=1

if ! start_service_and_check; then
  rollback_trusted_deployment
fi

[[ "$(<"$INSTALL_DIR/release-marker")" == old ]]
[[ "$(<"$CLI_PATH")" == old-cli ]]
[[ "$(<"$CLI_IMPL_PATH")" == old-impl ]]
[[ "$(<"$ENV_FILE")" == old-env ]]
[[ "$(<"$INSTALL_CONFIG")" == old-config ]]
[[ "$(<"$UNIT_FILE")" == old-unit ]]
[[ ! -e "$FAILED_RELEASE" ]]
grep -q '^start transmission3-bot$' "$TEST_ROOT/systemctl.calls"
grep -q '^is-active --quiet transmission3-bot$' "$TEST_ROOT/systemctl.calls"
grep -q '^enable transmission3-bot$' "$TEST_ROOT/systemctl.calls"
""",
    )

    assert result.returncode == 0, result.stderr


def test_untrusted_previous_release_is_quarantined_and_never_started(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
mkdir -p "$INSTALL_DIR" "$PREVIOUS_RELEASE"
printf clean-new > "$INSTALL_DIR/release-marker"
printf untrusted-old > "$PREVIOUS_RELEASE/release-marker"
CURRENT_PRESENT=1
CURRENT_TRUSTED=0
SERVICE_WAS_ACTIVE=1
RELEASE_SWAPPED=1

rollback_or_fail_closed

[[ "$(<"$INSTALL_DIR/release-marker")" == clean-new ]]
[[ "$(<"$PREVIOUS_RELEASE/release-marker")" == untrusted-old ]]
! grep -q '^start ' "$TEST_ROOT/systemctl.calls"
grep -q '^stop transmission3-bot$' "$TEST_ROOT/systemctl.calls"
grep -q '^disable transmission3-bot$' "$TEST_ROOT/systemctl.calls"
""",
    )

    assert result.returncode == 0, result.stderr


def test_release_swap_uses_sibling_rename_and_keeps_previous(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
mkdir -p "$INSTALL_DIR" "$TEST_ROOT/staged"
printf old > "$INSTALL_DIR/release-marker"
printf new > "$TEST_ROOT/staged/release-marker"
STAGED_RELEASE="$TEST_ROOT/staged"
CURRENT_PRESENT=1
CURRENT_TRUSTED=1

swap_release

[[ "$(<"$INSTALL_DIR/release-marker")" == new ]]
[[ "$(<"$PREVIOUS_RELEASE/release-marker")" == old ]]
[[ $RELEASE_SWAPPED -eq 1 ]]
""",
    )

    assert result.returncode == 0, result.stderr


def test_validate_inputs_rejects_noncanonical_and_protected_paths(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
REPO_URL=https://example.invalid/repository.git
for candidate in \
  /opt/../etc \
  /opt/..//etc/// \
  /opt//transmission3-bot \
  /opt/./transmission3-bot \
  /opt/transmission3-bot/ \
  /etc/ssh \
  /usr/bin/python3 \
  /var/lib/transmission3-bot; do
  if (INSTALL_DIR="$candidate"; validate_inputs >/dev/null 2>&1); then
    printf 'unsafe INSTALL_DIR was accepted: %s\n' "$candidate" >&2
    exit 1
  fi
done

if (INSTALL_DIR=/opt/transmission3-bot-new; EXPECTED_COMMIT=not-a-commit; validate_inputs >/dev/null 2>&1); then
  echo 'invalid EXPECTED_COMMIT was accepted' >&2
  exit 1
fi
""",
    )

    assert result.returncode == 0, result.stderr


def test_validate_inputs_rejects_unrelated_existing_targets_but_accepts_app_markers(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
REPO_URL=https://example.invalid/repository.git
is_protected_install_path() { return 1; }

unrelated_file="$TEST_ROOT/unrelated-file"
unrelated_dir="$TEST_ROOT/unrelated-dir"
marked_dir="$TEST_ROOT/marked-install"
legacy_dir="$TEST_ROOT/legacy-install"
symlink_install="$TEST_ROOT/symlink-install"
symlink_parent="$TEST_ROOT/symlink-parent"
printf unrelated > "$unrelated_file"
mkdir -p "$unrelated_dir" "$marked_dir" "$legacy_dir/.venv" "$TEST_ROOT/physical-parent"
printf '%s\n' "$SERVICE_NAME" > "$marked_dir/$INSTALL_MARKER_NAME"
ln -s "$marked_dir" "$symlink_install"
ln -s "$TEST_ROOT/physical-parent" "$symlink_parent"
printf 'from transmission_rpc import Client\n' > "$legacy_dir/bot.py"
printf 'transmission-rpc>=7,<8\n' > "$legacy_dir/requirements.txt"
printf 'SERVICE_NAME = "transmission3-bot"\n' > "$legacy_dir/transmission3-bot"

for candidate in "$unrelated_file" "$unrelated_dir" "$symlink_install" "$symlink_parent/new-install"; do
  if (INSTALL_DIR="$candidate"; validate_inputs >/dev/null 2>&1); then
    printf 'unrelated existing target was accepted: %s\n' "$candidate" >&2
    exit 1
  fi
done

(INSTALL_DIR="$marked_dir"; validate_inputs)
(INSTALL_DIR="$legacy_dir"; validate_inputs)
""",
    )

    assert result.returncode == 0, result.stderr


def test_new_release_marker_is_written_safely(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
release="$TEST_ROOT/release"
mkdir -p "$release"
chown() { :; }
printf outside > "$TEST_ROOT/outside-marker-target"
ln -s "$TEST_ROOT/outside-marker-target" "$release/$INSTALL_MARKER_NAME"

write_install_marker "$release"

[[ -f "$release/$INSTALL_MARKER_NAME" ]]
[[ ! -L "$release/$INSTALL_MARKER_NAME" ]]
[[ "$(<"$release/$INSTALL_MARKER_NAME")" == "$SERVICE_NAME" ]]
[[ "$(<"$TEST_ROOT/outside-marker-target")" == outside ]]
""",
    )

    assert result.returncode == 0, result.stderr


def test_validate_inputs_rejects_root_service_user_and_group(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
REPO_URL=https://example.invalid/repository.git
INSTALL_DIR="$TEST_ROOT/new-install"
is_protected_install_path() { return 1; }

if (
  lookup_user_id() { printf '0\n'; }
  lookup_group_id() { return 1; }
  validate_inputs >/dev/null 2>&1
); then
  echo 'UID 0 service user was accepted' >&2
  exit 1
fi

if (
  lookup_user_id() { printf '100\n'; }
  lookup_group_id() { printf '0\n'; }
  validate_inputs >/dev/null 2>&1
); then
  echo 'GID 0 service group was accepted' >&2
  exit 1
fi
""",
    )

    assert result.returncode == 0, result.stderr


def test_sanitize_build_environment_removes_root_updater_overrides(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
export BASH_ENV="$TEST_ROOT/bash-env"
export ENV="$TEST_ROOT/sh-env"
export CDPATH="$TEST_ROOT"
export IFS=:
export HTTP_PROXY=http://attacker.invalid:8080
export https_proxy=http://attacker.invalid:8080
export REQUESTS_CA_BUNDLE="$TEST_ROOT/attacker-ca.pem"
export CURL_CA_BUNDLE="$TEST_ROOT/attacker-ca.pem"
export SSL_CERT_FILE="$TEST_ROOT/attacker-ca.pem"
export SSL_CERT_DIR="$TEST_ROOT/attacker-certs"
export SSLKEYLOGFILE="$TEST_ROOT/tls-keys.log"
export SHELLOPTS

sanitize_build_environment

for variable in \
  BASH_ENV ENV CDPATH IFS HTTP_PROXY https_proxy REQUESTS_CA_BUNDLE \
  CURL_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR SSLKEYLOGFILE SHELLOPTS; do
  if /usr/bin/env | /usr/bin/grep -q "^${variable}="; then
    printf 'unsafe variable remained exported: %s\n' "$variable" >&2
    exit 1
  fi
done
[[ "$PATH" == /usr/sbin:/usr/bin:/sbin:/bin ]]
""",
    )

    assert result.returncode == 0, result.stderr


def test_untrusted_legacy_environment_requires_explicit_opt_in(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
CONFIG_DIR="$TEST_ROOT/config"
ENV_FILE="$CONFIG_DIR/environment"
INSTALL_CONFIG="$CONFIG_DIR/install.conf"
INSTALL_DIR="$TEST_ROOT/legacy-install"
mkdir -p "$CONFIG_DIR" "$INSTALL_DIR"
printf 'TG_TOKEN=attacker-token\nALLOW_ALL_USERS=1\n' > "$INSTALL_DIR/.env"

is_secure_regular_file() { return 1; }
is_secure_parent_directory() { return 1; }
chown() { :; }
install() {
  local arguments=("$@")
  local count="${#arguments[@]}"
  /bin/cp "${arguments[$((count - 2))]}" "${arguments[$((count - 1))]}"
}

migrate_configuration_by_copy

! /usr/bin/grep -q '^TG_TOKEN=' "$ENV_FILE"
! /usr/bin/grep -q '^ALLOW_ALL_USERS=' "$ENV_FILE"
/usr/bin/grep -q "^STATE_DIR=$STATE_DIR$" "$ENV_FILE"
/usr/bin/grep -q "^LOG_FILE=$LOG_DIR/bot-errors.log$" "$ENV_FILE"

rm -f "$ENV_FILE" "$INSTALL_CONFIG"
IMPORT_UNTRUSTED_LEGACY_ENV=1
migrate_configuration_by_copy
/usr/bin/grep -q '^TG_TOKEN=attacker-token$' "$ENV_FILE"
/usr/bin/grep -q '^ALLOW_ALL_USERS=1$' "$ENV_FILE"
""",
    )

    assert result.returncode == 0, result.stderr
    assert "Skipped untrusted legacy configuration" in result.stderr
    assert "imported service-writable legacy configuration by explicit request" in result.stderr


def test_secure_legacy_environment_is_allowlisted_during_import(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
CONFIG_DIR="$TEST_ROOT/config"
ENV_FILE="$CONFIG_DIR/environment"
INSTALL_CONFIG="$CONFIG_DIR/install.conf"
INSTALL_DIR="$TEST_ROOT/legacy-install"
mkdir -p "$CONFIG_DIR" "$INSTALL_DIR"
{
  printf '%s\n' \
    'TG_TOKEN=trusted-token' \
    'ALLOWED_USER_IDS=12345' \
    'TR_PASS=transmission-secret' \
    'HTTP_PROXY=http://attacker.invalid:8080' \
    'SSL_CERT_FILE=/tmp/attacker-ca.pem' \
    'BASH_ENV=/tmp/attacker-script' \
    'PATH=/tmp/attacker-bin' \
    'STATE_DIR=/tmp/attacker-state' \
    'LOG_FILE=/tmp/attacker.log' \
    'UNKNOWN_SETTING=unexpected'
} > "$INSTALL_DIR/.env"

is_secure_regular_file() { return 0; }
is_secure_parent_directory() { return 0; }
chown() { :; }
install() {
  local arguments=("$@")
  local count="${#arguments[@]}"
  /bin/cp "${arguments[$((count - 2))]}" "${arguments[$((count - 1))]}"
}

migrate_configuration_by_copy

/usr/bin/grep -q '^TG_TOKEN=trusted-token$' "$ENV_FILE"
/usr/bin/grep -q '^ALLOWED_USER_IDS=12345$' "$ENV_FILE"
/usr/bin/grep -q '^TR_PASS=transmission-secret$' "$ENV_FILE"
for variable in HTTP_PROXY SSL_CERT_FILE BASH_ENV PATH UNKNOWN_SETTING; do
  ! /usr/bin/grep -q "^${variable}=" "$ENV_FILE"
done
/usr/bin/grep -q "^STATE_DIR=$STATE_DIR$" "$ENV_FILE"
/usr/bin/grep -q "^LOG_FILE=$LOG_DIR/bot-errors.log$" "$ENV_FILE"
""",
    )

    assert result.returncode == 0, result.stderr


def test_deployment_lock_rejects_parallel_installer_and_releases_cleanly(tmp_path: Path) -> None:
    result = run_installer_functions(
        tmp_path,
        r"""
LOCK_DIR="$TEST_ROOT/lock-dir"
LOCK_FILE="$LOCK_DIR/transmission3-bot.install.lock"
mkdir -p "$LOCK_DIR"

flock_is_available() { return 0; }
is_secure_lock_directory() { return 0; }
is_secure_regular_file() { return 0; }
flock_cmd() {
  if [[ "$1" == -n ]]; then
    /bin/mkdir "$TEST_ROOT/flock-held" 2>/dev/null
  else
    /bin/rmdir "$TEST_ROOT/flock-held" 2>/dev/null || true
  fi
}

acquire_deployment_lock
[[ $DEPLOYMENT_LOCKED -eq 1 ]]
[[ -f "$LOCK_FILE" && ! -L "$LOCK_FILE" ]]

if (DEPLOYMENT_LOCKED=0; acquire_deployment_lock); then
  echo 'parallel deployment unexpectedly acquired the lock' >&2
  exit 1
fi

release_deployment_lock
[[ $DEPLOYMENT_LOCKED -eq 0 ]]
acquire_deployment_lock
release_deployment_lock
""",
    )

    assert result.returncode == 0, result.stderr
    assert "Another transmission3-bot installation or update is already running" in result.stderr


def test_release_build_requires_hashes_checks_dependencies_and_removes_packaging_tools() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")

    require_hashes = installer.index("--require-hashes")
    check_command = '"$STAGED_RELEASE/.venv/bin/python" -I -m pip check'
    first_dependency_check = installer.index(check_command)
    remove_setuptools = installer.index("pip uninstall --yes setuptools wheel")
    final_dependency_check = installer.index(check_command, first_dependency_check + len(check_command))
    remove_pip = installer.index("pip uninstall --yes pip")
    harden_release = installer.index('harden_release "$STAGED_RELEASE"')

    assert (
        require_hashes
        < first_dependency_check
        < remove_setuptools
        < final_dependency_check
        < remove_pip
        < harden_release
    )
