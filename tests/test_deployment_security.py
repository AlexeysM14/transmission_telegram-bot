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
