from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from .models import CommandResult


class UnsafeCommand(ValueError):
    """Raised when a command is outside PatchPilot's explicit safe policy."""


_BLOCKED_WORDS = {
    "sudo", "rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "passwd",
    "curl", "wget", "ssh", "scp", "nc", "chmod", "chown",
}
_BLOCKED_GIT_ACTIONS = {"reset", "clean", "push", "commit", "checkout", "restore"}
_SHELL_OPERATORS = {";", "&&", "||", "|", ">", ">>", "<", "$", "`"}
_ALLOWED_PREFIXES = (
    ("git", "status"), ("git", "diff"), ("git", "log"), ("git", "branch"),
    ("python", "-m", "pytest"), ("python3", "-m", "pytest"), ("pytest",),
    ("npm", "test"), ("go", "test"), ("cargo", "test"),
)


def validate_command(command: str) -> tuple[str, ...]:
    if not command.strip():
        raise UnsafeCommand("empty commands are not allowed")
    if any(operator in command for operator in _SHELL_OPERATORS):
        raise UnsafeCommand("shell operators and redirection are not allowed")
    try:
        args = tuple(shlex.split(command))
    except ValueError as exc:
        raise UnsafeCommand(f"cannot parse command: {exc}") from exc
    if not args:
        raise UnsafeCommand("empty commands are not allowed")
    if args[0] in _BLOCKED_WORDS or any(part in _BLOCKED_WORDS for part in args):
        raise UnsafeCommand("command contains a blocked executable")
    if args[0] == "git" and len(args) > 1 and args[1] in _BLOCKED_GIT_ACTIONS:
        raise UnsafeCommand(f"git {args[1]} is blocked")
    if not any(args[:len(prefix)] == prefix for prefix in _ALLOWED_PREFIXES):
        raise UnsafeCommand("command is not on the allow-list")
    return args


def run_safe(command: str, repository: Path, timeout: float = 30.0) -> CommandResult:
    root = repository.resolve()
    if not root.is_dir():
        raise ValueError(f"repository does not exist: {root}")
    args = validate_command(command)
    if args[:3] in {
        ("python", "-m", "pytest"),
        ("python3", "-m", "pytest"),
    }:
        args = (sys.executable, *args[1:])
    try:
        completed = subprocess.run(
            args, cwd=root, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return CommandResult(command, 124, stdout, stderr, timed_out=True)
