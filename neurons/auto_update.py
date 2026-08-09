"""Auto-update watchtower for Verathos neurons.

Periodically checks the Git remote for new commits and, when the relevant
role version has been bumped, pulls the latest code, reinstalls the
package, and restarts the process.

Version-aware restarts
----------------------
The updater does NOT restart on every commit.  It reads
``neurons/version.py`` from the **remote** (via ``git show``) and
compares the role-specific version:

- Miner processes check ``miner_version``
- Validator and proxy processes check ``validator_version``

Only if the remote version is **higher** does it pull + restart.  This
means a validator-only fix doesn't bounce miners (and vice versa).

CLI integration
---------------
Each neuron adds ``--auto-update`` to its argparser.  When enabled::

    from neurons.auto_update import AutoUpdater
    updater = AutoUpdater(role="validator", check_interval=1800)
    updater.start()
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import bittensor as bt
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Repo root = directory containing this file's parent (neurons/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIRST_PARTY_SOURCE_ROOTS = ("neurons", "verallm", "zkllm", "trainllm")


def _run_git(*args: str, cwd: Optional[Path] = None) -> tuple[int, str]:
    """Run a git command and return (returncode, stdout+stderr)."""
    cmd = ["git"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or _REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "git command timed out"
    except Exception as e:
        return 1, str(e)


def _current_python_tag() -> str:
    """Return the wheel ABI tag for the running Python minor, e.g. cp312."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _find_local_wheel(prefix: str, py_tag: Optional[str] = None) -> Optional[Path]:
    """Find the bundled wheel matching the running Python minor."""
    tag = py_tag or _current_python_tag()
    dist_dir = _REPO_ROOT / "dist"
    wheels = sorted(dist_dir.glob(f"{prefix}-*-{tag}-*.whl"))
    return wheels[-1] if wheels else None


def install_local_wheel(prefix: str, *, required: bool = True) -> bool:
    """Force-install a bundled wheel for this Python minor."""
    wheel = _find_local_wheel(prefix)
    if wheel is None:
        log_fn = bt.logging.error if required else bt.logging.warning
        log_fn(
            f"No bundled {prefix} wheel found for {_current_python_tag()} in {_REPO_ROOT / 'dist'}"
        )
        return not required

    bt.logging.info(f"Installing {prefix} wheel: {wheel.name}")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                str(wheel),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            bt.logging.error(f"{prefix} wheel install failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        bt.logging.error(f"{prefix} wheel install timed out")
        return False

    bt.logging.info(f"{prefix} wheel installed successfully")
    return True


def install_local_zkllm_wheel() -> bool:
    """Force-install the bundled zkllm wheel for this Python minor."""
    return install_local_wheel("zkllm")


def install_local_hot_capacity_workspace_wheel(*, required: bool = True) -> bool:
    """Force-install the bundled hot-capacity workspace wheel for this Python minor."""
    return install_local_wheel("hot_capacity_workspace_cuda", required=required)


def _proof_v3_cuda_runtime_tag() -> Optional[str]:
    """Return the bundled proof-runtime tag for the installed torch CUDA ABI."""
    try:
        import torch
    except Exception:
        return None
    cuda = str(torch.version.cuda or "")
    if cuda.startswith("12.8"):
        return "cu128"
    if cuda.startswith("13.0"):
        return "cu130"
    return None


def install_local_proof_v3_cuda_wheel() -> bool:
    """Force-install the proof-v3 CUDA wheel matching torch and Python."""
    runtime_tag = _proof_v3_cuda_runtime_tag()
    if runtime_tag is None:
        bt.logging.error(
            "No bundled proof-v3 CUDA runtime matches the installed torch CUDA ABI"
        )
        return False
    py_tag = _current_python_tag()
    wheels = sorted(
        (_REPO_ROOT / "dist").glob(
            f"verathos_proof_v3_cuda-*+{runtime_tag}-{py_tag}-{py_tag}-*.whl"
        )
    )
    if len(wheels) != 1:
        bt.logging.error(
            "Expected exactly one bundled proof-v3 CUDA wheel for "
            f"runtime={runtime_tag} python={py_tag}; found {len(wheels)}"
        )
        return False

    wheel = wheels[0]
    bt.logging.info(f"Installing proof-v3 CUDA wheel: {wheel.name}")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            bt.logging.error(f"proof-v3 CUDA wheel install failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        bt.logging.error("proof-v3 CUDA wheel install timed out")
        return False
    bt.logging.info("proof-v3 CUDA wheel installed successfully")
    return True


def _selected_proof_v3_cuda_wheel() -> Optional[Path]:
    runtime_tag = _proof_v3_cuda_runtime_tag()
    if runtime_tag is None:
        return None
    py_tag = _current_python_tag()
    wheels = sorted(
        (_REPO_ROOT / "dist").glob(
            f"verathos_proof_v3_cuda-*+{runtime_tag}-{py_tag}-{py_tag}-*.whl"
        )
    )
    return wheels[0] if len(wheels) == 1 else None


def _installed_proof_v3_cuda_package_dir() -> Optional[Path]:
    spec = importlib.util.find_spec("verathos_proof_v3_cuda")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def proof_v3_cuda_wheel_is_current(wheel: Path) -> bool:
    """Compare installed proof-runtime payload bytes with the release wheel."""
    package_dir = _installed_proof_v3_cuda_package_dir()
    if package_dir is None:
        return False
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.startswith("verathos_proof_v3_cuda/")
                and not name.endswith("/")
            ]
            if not members:
                return False
            for name in members:
                installed = package_dir / Path(name).name
                if not installed.is_file():
                    return False
                if hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(
                    archive.read(name)
                ).digest():
                    return False
    except (OSError, zipfile.BadZipFile, KeyError):
        return False
    return True


def proof_v3_cuda_runtime_smoke() -> bool:
    """Load and execute both bundled kernel families in a fresh process."""
    program = """
import torch
from verathos_proof_v3_cuda import load_fused_kernels, load_tree_kernels
fold = load_fused_kernels()
tree = load_tree_kernels()
for name in ('round_partials', 'lerp_fold', 'product_round_partials', 'fs_round_v2'):
    assert hasattr(fold, name), name
for name in ('leaf_hash_w1', 'leaf_hash_wn_base', 'node_hash', 'node_hash_base'):
    assert hasattr(tree, name), name
folded = fold.lerp_fold(torch.tensor([1, 2], dtype=torch.int64, device='cuda'), 0)
assert folded.cpu().tolist() == [1]
leaf = tree.leaf_hash_wn_base(
    torch.zeros(90, dtype=torch.uint8, device='cuda'),
    torch.tensor([1], dtype=torch.int64, device='cuda'),
    0,
    1,
)
assert tuple(leaf.shape) == (32,)
torch.cuda.synchronize()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        bt.logging.error("proof-v3 CUDA runtime smoke timed out")
        return False
    if result.returncode != 0:
        bt.logging.error(
            "proof-v3 CUDA runtime smoke failed: "
            f"{(result.stderr or result.stdout).strip()}"
        )
        return False
    return True


def ensure_local_proof_v3_cuda_wheel() -> bool:
    """Ensure first-start after an old updater has the release proof runtime."""
    wheel = _selected_proof_v3_cuda_wheel()
    if wheel is None:
        bt.logging.error(
            "No unique bundled proof-v3 CUDA wheel matches Python and torch CUDA"
        )
        return False
    if not proof_v3_cuda_wheel_is_current(wheel):
        if not install_local_proof_v3_cuda_wheel():
            return False
        importlib.invalidate_caches()
        if not proof_v3_cuda_wheel_is_current(wheel):
            return False
    return proof_v3_cuda_runtime_smoke()


def get_local_head() -> Optional[str]:
    """Get the current local HEAD commit hash."""
    rc, out = _run_git("rev-parse", "HEAD")
    return out if rc == 0 else None


def get_current_branch() -> Optional[str]:
    """Get the current branch name."""
    rc, out = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return out if rc == 0 else None


def _clear_first_party_bytecode_caches() -> bool:
    """Remove stale timestamp-based bytecode after a source fast-forward.

    Git can replace a same-sized Python file within the same timestamp second.
    CPython's default bytecode header then cannot distinguish the new source
    from the old one.  Clear only first-party ``__pycache__`` entries before
    restart; environments and third-party packages are deliberately excluded.
    """

    try:
        for root_name in _FIRST_PARTY_SOURCE_ROOTS:
            source_root = _REPO_ROOT / root_name
            if not source_root.is_dir():
                continue
            for cache_dir in source_root.rglob("__pycache__"):
                if not cache_dir.is_dir():
                    continue
                for bytecode in cache_dir.glob("*.pyc"):
                    bytecode.unlink()
                try:
                    cache_dir.rmdir()
                except OSError:
                    # Leave directories containing non-bytecode files alone.
                    pass
    except OSError as exc:
        bt.logging.error(f"Cannot clear first-party bytecode cache: {exc}")
        return False
    importlib.invalidate_caches()
    return True


def fetch_origin() -> bool:
    """Fetch from origin. Returns True on success."""
    rc, out = _run_git("fetch", "origin", "--quiet")
    if rc != 0:
        if "not a git repository" in out:
            bt.logging.debug("Auto-update skipped: not a git repository")
        else:
            bt.logging.warning(f"git fetch failed: {out}")
        return False
    return True


def get_remote_head(branch: Optional[str] = None) -> Optional[str]:
    """Return the remote HEAD hash for the current branch (after fetch)."""
    if branch is None:
        branch = get_current_branch()
        if not branch:
            return None
    rc, remote_hash = _run_git("rev-parse", f"origin/{branch}")
    return remote_hash if rc == 0 else None


def _read_remote_version_file(branch: Optional[str] = None) -> Optional[str]:
    """Read neurons/version.py from the remote branch without pulling.

    Uses ``git show origin/<branch>:neurons/version.py`` so we can
    inspect the remote version before deciding whether to pull.
    """
    if branch is None:
        branch = get_current_branch()
        if not branch:
            return None

    rc, content = _run_git("show", f"origin/{branch}:neurons/version.py")
    if rc != 0:
        bt.logging.warning(f"Cannot read remote version.py: {content}")
        return None
    return content


def _parse_version_from_source(source: str, var_name: str) -> Optional[int]:
    """Extract a version integer from version.py source code.

    Looks for ``<var_name>: int = _encode(MAJOR, MINOR, PATCH)`` or
    the raw assignment pattern and computes the value.

    We parse the MAJOR/MINOR/PATCH constants for each role prefix and
    compute the encoded integer.
    """
    if var_name == "spec_version":
        miner = _parse_version_from_source(source, "miner_version")
        validator = _parse_version_from_source(source, "validator_version")
        if miner is None or validator is None:
            return None
        return max(miner, validator)

    # Determine prefix: miner_version → MINER_, validator_version → VALIDATOR_
    prefix_map = {
        "miner_version": "MINER_",
        "validator_version": "VALIDATOR_",
    }
    prefix = prefix_map.get(var_name)
    if not prefix:
        return None

    def _find_int(name: str) -> Optional[int]:
        pattern = rf"^{re.escape(name)}\s*=\s*(\d+)"
        m = re.search(pattern, source, re.MULTILINE)
        return int(m.group(1)) if m else None

    major = _find_int(f"{prefix}MAJOR")
    minor = _find_int(f"{prefix}MINOR")
    patch = _find_int(f"{prefix}PATCH")

    if major is None or minor is None or patch is None:
        return None

    return major * 1_000_000 + minor * 1_000 + patch


def get_remote_version(role: str) -> Optional[int]:
    """Return the role's version integer from the remote branch, or None.

    Unlike ``check_remote_version`` this does NOT compare against the local
    version — it always returns the remote value when readable.  Used by
    the validator's restart-forgiveness window logic, which needs to detect
    when the remote miner_version has *changed* (including when remote ==
    local after the validator itself just upgraded).

    Caller is responsible for running ``fetch_origin()`` first if a fresh
    value is required.
    """
    var_name = "miner_version" if role == "miner" else "validator_version"
    remote_source = _read_remote_version_file()
    if remote_source is None:
        return None
    return _parse_version_from_source(remote_source, var_name)


def check_remote_version(role: str) -> Optional[tuple[str, int, int]]:
    """Check if the remote has a newer version for the given role.

    Parameters
    ----------
    role:
        "miner", "validator", or "proxy" (proxy uses validator_version).

    Returns
    -------
    (remote_commit, remote_version, local_version) if update needed,
    None if up to date or on error.
    """
    if not fetch_origin():
        return None

    local_head = get_local_head()
    remote_head = get_remote_head()
    if not local_head or not remote_head:
        return None

    # No new commits at all — skip version parsing
    if local_head == remote_head:
        return None

    # New commits exist — check if our role's version changed
    var_name = "miner_version" if role == "miner" else "validator_version"

    # Local version (from the imported module)
    from neurons.version import miner_version, validator_version
    local_version = miner_version if role == "miner" else validator_version

    # Remote version (from git show, without pulling)
    remote_source = _read_remote_version_file()
    if remote_source is None:
        # Can't read remote version.py — remote doesn't have it yet, skip
        bt.logging.debug("Remote has no neurons/version.py — skipping update")
        return None

    remote_version = _parse_version_from_source(remote_source, var_name)
    if remote_version is None:
        # Can't parse version from remote — malformed file, skip
        bt.logging.warning(f"Cannot parse {var_name} from remote version.py — skipping")
        return None

    if remote_version > local_version:
        bt.logging.info(f"Remote {var_name}={remote_version} > local {local_version} (commits: {local_head[:8]}→{remote_head[:8]})")
        return (remote_head, remote_version, local_version)

    # New commits but our role's version didn't change — skip
    bt.logging.debug(f"New commits available ({local_head[:8]}→{remote_head[:8]}) but {var_name} unchanged ({local_version}) — skipping")
    return None


def pull_and_install(
    *,
    install_extras: str = "neurons",
    install_proof_cuda: bool = False,
) -> bool:
    """Pull latest code and reinstall the package.

    Returns True on success, False on failure.
    """
    branch = get_current_branch()
    if not branch:
        bt.logging.error("Cannot determine current branch for pull")
        return False

    # Pull (suppress git hints about diverging branches)
    _run_git("config", "advice.diverging", "false")
    rc, out = _run_git("pull", "origin", branch, "--ff-only")
    if rc != 0:
        bt.logging.debug(f"Fast-forward pull not possible, resetting to origin/{branch}")
        rc2, out2 = _run_git("reset", "--hard", f"origin/{branch}")
        if rc2 != 0:
            bt.logging.error(f"git reset also failed: {out2}")
            return False
        bt.logging.info(f"Reset to origin/{branch} successful")

    if not _clear_first_party_bytecode_caches():
        return False

    new_head = get_local_head()
    _head = new_head[:8] if new_head else "unknown"
    bt.logging.info(f"Pulled to {_head}")

    # Reinstall package in editable mode with the same public setup extras.
    # This picks up newly added optional dependencies instead of relying on
    # whatever extras happened to be installed in the old venv.
    extras = str(install_extras or "").strip()
    editable_target = f".[{extras}]" if extras else "."
    bt.logging.info(f"Reinstalling package ({editable_target})...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", editable_target, "--quiet"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            bt.logging.error(f"pip install failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        bt.logging.error("pip install timed out")
        return False

    bt.logging.info("Package reinstalled successfully")
    if not install_local_zkllm_wheel():
        return False
    install_local_hot_capacity_workspace_wheel(required=False)
    if install_proof_cuda and not install_local_proof_v3_cuda_wheel():
        return False
    return True


def _detect_pm2() -> Optional[str]:
    """Detect if running under PM2 and return the process name.

    PM2 sets ``pm_id`` in the environment for managed processes.
    We use ``pm2 jlist`` to find our process name by PID.
    """
    if "pm_id" not in os.environ:
        return None

    try:
        import json
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        processes = json.loads(result.stdout)
        my_pid = os.getpid()
        for proc in processes:
            if proc.get("pid") == my_pid:
                return proc.get("name")
            # Also check parent PID (PM2 might have spawned us)
            if proc.get("pid") == os.getppid():
                return proc.get("name")

        # Fallback: use pm_id to find name
        pm_id = os.environ.get("pm_id")
        for proc in processes:
            if str(proc.get("pm_id")) == pm_id:
                return proc.get("name")

    except Exception as e:
        bt.logging.debug(f"PM2 detection failed: {e}")

    return None


def restart_process() -> None:
    """Restart the current process.

    Uses PM2 if detected, otherwise re-execs via os.execv().
    This function does not return.
    """
    pm2_name = _detect_pm2()

    if pm2_name:
        bt.logging.info(f"Restarting via PM2 (process: {pm2_name})...")
        try:
            subprocess.Popen(
                ["pm2", "restart", pm2_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            bt.logging.error(f"PM2 restart failed: {e}, falling back to execv")
            _execv_restart()
        # Give PM2 a moment to register the restart, then exit
        time.sleep(2)
        sys.exit(0)
    else:
        _execv_restart()


def _execv_restart() -> None:
    """Re-exec the current process in-place (no PM2)."""
    bt.logging.info(f"Re-executing process: {sys.executable} {sys.argv}")
    os.execv(sys.executable, [sys.executable] + sys.argv)


class AutoUpdater:
    """Background thread that checks for role-specific version bumps.

    Parameters
    ----------
    role:
        "miner", "validator", or "proxy".  Determines which version
        variable is compared (``miner_version`` vs ``validator_version``).
        Proxy uses ``validator_version`` since they share the same codebase.
    check_interval:
        Seconds between update checks (default: 1800 = 30 min).
    restart_delay:
        Seconds to wait after pulling before restarting (default: 5).
        Gives in-flight requests time to complete.
    busy_check:
        Optional callback returning True if the process is busy and should
        NOT be restarted right now (e.g., validator mid-epoch-close).
    jitter_seconds:
        Maximum deterministic stagger window before restart (default: 0 =
        disabled).  Miners pass 1800 (half an epoch) so that simultaneous
        version bumps don't cause network-wide unavailability while every
        miner reloads vLLM weights.  Validators and proxies leave this at
        0 — they restart fast and a global validator restart is operator-
        observable, not a user-visible outage.
        jitter_seed:
        Bytes hashed to derive the stagger offset for this process (default
        empty).  Miners pass their 32-byte hotkey seed so the delay is
        reproducible per miner and verifiable by any third party from the
        public hotkey (no "secret stagger order" to game).
    install_extras:
        Optional extras passed to ``pip install -e`` during update.  Defaults
        mirror the public setup paths without reinstalling GPU-heavy vLLM.
    """

    def __init__(
        self,
        role: str = "validator",
        check_interval: int = 1800,
        restart_delay: int = 5,
        busy_check: Optional[Callable[[], bool]] = None,
        jitter_seconds: int = 0,
        jitter_seed: bytes = b"",
        install_extras: Optional[str] = None,
    ):
        requested_role = str(role or "validator")
        self.role = requested_role if requested_role != "proxy" else "validator"  # proxy uses validator_version
        self.install_proof_cuda = requested_role == "miner"
        self.install_extras = (
            str(install_extras).strip()
            if install_extras is not None
            else self._default_install_extras(requested_role)
        )
        self.check_interval = check_interval
        self.restart_delay = restart_delay
        self.busy_check = busy_check
        self.jitter_seconds = max(0, int(jitter_seconds))
        self.jitter_seed = bytes(jitter_seed) if jitter_seed else b""
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_state_lock = threading.Lock()
        self._update_pending = False  # Set when update deferred due to busy
        self._update_applying = False

    @property
    def drain_requested(self) -> bool:
        """Whether the caller should stop admitting new restart-sensitive work."""

        with self._update_state_lock:
            return bool(self._update_pending or self._update_applying)

    def _defer_update(self) -> None:
        with self._update_state_lock:
            self._update_applying = False
            self._update_pending = True

    def _clear_update(self) -> None:
        with self._update_state_lock:
            self._update_pending = False
            self._update_applying = False

    def _begin_update(self, *, require_pending: bool = False) -> bool:
        """Atomically claim the single installer/restart lane."""

        with self._update_state_lock:
            if self._update_applying:
                return False
            if require_pending and not self._update_pending:
                return False
            self._update_pending = False
            self._update_applying = True
            return True

    @staticmethod
    def _default_install_extras(role: str) -> str:
        normalized = str(role or "").strip().lower()
        if normalized == "miner":
            return "neurons,api,hashing"
        if normalized == "proxy":
            return "neurons,x402"
        return "neurons"

    def _compute_jitter_delay(self) -> int:
        """Return the deterministic stagger delay (seconds) for this process.

        Uniform over ``[0, jitter_seconds)`` keyed off ``jitter_seed``.  Same
        seed → same delay every call.  Empty seed or zero window → 0.
        """
        if self.jitter_seconds <= 0 or not self.jitter_seed:
            return 0
        digest = hashlib.sha256(self.jitter_seed).digest()
        return int.from_bytes(digest[:4], "big") % self.jitter_seconds

    def start(self) -> None:
        """Start the background update checker thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="auto-updater", daemon=True,
        )
        self._thread.start()

        var_name = "miner_version" if self.role == "miner" else "validator_version"
        _branch = get_current_branch() or "unknown"
        bt.logging.info(f"Auto-updater started (role={self.role}, watching={var_name}, interval={self.check_interval}s, branch={_branch})")

    def stop(self) -> None:
        """Signal the updater to stop."""
        self._stop_event.set()

    def _run(self) -> None:
        """Main loop: check → pull → restart."""
        # Initial delay — let the neuron finish startup before first check
        self._wait(60)

        while not self._stop_event.is_set():
            try:
                self._check_and_update()
            except Exception as e:
                bt.logging.error(f"Auto-update check failed: {e}")

            self._wait(self.check_interval)

    def _wait(self, seconds: int) -> None:
        """Interruptible sleep."""
        self._stop_event.wait(timeout=seconds)

    def notify_not_busy(self) -> None:
        """Call this when the process is no longer busy (e.g. epoch close done).

        If an update was deferred, applies it immediately instead of waiting
        for the next check cycle.
        """
        if not self.drain_requested:
            return
        if self.busy_check and self.busy_check():
            bt.logging.info(
                "Deferred update notification arrived while process is still "
                "busy — retaining it for the next idle window"
            )
            return
        if not self._begin_update(require_pending=True):
            return
        bt.logging.info("Deferred update ready — applying now")
        if not pull_and_install(
            install_extras=self.install_extras,
            install_proof_cuda=self.install_proof_cuda,
        ):
            bt.logging.error("Deferred update failed — will retry next cycle")
            self._clear_update()
            return
        bt.logging.info(f"Update applied, restarting in {self.restart_delay}s...")
        self._wait(self.restart_delay)
        if self.busy_check and self.busy_check():
            bt.logging.info(
                "Process became busy during deferred-update restart delay — "
                "retaining update for the next idle window"
            )
            self._defer_update()
            return
        self._stagger_sleep()
        if self.busy_check and self.busy_check():
            bt.logging.info(
                "Process became busy during update stagger — retaining update "
                "for the next idle window"
            )
            self._defer_update()
            return
        restart_process()

    def _stagger_sleep(self) -> None:
        """Sleep a deterministic per-process delay before restarting.

        Defeats the simultaneous-restart outage: with N miners on a fixed
        version bump, each picks a unique offset in [0, jitter_seconds) so
        that at any instant only a fraction of the network is reloading
        vLLM weights.  Stop signal interrupts the sleep cleanly.
        """
        delay = self._compute_jitter_delay()
        if delay <= 0:
            return
        bt.logging.info(
            f"Auto-update: deterministic stagger {delay}s before restart "
            f"(role={self.role}, window={self.jitter_seconds}s)"
        )
        self._wait(delay)

    def _check_and_update(self) -> None:
        """Single update check cycle."""
        result = check_remote_version(self.role)
        if result is None:
            bt.logging.debug(f"No update needed for role={self.role}")
            return

        remote_commit, remote_ver, local_ver = result
        var_name = "miner_version" if self.role == "miner" else "validator_version"
        bt.logging.info(f"Update required: {var_name} {local_ver} → {remote_ver} (remote commit {remote_commit[:8]})")

        # Check if we're busy
        if self.busy_check and self.busy_check():
            bt.logging.info("Process is busy — update scheduled for next idle window")
            self._defer_update()
            return

        if not self._begin_update():
            return
        if not pull_and_install(
            install_extras=self.install_extras,
            install_proof_cuda=self.install_proof_cuda,
        ):
            bt.logging.error("Update failed — will retry next cycle")
            self._clear_update()
            return

        bt.logging.info(f"Update applied, restarting in {self.restart_delay}s...")
        self._wait(self.restart_delay)

        if self.busy_check and self.busy_check():
            bt.logging.info("Process became busy during restart delay — deferring")
            self._defer_update()
            return

        self._stagger_sleep()
        if self.busy_check and self.busy_check():
            bt.logging.info(
                "Process became busy during update stagger — deferring restart"
            )
            self._defer_update()
            return
        restart_process()
