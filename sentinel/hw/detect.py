"""Read-only hardware and service detection for ``sentinel doctor``.

Answers three questions, without changing anything on the machine:

  1. What accelerator can run local models?  (NVIDIA / AMD / Apple / none)
  2. What is already running?                 (Frigate, MQTT, Ollama, ...)
  3. Which cloud API keys are available?      (presence only, never values)

Every external command goes through an injectable ``Runner`` and every file
read through ``read_text``, so the detection logic is tested against recorded
output from machines the test suite never runs on.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

Runner = Callable[[list[str], float], tuple[int, str]]

# Cloud providers we can configure, by the variable that holds their key.
API_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
)


def default_runner(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run a command and return (returncode, stdout); never raises."""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return 127, ""
    except (subprocess.TimeoutExpired, OSError):
        return 124, ""
    return proc.returncode, proc.stdout or ""


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


@dataclass
class GPUInfo:
    vendor: str  # nvidia | amd | intel | apple | other
    name: str
    vram_gb: float | None = None
    integrated: bool = False
    driver: str | None = None


@dataclass
class ServiceProbe:
    name: str
    url: str
    reachable: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)


@dataclass
class HardwareReport:
    os: str
    os_version: str
    arch: str
    cpu: str
    cpu_threads: int
    ram_gb: float
    gpus: list[GPUInfo] = field(default_factory=list)
    npu: str | None = None
    docker: dict[str, bool] = field(default_factory=dict)
    api_keys: dict[str, bool] = field(default_factory=dict)
    services: list[ServiceProbe] = field(default_factory=list)

    @property
    def accelerator(self) -> str:
        """The best local-model accelerator: nvidia, amd, apple or none.

        Integrated Intel graphics and NPUs are reported but not counted:
        common local model servers do not use them out of the box.
        """
        vendors = {g.vendor for g in self.gpus if not g.integrated or g.vendor == "apple"}
        for vendor in ("nvidia", "apple", "amd"):
            if vendor in vendors:
                return vendor
        return "none"

    @property
    def accelerator_memory_gb(self) -> float | None:
        """Memory available to models on the accelerator (unified on Apple)."""
        acc = self.accelerator
        if acc == "apple":
            return self.ram_gb
        sizes = [g.vram_gb for g in self.gpus if g.vendor == acc and g.vram_gb]
        return max(sizes) if sizes else None

    @property
    def cloud_keys(self) -> list[str]:
        return [k for k, present in self.api_keys.items() if present]

    def service(self, name: str) -> ServiceProbe | None:
        return next((s for s in self.services if s.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["accelerator"] = self.accelerator
        out["accelerator_memory_gb"] = self.accelerator_memory_gb
        return out


# ---------------------------------------------------------------- hardware


def detect_hardware(
    *,
    runner: Runner = default_runner,
    read_text: Callable[[str], str | None] = _read_text,
    system: str | None = None,
    machine: str | None = None,
    env: dict[str, str] | None = None,
    sys_root: str = "/",
    which: Callable[[str], str | None] = shutil.which,
) -> HardwareReport:
    system = system or platform.system()
    machine = (machine or platform.machine() or "").lower()
    env = dict(os.environ) if env is None else env

    report = HardwareReport(
        os=system,
        os_version=platform.release() if system == platform.system() else "",
        arch=machine,
        cpu=_cpu_name(system, runner, read_text) or platform.processor() or "unknown CPU",
        cpu_threads=os.cpu_count() or 1,
        ram_gb=_ram_gb(system, runner, read_text),
    )

    report.gpus.extend(_nvidia_gpus(runner))
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        report.gpus.append(
            GPUInfo(vendor="apple", name=report.cpu, vram_gb=report.ram_gb, integrated=True)
        )
        report.npu = "Apple Neural Engine"
    elif system == "Windows":
        report.gpus.extend(_windows_other_gpus(runner, report.gpus))
        report.npu = _windows_npu(runner)
    elif system == "Linux":
        report.gpus.extend(_amd_rocm_gpus(runner))
        report.gpus.extend(_linux_drm_gpus(sys_root, read_text, report.gpus))
        if Path(sys_root, "dev/accel/accel0").exists():
            report.npu = "NPU (/dev/accel/accel0)"

    report.docker = _docker_status(runner, which)
    report.api_keys = {var: bool(env.get(var)) for var in API_KEY_VARS}
    return report


def _cpu_name(system: str, runner: Runner, read_text: Callable[[str], str | None]) -> str:
    if system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            return ""
    if system == "Darwin":
        rc, out = runner(["sysctl", "-n", "machdep.cpu.brand_string"], 3.0)
        return out.strip() if rc == 0 else ""
    text = read_text("/proc/cpuinfo") or ""
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _ram_gb(system: str, runner: Runner, read_text: Callable[[str], str | None]) -> float:
    if system == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return round(status.ullTotalPhys / 1024**3, 1)
        except (AttributeError, OSError):
            pass
        return 0.0
    if system == "Darwin":
        rc, out = runner(["sysctl", "-n", "hw.memsize"], 3.0)
        return round(int(out.strip()) / 1024**3, 1) if rc == 0 and out.strip().isdigit() else 0.0
    text = read_text("/proc/meminfo") or ""
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    return round(int(match.group(1)) / 1024**2, 1) if match else 0.0


def _nvidia_gpus(runner: Runner) -> list[GPUInfo]:
    rc, out = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        5.0,
    )
    if rc != 0:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            vram = round(float(parts[1]) / 1024, 1)  # MiB -> GiB
        except ValueError:
            vram = None
        gpus.append(
            GPUInfo(
                vendor="nvidia",
                name=parts[0],
                vram_gb=vram,
                driver=parts[2] if len(parts) > 2 else None,
            )
        )
    return gpus


def _amd_rocm_gpus(runner: Runner) -> list[GPUInfo]:
    rc, out = runner(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"], 5.0)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    gpus = []
    for card, info in sorted(data.items()):
        if not isinstance(info, dict) or not card.lower().startswith("card"):
            continue
        name = next(
            (str(v) for k, v in info.items() if k in ("Card Series", "Card series", "Card SKU", "Marketing Name")),
            "AMD GPU",
        )
        vram_raw = next(
            (v for k, v in info.items() if "vram total memory" in k.lower()), None
        )
        try:
            vram = round(int(vram_raw) / 1024**3, 1) if vram_raw is not None else None
        except (TypeError, ValueError):
            vram = None
        gpus.append(GPUInfo(vendor="amd", name=name, vram_gb=vram))
    return gpus


def _vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    if "intel" in lowered:
        return "intel"
    return "other"


def _windows_other_gpus(runner: Runner, known: list[GPUInfo]) -> list[GPUInfo]:
    rc, out = runner(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion "
            "| ConvertTo-Json -Compress",
        ],
        10.0,
    )
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    rows = data if isinstance(data, list) else [data]
    have_nvidia = any(g.vendor == "nvidia" for g in known)
    gpus = []
    for row in rows:
        name = str(row.get("Name") or "").strip()
        if not name or name.lower().startswith("microsoft basic"):
            continue
        vendor = _vendor_from_name(name)
        if vendor == "nvidia" and have_nvidia:
            continue  # nvidia-smi already reported it, with accurate VRAM
        # AdapterRAM is a 32-bit field and meaningless for shared memory, so
        # it is only trusted for discrete cards.
        integrated = vendor == "intel" and "arc" not in name.lower()
        gpus.append(
            GPUInfo(
                vendor=vendor,
                name=name,
                vram_gb=None,
                integrated=integrated,
                driver=row.get("DriverVersion"),
            )
        )
    return gpus


def _windows_npu(runner: Runner) -> str | None:
    rc, out = runner(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            # \b matters: a bare 'NPU' also matches "Input" (PowerShell -match
            # is case-insensitive).
            "(Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match '\\bNPU\\b|AI Boost' } "
            "| Select-Object -First 1 -ExpandProperty Name)",
        ],
        10.0,
    )
    name = out.strip()
    return name if rc == 0 and name else None


_PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}


def _linux_drm_gpus(
    sys_root: str, read_text: Callable[[str], str | None], known: list[GPUInfo]
) -> list[GPUInfo]:
    drm = Path(sys_root, "sys/class/drm")
    if not drm.is_dir():
        return []
    known_vendors = {g.vendor for g in known}
    gpus = []
    for card in sorted(drm.glob("card[0-9]*")):
        if "-" in card.name:  # connectors such as card0-HDMI-A-1
            continue
        vendor_id = (read_text(str(card / "device" / "vendor")) or "").strip().lower()
        vendor = _PCI_VENDORS.get(vendor_id)
        if vendor is None or vendor in known_vendors:
            continue
        known_vendors.add(vendor)
        gpus.append(
            GPUInfo(
                vendor=vendor,
                name=f"{vendor.upper() if vendor == 'amd' else vendor.title()} GPU ({card.name})",
                integrated=vendor == "intel",
            )
        )
    return gpus


def _docker_status(runner: Runner, which: Callable[[str], str | None]) -> dict[str, bool]:
    if not which("docker"):
        return {"installed": False, "running": False}
    rc, _ = runner(["docker", "info", "--format", "{{.ServerVersion}}"], 8.0)
    return {"installed": True, "running": rc == 0}


# ---------------------------------------------------------------- services


def _ollama_url(env: dict[str, str]) -> str:
    raw = (env.get("OLLAMA_URL") or env.get("OLLAMA_HOST") or "").strip()
    if not raw:
        return "http://localhost:11434"
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    host = raw.replace("0.0.0.0", "localhost")
    return host.rstrip("/")


async def probe_services(
    *,
    frigate_url: str = "http://localhost:5000",
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
    env: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 1.5,
) -> list[ServiceProbe]:
    """Probe the usual local services concurrently. Unreachable is not an error."""
    env = dict(os.environ) if env is None else env
    owns = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    ollama = _ollama_url(env)

    async def get_json(url: str) -> tuple[bool, Any, str]:
        try:
            r = await http.get(url)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False, None, "not reachable"
        except httpx.HTTPError as exc:
            return False, None, type(exc).__name__
        if r.status_code >= 400:
            return False, None, f"HTTP {r.status_code}"
        try:
            return True, r.json(), "ok"
        except ValueError:
            return True, r.text, "ok"

    async def frigate() -> ServiceProbe:
        url = f"{frigate_url.rstrip('/')}/api/version"
        ok, body, detail = await get_json(url)
        version = body if isinstance(body, str) else json.dumps(body) if body else ""
        return ServiceProbe("frigate", frigate_url, ok, f"version {version.strip()}" if ok else detail)

    async def ollama_probe() -> ServiceProbe:
        ok, body, detail = await get_json(f"{ollama}/api/tags")
        models = (
            sorted(str(m.get("name")) for m in body.get("models", []))
            if ok and isinstance(body, dict)
            else []
        )
        return ServiceProbe("ollama", ollama, ok, f"{len(models)} model(s)" if ok else detail, models)

    async def openai_server(name: str, url: str) -> ServiceProbe:
        ok, body, detail = await get_json(f"{url}/v1/models")
        models = (
            sorted(str(m.get("id")) for m in body.get("data", []))
            if ok and isinstance(body, dict)
            else []
        )
        return ServiceProbe(name, url, ok, f"{len(models)} model(s)" if ok else detail, models)

    async def mqtt() -> ServiceProbe:
        target = f"{mqtt_host}:{mqtt_port}"
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(mqtt_host, mqtt_port), timeout=timeout
            )
            writer.close()
            return ServiceProbe("mqtt", target, True, "accepting connections")
        except (OSError, asyncio.TimeoutError):
            return ServiceProbe("mqtt", target, False, "not reachable")

    try:
        return list(
            await asyncio.gather(
                frigate(),
                mqtt(),
                ollama_probe(),
                openai_server("lm_studio", "http://localhost:1234"),
                openai_server("llama_cpp", "http://localhost:8080"),
                openai_server("vllm", "http://localhost:8000"),
            )
        )
    finally:
        if owns:
            await http.aclose()
