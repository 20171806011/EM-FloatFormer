"""Process isolation and host CPU accounting for the development latency run."""

from contextlib import contextmanager
import ctypes
import os
import struct
import time

import psutil


def project_workers():
    """Recognize study and audit launchers without recording their command lines."""
    blocked, unreadable = [], []
    for proc in psutil.process_iter(["name"]):
        if proc.pid == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        try:
            args = [a.lower().replace("\\", "/") for a in proc.cmdline()[1:]]
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            unreadable.append(proc.pid)
            continue
        tokens = ("p6.run_dev_iter2", "p6.verify_iter2_checkpoints", "p6.tests", "p6.diagnostics",
                  "p6.extended_audit", "p6.legacy_regression", "p6.study", "p6.verify_dev",
                  "publication.verify_p4", "p6.run_dev_iter4", "p6.verify_iter4",
                  "p6.iter4_diagnostics", "p6.write_dev_iter4")
        paths = tuple(t.replace(".", "/") for t in tokens)
        is_project = any(a.startswith(tokens) or (a.endswith(".py") and any(t in a for t in paths))
                         or (not a.startswith("-") and a.rstrip("/") in paths) for a in args)
        if is_project:
            blocked.append(proc.pid)
    if blocked:
        raise RuntimeError(f"Concurrent study, audit or benchmark process(es): {blocked}")
    return dict(concurrent_project_processes=[], unreadable_python_pids=unreadable)


def windows_cpu_sets():
    """Read the documented fixed prefix of each variable-length Windows CPU set."""
    if os.name != "nt" or (os.cpu_count() or 0) > 64:
        return []
    needed = ctypes.c_ulong()
    query = ctypes.windll.kernel32.GetSystemCpuSetInformation
    query(None, 0, ctypes.byref(needed), None, 0)
    if not needed.value:
        return []
    buf = ctypes.create_string_buffer(needed.value)
    if not query(buf, needed.value, ctypes.byref(needed), None, 0):
        return []
    data = buf.raw[:needed.value]
    sets = []
    offset = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            return []
        if kind == 0 and size >= 20:
            group, logical, core, _, _, efficiency, flags = struct.unpack_from("<H6B", data, offset+12)
            if group == 0 and not flags & 3:  # parked or reserved for a process
                sets.append(dict(logical=logical, core=core, efficiency_class=efficiency))
        offset += size
    return sets


def snapshot():
    host = psutil.cpu_times()
    proc = os.times()
    # Windows GetSystemTimes includes interrupt/DPC in kernel CPU time.
    busy = host.user + host.system + sum(getattr(host, k, 0.) for k in
                                         ("nice", "irq", "softirq", "steal"))
    return (time.perf_counter(), busy, busy + host.idle + getattr(host, "iowait", 0.),
            proc.user, proc.system)


def interval(start, end):
    wall, busy, total, user, system = [b-a for a,b in zip(start, end)]
    if total <= 0:
        raise RuntimeError("No host CPU interval available")
    return dict(wall_s=wall, process_user_s=user, process_system_s=system,
                host_busy_percent=100 * busy / total,
                background_percent_of_host=100 * max(0., busy - user - system) / total)


@contextmanager
def benchmark_environment():
    before = project_workers()
    proc = psutil.Process()
    original = proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None
    start = snapshot()
    core_load = psutil.cpu_percent(interval=1., percpu=True)
    baseline = interval(start, snapshot())
    choice = None
    candidates = []
    if original:
        try:
            cores = [c for c in windows_cpu_sets() if c["logical"] in original and c["logical"] < len(core_load)]
            if cores:
                highest = max(c["efficiency_class"] for c in cores)
                candidates = [c for c in cores if c["efficiency_class"] == highest]
                for c in candidates:
                    siblings = [p for p in cores if p["core"] == c["core"]]
                    c["physical_core_load_sum"] = sum(core_load[p["logical"]] for p in siblings)
                choice = min(candidates, key=lambda c: (c["physical_core_load_sum"],
                                                        core_load[c["logical"]], c["logical"]))
                proc.cpu_affinity([choice["logical"]])
        except (AttributeError, OSError, psutil.Error, ValueError):
            choice = None
    try:
        yield dict(project_scan_before=before, baseline=baseline, logical_load_percent=core_load,
                   candidate_cpu_sets=candidates, selected_cpu_set=choice, initial_affinity=original,
                   effective_affinity=proc.cpu_affinity() if original else None,
                   host_logical_cpus=psutil.cpu_count(), host_physical_cores=psutil.cpu_count(False))
    finally:
        if choice:
            proc.cpu_affinity(original)
