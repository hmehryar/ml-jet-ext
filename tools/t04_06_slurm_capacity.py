"""Read-only CSC1 Slurm resource inspection for T04-06 v2."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

GPU_TYPE = "nvidia_a100_80gb_pcie_1g.10gb"
GPU_KEY = "gres/gpu:" + GPU_TYPE
NODE = "csc1"


def parse_node(output, *, cpus_per_job=1, mem_gib=12):
    """Calculate physical headroom from one scontrol node record."""
    fields = dict(re.findall(r"([A-Za-z][A-Za-z0-9]*)=(\S*)", output))

    required = (
        "NodeName", "State", "CPUTot", "CPUAlloc",
        "RealMemory", "AllocMem", "CfgTRES", "AllocTRES"
    )
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError("Missing Slurm node fields: " + str(missing))

    def tres_map(raw):
        result = {}
        for item in raw.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                result[key] = value
        return result

    configured = tres_map(fields["CfgTRES"])
    allocated = tres_map(fields["AllocTRES"])

    if GPU_KEY not in configured:
        raise ValueError("Expected typed A100 MIG resource is unavailable")

    if cpus_per_job < 1 or mem_gib < 1:
        raise ValueError("Invalid resource profile")

    if fields["NodeName"] != NODE:
        raise ValueError("Unexpected Slurm node")

    states = set(fields["State"].split("+"))
    if states & {
        "DRAIN", "DRAINING", "DRAINED", "DOWN",
        "FAIL", "FAILING", "MAINT", "NO_RESPOND",
    } or not states & {"IDLE", "MIXED", "ALLOCATED"}:
        raise RuntimeError("Node is not in an accepted scheduling state")

    cpu_free = int(fields["CPUTot"]) - int(fields["CPUAlloc"])

    # Slurm reports RealMemory and AllocMem in MiB.
    memory_free_mib = (
        int(fields["RealMemory"]) - int(fields["AllocMem"])
    )

    gpu_free = (
        int(configured[GPU_KEY]) -
        int(allocated.get(GPU_KEY, allocated.get("gres/gpu", "0")))
    )

    if min(cpu_free, memory_free_mib, gpu_free) < 0:
        raise ValueError("Inconsistent Slurm resource allocation")

    memory_per_job_mib = mem_gib * 1024

    return {
        "node": fields["NodeName"],
        "state": fields["State"],
        "free_cpus": cpu_free,
        "free_memory_mib": memory_free_mib,
        "free_mig_gpus": gpu_free,
        "cpus_per_job": cpus_per_job,
        "memory_per_job_gib": mem_gib,
        "additional_jobs_by_cpu": cpu_free // cpus_per_job,
        "additional_jobs_by_memory": (
            memory_free_mib // memory_per_job_mib
        ),
        "additional_jobs_by_gpu": gpu_free,
        "physical_capacity": min(
            cpu_free // cpus_per_job,
            memory_free_mib // memory_per_job_mib,
            gpu_free,
        ),
    }


def inspect(node=NODE, *, cpus_per_job=1, mem_gib=12):
    if node != NODE:
        raise ValueError("Only CSC1 is supported")

    result = subprocess.run(
        ["scontrol", "show", "node", "-o", node],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    return parse_node(
        result.stdout,
        cpus_per_job=cpus_per_job,
        mem_gib=mem_gib,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus-per-job", type=int, default=1)
    parser.add_argument("--mem-gib", type=int, default=12)
    args = parser.parse_args()

    result = inspect(
        cpus_per_job=args.cpus_per_job,
        mem_gib=args.mem_gib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("READ_ONLY: no jobs submitted or reserved")


if __name__ == "__main__":
    main()
