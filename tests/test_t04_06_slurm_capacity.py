import pytest

from tools.t04_06_slurm_capacity import parse_node


NODE = (
    "NodeName=csc1 State=MIXED "
    "CPUAlloc=1 CPUTot=16 "
    "RealMemory=192000 AllocMem=16384 "
    "CfgTRES=cpu=16,mem=187.50G,gres/gpu=14,"
    "gres/gpu:nvidia_a100_80gb_pcie_1g.10gb=14 "
    "AllocTRES=cpu=1,mem=16G,gres/gpu=1,"
    "gres/gpu:nvidia_a100_80gb_pcie_1g.10gb=1"
)


def test_one_cpu_capacity():
    result = parse_node(NODE)
    assert result["free_cpus"] == 15
    assert result["free_mig_gpus"] == 13
    assert result["physical_capacity"] == 13


def test_two_cpu_capacity():
    result = parse_node(NODE, cpus_per_job=2)
    assert result["physical_capacity"] == 7


def test_memory_limits_capacity():
    text = NODE.replace("AllocMem=16384", "AllocMem=180000")
    result = parse_node(text)
    assert result["physical_capacity"] == 0


def test_wrong_node_rejected():
    with pytest.raises(ValueError, match="Unexpected"):
        parse_node(NODE.replace("NodeName=csc1", "NodeName=other"))


def test_missing_gpu_resource_rejected():
    text = NODE.replace(
        ",gres/gpu:nvidia_a100_80gb_pcie_1g.10gb=14", ""
    )
    with pytest.raises(ValueError, match="MIG resource"):
        parse_node(text)


def test_drained_node_rejected():
    with pytest.raises(RuntimeError, match="scheduling state"):
        parse_node(NODE.replace("State=MIXED", "State=DRAIN"))


def test_invalid_profile_rejected():
    with pytest.raises(ValueError, match="Invalid resource"):
        parse_node(NODE, cpus_per_job=0)


def test_mixed_draining_node_rejected():
    with pytest.raises(RuntimeError, match="scheduling state"):
        parse_node(NODE.replace("State=MIXED", "State=MIXED+DRAIN"))
