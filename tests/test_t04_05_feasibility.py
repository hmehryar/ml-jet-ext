from tools.t04_05_feasibility import classify_failure


def test_oom_classification():
    assert classify_failure(RuntimeError('CUDA out of memory')) == 'oom'
    assert classify_failure(RuntimeError('CUBLAS_STATUS_ALLOC_FAILED')) == 'oom'
    assert classify_failure(ValueError('shape mismatch')) == 'error'
