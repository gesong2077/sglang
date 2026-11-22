"""Opt-in read-only snapshots, triggered only by a one-token diagnostic prefill.

No cache clearing, allocator reset, or inference arithmetic changes. The normal
launcher leaves SGLANG_XPU_MUSE_HOST_MEMORY_AUDIT unset.
"""
import ctypes as C
import json
import os
from pathlib import Path
import time

import torch

_last_phase = None


def snapshot(model, directory):
    global _last_phase
    directory = Path(directory)
    trigger = directory / 'trigger.json'
    if not trigger.exists():
        return
    phase = json.loads(trigger.read_text())['phase']
    if phase == _last_phase:
        return
    if not phase.replace('_', '').isalnum():
        raise ValueError('Invalid diagnostic snapshot phase')
    _last_phase = phase
    state = getattr(model, '_xpu_streamed_mlp', None)
    tensors = [t for pair in state.hosts for t in pair] if state is not None else []
    slots = [t for pair in state.slots for t in pair] if state is not None else []
    result = dict(phase=phase, pid=os.getpid(), device=torch.xpu.current_device(),
                  wall=time.time(), streamed_state_present=state is not None,
                  dedicated_host_tensor_count=len(tensors),
                  dedicated_host_logical_bytes=sum(t.numel()*t.element_size() for t in tensors),
                  dedicated_host_tensors_all_pinned=all(t.is_pinned() for t in tensors) if tensors else None,
                  dedicated_device_slot_bytes=sum(t.numel()*t.element_size() for t in slots))
    lib = C.CDLL(os.environ['SGLANG_XPU_HOST_STATS_LIB'])
    lib.host_stats.argtypes = [C.POINTER(C.c_int64)]
    lib.host_stats.restype = C.c_int
    lib.host_stats_error.restype = C.c_char_p
    values = (C.c_int64 * 8)()
    if lib.host_stats(values):
        result['torch_host_allocator_error'] = lib.host_stats_error().decode()
    else:
        result['torch_xpu_host_allocator'] = dict(zip(
            ['active_blocks','active_bytes','reserved_blocks','reserved_bytes',
             'peak_active_bytes','peak_reserved_bytes','num_alloc','num_free'],
            list(values)))
    result['scope'] = 'PyTorch XPU pinned-host allocator only; excludes other libraries/driver allocations. Logical tensor bytes exclude allocator rounding. VmLck is not a complete driver pin count.'
    output = directory / f'{phase}-device{result["device"]}.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output)
