import torch
from collections import deque

class CUDAPrefetcher:
    def __init__(self, loader, device, n_prefetch=4):
        assert device.type.startswith("cuda")
        self.loader = loader
        self.device = device
        self.n_prefetch = n_prefetch
        self.stream = torch.cuda.Stream()

    def _move_to_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, dict):
            return {k: self._move_to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            out = [self._move_to_device(v) for v in x]
            return type(x)(out) if isinstance(x, tuple) else out
        return x

    def __iter__(self):
        self._it = iter(self.loader)
        self._queue = deque()
        # Pre-fill queue with n_prefetch batches on a background CUDA stream
        with torch.cuda.stream(self.stream):
            for _ in range(self.n_prefetch):
                try:
                    b = next(self._it)
                except StopIteration:
                    break
                b = self._move_to_device(b)
                ev = torch.cuda.Event()
                ev.record(self.stream)
                self._queue.append((b, ev))
        return self

    def __next__(self):
        if not self._queue:
            raise StopIteration
        # Schedule next copy
        try:
            nxt = next(self._it)
            with torch.cuda.stream(self.stream):
                nxt = self._move_to_device(nxt)
                evn = torch.cuda.Event()
                evn.record(self.stream)
                self._queue.append((nxt, evn))
        except StopIteration:
            pass
        # Wait for the oldest staged batch to be ready on the current stream
        batch, ev = self._queue.popleft()
        torch.cuda.current_stream().wait_event(ev)
        return batch
