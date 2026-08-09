"""Threaded batch prefetcher to overlap CPU-bound batch preparation with GPU compute.

The 2D AE training step is serialized: building one batch (single-threaded slice
extraction + per-sample MONAI augment + ``F.interpolate``) costs ~120 ms on one core,
while the actual ``train_step`` is only a few ms on the GPU. The GPU therefore starves.

:class:`BatchPrefetcher` spawns N daemon worker threads, each of which repeatedly builds
a batch via a caller-provided ``make_batch(worker_rng, worker_aug)`` callable and pushes
it onto a bounded queue. The main training loop pops ready batches with :meth:`get`, so
batch preparation overlaps the GPU's training work.

Thread-safety is the caller's responsibility for any shared mutable state. The standard
recipe (see ``train_ae2d.py``) is:

  * give each worker its **own** ``np.random.RandomState`` (seeded deterministically as
    ``base_seed + worker_index``) -- numpy RandomState is not thread-safe and the main
    loop's rng is also live,
  * give each worker its **own** built augment transform (MONAI ``Rand*d`` objects carry
    internal RNG state),
  * serialize any shared cursor/loader advance (e.g. ``cache.next_train()``) inside the
    ``make_batch`` callable with a ``threading.Lock``.

Enabling workers makes the data-sampling RNG draw order NONDETERMINISTIC relative to the
single-thread path -- batches are still independently sampled and augmented, just in a
nondeterministic interleaving. This is an accepted trade-off for a perf-only code path.

Worker threads should NOT touch CUDA directly for their own allocations; let the main
thread own device placement of model/optimizer. (Any H2D copy reached transitively via
``make_batch`` must itself be serialized by the caller's lock.)
"""

import queue
import threading


class BatchPrefetcher:
    """Bounded, multi-worker background batch producer.

    Parameters
    ----------
    make_batch:
        Callable ``make_batch(worker_rng, worker_aug, worker_state) -> batch``. Invoked
        repeatedly on each worker thread; whatever it returns is handed to the main loop
        verbatim. ``worker_state`` is a private mutable dict (one per worker, persisted
        across calls) for amortizing expensive per-worker resources -- e.g. holding a
        loaded patient and a remaining-reuse count so a single load feeds K batches.
    num_workers:
        Number of daemon worker threads to spawn (must be >= 1; the caller gates the
        whole prefetcher behind ``num_workers > 0`` and keeps the synchronous path
        otherwise).
    make_rng:
        Callable ``make_rng(worker_index) -> rng`` building a per-worker RNG. Each worker
        owns its rng so workers don't share non-thread-safe RandomState.
    make_aug:
        Callable ``make_aug() -> augment`` building a per-worker augment transform. Each
        worker owns its augment so MONAI ``Rand*d`` internal RNG state isn't shared.
    """

    def __init__(self, make_batch, num_workers, make_rng, make_aug):
        if num_workers < 1:
            raise ValueError("BatchPrefetcher requires num_workers >= 1.")
        self._make_batch = make_batch
        self._num_workers = int(num_workers)
        self._make_rng = make_rng
        self._make_aug = make_aug
        # Bounded queue so workers block (back-pressure) once 2*N batches are buffered,
        # capping memory while still keeping the GPU fed.
        self._q = queue.Queue(maxsize=2 * self._num_workers)
        self._stop = threading.Event()
        self._exc = None
        self._exc_lock = threading.Lock()
        self._workers = []
        for w in range(self._num_workers):
            t = threading.Thread(
                target=self._run, args=(w,), name=f"batch-prefetch-{w}", daemon=True
            )
            self._workers.append(t)
        for t in self._workers:
            t.start()

    def _run(self, worker_index):
        # Each worker owns its rng + augment so no non-thread-safe state is shared.
        worker_rng = self._make_rng(worker_index)
        worker_aug = self._make_aug()
        # Private per-worker scratch dict, persisted across make_batch calls. Lets a
        # worker amortize an expensive resource (e.g. a loaded patient reused for K
        # batches) without sharing state across workers. make_batch must accept it.
        worker_state = {}
        while not self._stop.is_set():
            try:
                batch = self._make_batch(worker_rng, worker_aug, worker_state)
            except BaseException as exc:  # surface to the main thread via get()
                with self._exc_lock:
                    if self._exc is None:
                        self._exc = exc
                self._stop.set()
                return
            # Block (with timeout) until there's room or we're asked to stop, so a
            # blocked put() can still observe the stop flag and exit cleanly.
            while not self._stop.is_set():
                try:
                    self._q.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _raise_worker_exc(self):
        with self._exc_lock:
            exc, self._exc = self._exc, None
        if exc is not None:
            raise exc

    def get(self):
        """Pop the next ready batch, blocking until one is available.

        Re-raises any exception captured on a worker thread so failures aren't silently
        swallowed. Raises ``RuntimeError`` if every worker has stopped and the queue has
        drained (e.g. all workers died), to avoid deadlocking the training loop.
        """
        while True:
            try:
                return self._q.get(timeout=0.1)
            except queue.Empty:
                self._raise_worker_exc()
                if not any(t.is_alive() for t in self._workers):
                    raise RuntimeError("All batch-prefetch workers stopped unexpectedly.")

    def close(self):
        """Signal workers to stop, drain the queue, and join. Safe to call repeatedly."""
        self._stop.set()
        # Drain so a worker blocked on put() wakes and observes the stop flag.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        for t in self._workers:
            t.join(timeout=10.0)
        self._workers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
