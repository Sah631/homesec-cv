from queue import Queue


class SlidingQueue(Queue):
    """
    Fixed-size queue where put() drops the oldest item when full.

    Good for real-time frame queues where newest data matters more than old data.
    Do not use this for clip jobs, storage jobs, metadata writes, etc.
    """

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("SlidingQueue requires maxsize > 0")
        super().__init__(maxsize=maxsize)

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            if self._qsize() >= self.maxsize:
                self._get()  # drop oldest item

                # Only relevant if using queue.join()/task_done().
                if self.unfinished_tasks > 0:
                    self.unfinished_tasks -= 1

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def put_nowait(self, item):
        return self.put(item, block=False)
