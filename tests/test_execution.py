import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from sm9rrsfl.execution import completed_pairs, device_affine_futures


class DeviceQueueTest(unittest.TestCase):
    def test_eight_devices_keep_one_active_job_each(self):
        active = set()
        peaks = []
        lock = threading.Lock()

        def work(task):
            device, index = task
            with lock:
                self.assertNotIn(device, active)
                active.add(device)
                peaks.append(len(active))
            time.sleep(.01 if device == 0 else .002)
            with lock:
                active.remove(device)
            return task

        tasks = [(i % 8, i) for i in range(40)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(completed_pairs(device_affine_futures(
                executor, tasks, submit=lambda pool, t: pool.submit(work, t),
                device_key=lambda t: t[0], jobs=8)))
        self.assertEqual({f.result() for f, _ in results}, set(tasks))
        self.assertGreater(max(peaks), 1)
        self.assertFalse(active)

    def test_more_devices_than_cpu_slots_does_not_drop_tasks(self):
        tasks = list(range(16))
        with ThreadPoolExecutor(max_workers=2) as pool:
            pairs = device_affine_futures(pool, tasks,
                submit=lambda p, t: p.submit(lambda: t), device_key=lambda t: t % 8, jobs=2)
            values = [f.result() for f, _ in pairs]
        self.assertEqual(sorted(values), tasks)

    def test_failure_remains_visible_to_snapshot_caller(self):
        def fail():
            raise RuntimeError("device failure")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pairs = device_affine_futures(pool, [0], submit=lambda p, t: p.submit(fail),
                                          device_key=lambda _: "cuda:0", jobs=1)
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                for future, _ in pairs:
                    future.result()

    def test_resume_with_one_remaining_device_does_not_multiply_its_slots(self):
        active = 0
        def work():
            nonlocal active
            self.assertEqual(active, 0)
            active += 1
            time.sleep(.002)
            active -= 1
        with ThreadPoolExecutor(max_workers=8) as pool:
            pairs = device_affine_futures(pool, list(range(8)),
                    submit=lambda p, _: p.submit(work), device_key=lambda _: "cuda:0", jobs=8)
            for future, _ in pairs:
                future.result()
