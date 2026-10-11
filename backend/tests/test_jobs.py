import threading
import time

from vmb_backend.jobs import EventBus, Job, JobQueue


def run_until_done(queue: JobQueue, jobs: list[Job], timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(j.status in ("done", "failed") for j in jobs):
            return
        time.sleep(0.01)
    raise AssertionError([(j.id, j.status) for j in jobs])


def test_gpu_jobs_grouped_by_loaded_model():
    loaded = {"model": None}
    order: list[str] = []
    gate = threading.Event()
    queue = JobQueue(EventBus(), cpu_workers=1, loaded_model=lambda: loaded["model"])

    def handler(job: Job) -> None:
        gate.wait()
        loaded["model"] = job.model_key
        order.append(job.id)

    queue.register("motion", handler)
    jobs = [Job(id=i, type="motion", resource="gpu", model_key=m) for i, m in
            [("a1", "A"), ("b1", "B"), ("a2", "A"), ("b2", "B"), ("a3", "A")]]
    for j in jobs:
        queue.submit(j)
    queue.start()
    gate.set()
    run_until_done(queue, jobs)
    queue.stop()
    # a1 の後は A が読み込まれているので A を先にまとめ、次に B
    assert order == ["a1", "a2", "a3", "b1", "b2"]


def test_dependency_failure_propagates():
    queue = JobQueue(EventBus(), cpu_workers=1)

    def fail(job: Job) -> None:
        raise RuntimeError("boom")

    queue.register("motion", fail)
    queue.register("convert", lambda job: None)
    parent = Job(id="p", type="motion", resource="gpu", model_key="A")
    child = Job(id="c", type="convert", resource="cpu", depends_on=["p"])
    queue.submit(child)
    queue.submit(parent)
    queue.start()
    run_until_done(queue, [parent, child])
    queue.stop()
    assert parent.status == "failed" and parent.error == "boom"
    assert child.status == "failed" and child.error == "dependency failed"


def test_cpu_job_waits_for_dependency():
    queue = JobQueue(EventBus(), cpu_workers=2)
    order: list[str] = []
    queue.register("motion", lambda job: (time.sleep(0.05), order.append(job.id)))
    queue.register("convert", lambda job: order.append(job.id))
    parent = Job(id="p", type="motion", resource="gpu", model_key="A")
    child = Job(id="c", type="convert", resource="cpu", depends_on=["p"])
    queue.submit(child)
    queue.submit(parent)
    queue.start()
    run_until_done(queue, [parent, child])
    queue.stop()
    assert order == ["p", "c"]
