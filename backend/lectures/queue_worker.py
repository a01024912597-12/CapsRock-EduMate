# lectures/queue_worker.py

import logging
import threading

from django.db import transaction
from django.db.models import Max

from .models import Lecture


logger = logging.getLogger(__name__)


class ReorderableTaskQueue:
    """DB의 queue_order를 기준으로 다음 작업을 선택할 수 있는 단일 워커용 메모리 큐."""

    def __init__(self):
        self._items = []
        self._condition = threading.Condition()
        self._unfinished_tasks = 0

    @staticmethod
    def _lecture_id(task_item):
        if isinstance(task_item, tuple):
            return task_item[0]
        return task_item

    def put(self, task_item):
        with self._condition:
            self._items.append(task_item)
            self._unfinished_tasks += 1
            self._condition.notify()

    def get(self):
        with self._condition:
            while not self._items:
                self._condition.wait()

            lecture_ids = [self._lecture_id(item) for item in self._items]
            lecture_rows = {
                row["id"]: row
                for row in Lecture.objects.filter(id__in=lecture_ids).values(
                    "id",
                    "queue_order",
                    "analysis_requested_at",
                    "created_at",
                    "status",
                )
            }

            def sort_key(task_item):
                lecture_id = self._lecture_id(task_item)
                row = lecture_rows.get(lecture_id)
                if not row:
                    return (2, 10**18, lecture_id)

                # 취소/완료/실패 등 더 이상 처리할 필요가 없는 작업은 뒤로 보낸다.
                if row["status"] != Lecture.STATUS_PENDING:
                    return (1, 10**18, lecture_id)

                order = row["queue_order"] if row["queue_order"] > 0 else 10**18
                requested_at = row["analysis_requested_at"] or row["created_at"]
                return (0, order, requested_at, lecture_id)

            self._items.sort(key=sort_key)
            return self._items.pop(0)

    def task_done(self):
        with self._condition:
            if self._unfinished_tasks > 0:
                self._unfinished_tasks -= 1

    def qsize(self):
        with self._condition:
            return len(self._items)

    def remove_lecture(self, lecture_id):
        removed = 0
        with self._condition:
            kept = []
            for item in self._items:
                if str(self._lecture_id(item)) == str(lecture_id):
                    removed += 1
                else:
                    kept.append(item)

            if removed:
                self._items = kept
                self._unfinished_tasks = max(0, self._unfinished_tasks - removed)

        return removed


# 단일 백그라운드 워커가 처리하는 재정렬 가능 대기열.
task_queue = ReorderableTaskQueue()


class DummyRequest:
    def __init__(self, lecture_id, user, session_data=None):
        self.GET = {"lecture_id": str(lecture_id)}
        self.user = user

        # 기존 analyze_video_request()는 실제 Django 세션처럼
        # lecture_analysis_options_{lecture_id} 키에서 분석 옵션을 읽는다.
        session_key = f"lecture_analysis_options_{lecture_id}"
        self.session = {
            session_key: session_data or {}
        }


def _worker_loop():
    from .services.analysis_service import analyze_video_request

    while True:
        task_item = task_queue.get()
        lecture_id = None

        try:
            if isinstance(task_item, tuple):
                lecture_id, session_data = task_item
            else:
                lecture_id, session_data = task_item, {}

            lecture = Lecture.objects.filter(id=lecture_id).first()
            if not lecture:
                continue

            # 취소·완료·실패·요약 전 상태 등 pending이 아닌 항목은 절대 다시 실행하지 않는다.
            if lecture.status != Lecture.STATUS_PENDING:
                print(
                    f"[Queue Worker] 처리 건너뜀: Lecture ID={lecture.id}, "
                    f"status={lecture.status}"
                )
                continue

            lecture.status = Lecture.STATUS_PROCESSING
            lecture.save(update_fields=["status"])

            print(
                f"[Queue Worker] 순차 분석 시작: "
                f"Lecture ID={lecture.id}, "
                f"queue_order={lecture.queue_order}, "
                f"Title={lecture.title}"
            )

            print(
                f"[Queue Worker 옵션] "
                f"Lecture ID={lecture.id}, "
                f"whisper_model={session_data.get('whisper_model')}, "
                f"workers={session_data.get('workers')}, "
                f"subject_code={session_data.get('subject_code')}, "
                f"summary_api={session_data.get('summary_api')}"
            )

            req = DummyRequest(
                lecture.id,
                lecture.user,
                session_data,
            )

            analyze_video_request(req)

            lecture.refresh_from_db()

            # 분석 서비스가 완료 상태를 기록하지 않았더라도 실제 요약이 생성됐다면 완료 처리한다.
            if lecture.status == Lecture.STATUS_PROCESSING:
                if (lecture.summary_text or "").strip():
                    lecture.status = Lecture.STATUS_COMPLETED
                    lecture.save(update_fields=["status"])
                else:
                    lecture.status = Lecture.STATUS_FAILED
                    if not lecture.error_message:
                        lecture.error_message = "분석이 완료되지 않았습니다."
                    lecture.save(update_fields=["status", "error_message"])

            print(
                f"[Queue Worker] 분석 종료: "
                f"Lecture ID={lecture.id}, status={lecture.status}"
            )

        except Exception as exc:
            print(
                f"[Queue Worker 오류] "
                f"Lecture ID={lecture_id}: {exc}"
            )

            if lecture_id is not None:
                Lecture.objects.filter(id=lecture_id).exclude(
                    status=Lecture.STATUS_CANCELED
                ).update(
                    status=Lecture.STATUS_FAILED,
                    error_message=str(exc),
                )

        finally:
            task_queue.task_done()


worker_thread = threading.Thread(
    target=_worker_loop,
    daemon=True,
)
worker_thread.start()


def _assign_queue_order_if_needed(lecture_id):
    """새 pending 작업에 전역 queue_order가 없으면 대기열 맨 뒤 순번을 부여한다."""
    with transaction.atomic():
        lecture = Lecture.objects.select_for_update().get(id=lecture_id)

        if lecture.queue_order > 0:
            return lecture.queue_order

        max_order = Lecture.objects.aggregate(max_order=Max("queue_order"))["max_order"] or 0
        lecture.queue_order = max_order + 1
        lecture.save(update_fields=["queue_order"])
        return lecture.queue_order


def add_lecture_to_queue(lecture_id, session_data=None):
    """새 강의를 대기열 맨 뒤에 적재한다."""
    queue_order = _assign_queue_order_if_needed(lecture_id)

    task_queue.put(
        (
            lecture_id,
            session_data or {},
        )
    )

    print(
        f"[Queue Enqueue] 대기열 적재 완료: "
        f"Lecture ID={lecture_id}, "
        f"queue_order={queue_order} "
        f"(현재 대기열 크기: {task_queue.qsize()})"
    )


def remove_lecture_from_queue(lecture_id):
    """아직 워커가 가져가지 않은 대기 작업을 메모리 큐에서 제거한다."""
    removed = task_queue.remove_lecture(lecture_id)
    print(
        f"[Queue Remove] Lecture ID={lecture_id}, removed={removed}, "
        f"남은 대기열 크기={task_queue.qsize()}"
    )
    return removed
