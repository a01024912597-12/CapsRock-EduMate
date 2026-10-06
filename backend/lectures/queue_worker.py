# lectures/queue_worker.py
import queue
import threading
import logging
from .models import Lecture

logger = logging.getLogger(__name__)

# 선입선출(FIFO) 큐
task_queue = queue.Queue()


class DummyRequest:
    def __init__(self, lecture_id, user, session_data=None):
        self.GET = {'lecture_id': str(lecture_id)}
        self.user = user
        # 세션 옵션 전달 (빈 딕셔너리가 아니라 실제 저장된 옵션 반영)
        self.session = session_data or {}


def _worker_loop():
    from .services.analysis_service import analyze_video_request

    while True:
        task_item = task_queue.get()
        try:
            # 튜플로 들어왔을 경우와 id만 들어왔을 경우 모두 호환
            if isinstance(task_item, tuple):
                lecture_id, session_data = task_item
            else:
                lecture_id, session_data = task_item, {}

            lecture = Lecture.objects.filter(id=lecture_id).first()
            if not lecture:
                continue

            # 이미 요약이 완료된 강의면 건너뜀
            if lecture.summary_text and lecture.status == Lecture.STATUS_COMPLETED:
                continue

            # 1. 상태를 'processing'으로 변경
            lecture.status = Lecture.STATUS_PROCESSING
            lecture.save(update_fields=['status'])
            print(f"[Queue Worker] 1개씩 순차 분석 시작 (FIFO): Lecture ID={lecture.id}, Title={lecture.title}")

            # 2. 분석 함수 실행 (옵션 세션 포함)
            req = DummyRequest(lecture.id, lecture.user, session_data)
            analyze_video_request(req)

            # 3. DB 최신화 및 완료 검증
            lecture.refresh_from_db()
            if lecture.status != Lecture.STATUS_COMPLETED:
                lecture.status = Lecture.STATUS_COMPLETED
                lecture.save(update_fields=['status'])
            print(f"[Queue Worker] 분석 완료: Lecture ID={lecture.id}")

        except Exception as e:
            print(f"[Queue Worker 오류] Lecture ID={lecture_id}: {e}")
            Lecture.objects.filter(id=lecture_id).update(
                status=Lecture.STATUS_FAILED,
                error_message=str(e)
            )
        finally:
            task_queue.task_done()


# 백그라운드 단일 워커 스레드 가동 (하나씩 순차 처리)
worker_thread = threading.Thread(target=_worker_loop, daemon=True)
worker_thread.start()


def add_lecture_to_queue(lecture_id, session_data=None):
    """새 강의를 대기열 맨 뒤에 적재 (선입선출)"""
    task_queue.put((lecture_id, session_data or {}))
    print(f"[Queue Enqueue] 대기열 적재 완료: Lecture ID={lecture_id} (현재 대기열 크기: {task_queue.qsize()})")