import datetime
import time
from threading import Thread

from apscheduler.schedulers.background import BackgroundScheduler

from OpenCVRoutine import OpenCVRoutine


class OpenCVRoutineManager(Thread):
    def __init__(self, schedule, frame_queue):
        super(OpenCVRoutineManager, self).__init__()

        self.frame_queue = frame_queue
        self.schedule = schedule
        self.open_cv_routine = None
        self.is_running = True

        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

        self.setDaemon(True)
        self.start()

    def run(self):
        self.open_cv_managing_routine()

    def stop(self):
        self.is_running = False

    def _start_opencv_routine(self):
        self.open_cv_routine = OpenCVRoutine(self.frame_queue)

    def _stop_opencv_routine(self):
        if self.open_cv_routine:
            self.open_cv_routine.stop_routine()

    def _update_datetimes(self):
        format = self.schedule['format']
        start_opencv_routine_from = datetime.datetime.strptime(self.schedule['start_opencv_routine_from'],
                                                               format).time()
        stop_opencv_routine_at = datetime.datetime.strptime(self.schedule['stop_opencv_routine_at'], format).time()

        now_datetime = datetime.datetime.now()
        start_dt = datetime.datetime.combine(now_datetime.date(), start_opencv_routine_from)
        stop_dt = datetime.datetime.combine(now_datetime.date(), stop_opencv_routine_at)
        stop_dt += datetime.timedelta(days=1)

        if now_datetime.time() < stop_opencv_routine_at:
            start_dt += datetime.timedelta(days=-1)
            stop_dt += datetime.timedelta(days=-1)

        return start_dt, stop_dt

    def open_cv_managing_routine(self):
        start_dt, stop_dt = self._update_datetimes()

        while self.is_running:
            now = datetime.datetime.now()
            if start_dt <= now <= stop_dt:
                if self.open_cv_routine is None:
                    self.open_cv_routine = OpenCVRoutine(self.frame_queue)
            elif self.open_cv_routine is not None:
                self.open_cv_routine.stop_routine()
                self.open_cv_routine = None

            if now > stop_dt:
                start_dt, stop_dt = self._update_datetimes()

            time.sleep(1)
