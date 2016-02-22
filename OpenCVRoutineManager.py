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

    def open_cv_managing_routine(self):
        format = self.schedule['format']
        start_opencv_routine_from = datetime.datetime.strptime(self.schedule['start_opencv_routine_from'],
                                                               format).time()
        stop_opencv_routine_at = datetime.datetime.strptime(self.schedule['stop_opencv_routine_at'], format).time()
        while self.is_running:
            now_date = datetime.datetime.now().date()
            start_dt = datetime.datetime.combine(now_date, start_opencv_routine_from)
            stop_dt = datetime.datetime.combine(now_date, stop_opencv_routine_at)
            stop_dt += datetime.timedelta(days=1)

            if start_dt <= datetime.datetime.now() <= stop_dt:
                if self.open_cv_routine is None:
                    self.open_cv_routine = OpenCVRoutine(self.frame_queue)
            else:
                self.open_cv_routine.stop_routine()
                self.open_cv_routine = None

            time.sleep(1)
