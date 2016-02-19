from threading import Thread

from apscheduler.schedulers.background import BackgroundScheduler

from OpenCVRoutine import OpenCVRoutine


class OpenCVRoutineManager(Thread):
    def __init__(self, schedule, frame_queue):
        super(OpenCVRoutineManager, self).__init__()

        self.frame_queue = frame_queue
        self.schedule = schedule
        self.open_cv_routine = None

        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

        self.setDaemon(True)
        self.start()

    def run(self):
        self.open_cv_managing_routine()

    def _start_opencv_routine(self):
        self.open_cv_routine = OpenCVRoutine(self.frame_queue)

    def _stop_opencv_routine(self):
        if self.open_cv_routine:
            self.open_cv_routine.stop_routine()

    def open_cv_managing_routine(self):
        self._start_opencv_routine()
        self.scheduler.add_job(func=self._start_opencv_routine, trigger="cron",
                               year='*', month='*', day='*', week='*', day_of_week='1/1', hour=17, minute=0, second=0)
        self.scheduler.add_job(func=self._stop_opencv_routine, trigger="cron",
                               year='*', month='*', day='*', week='*', day_of_week='1/1', hour=7, minute=0, second=0)
