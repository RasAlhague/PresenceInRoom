from threading import Thread

import requests


class SendPostAsync(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(SendPostAsync, self).__init__(group, target, name, args, kwargs, verbose)
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            print("Sending post to " + self.args[0])
            requests.post(self.args, self.kwargs)
        except Exception as e:
            print e.message
