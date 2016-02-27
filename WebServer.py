from flask import Flask

from OpenCVRoutine import OpenCVRoutine

record_mode = 0


def run():
    app = Flask(__name__)

    @app.route('/recordmode/<int:rm>', methods=['POST'])
    def post(rm):
        global record_mode

        OpenCVRoutine.record_mode = rm
        record_mode = rm

        return {'record_mode': rm}

    app.run(host='0.0.0.0')
