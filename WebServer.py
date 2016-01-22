from flask import Flask

from OpenCVRoutine import set_rm


def run():
    app = Flask(__name__)

    @app.route('/mode/set/<int:rm>', methods=['POST'])
    def post(rm):
        set_rm(rm)
        return {'record_mode': rm}

    app.run(host='0.0.0.0')
