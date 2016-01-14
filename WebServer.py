from flask import Flask
from flask_restful import Api


class WebServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.api = Api(self.app)

    def run(self):
        self.app.run(host='0.0.0.0', debug=False, threaded=True, use_reloader=False)

    def map_post(self, resource_to_url):
        for key in resource_to_url:
            self.api.add_resource(key, resource_to_url[key])
        return self
