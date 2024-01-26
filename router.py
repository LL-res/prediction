from flask import Flask, jsonify, request,make_response
from DL import  gru

class Router:
    def __init__(self, app):
        self.app = app
        self.setup_routes()
        self.trainig = False

    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index, methods=['GET'])
        self.app.add_url_rule('/train', 'train', self.train, methods=['POST'])
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])

    def index(self):
        return 'GRU backend is running'

    def train(self):
        if self.trainig:
            return make_response(jsonify(message="GRU model is training"), 400)
        request_json = request.json
        try:
            self.trainig = True
            gru_instance = gru.GRU(json=request_json)
            gru_instance.prepare_train_data(request_json.get('metrics'))
            gru_instance.train(request_json.get('key'))
        except Exception as e:
            self.trainig = False
            return make_response(jsonify(message="GRU model failed to start training"), 500)
        finally:
            self.trainig = False
            return make_response(jsonify(message="GRU model finished training"), 200)

    def predict(self):
        request_json = request.json
        try:
            gru_instance = gru.GRU(json=request_json)
            predict_result = gru_instance.predict(request_json.get('metrics'),request_json.get('key'))
        except Exception as e:
            return make_response(jsonify(message="GRU model failed to predict"), 500)
        finally:
            return make_response(jsonify(predictMetric=predict_result), 200)

if __name__ == '__main__':
    app = Flask(__name__)
    router = Router(app)
    app.run()

