import data_preparation
import net
import param
import sys
import json
import socket

class Request(object):
    def __init__(self, d):
       self.__dict__ = d
        
        
class ModelInfo(object):
    def __init__(self,trained,model_type,metric_type):
        self.trained = trained
        self.model_type = model_type
        self.metric_type = metric_type
        
        
class Response(object):
    def __init__(self,model_info,prediction):
        self.model_info =model_info
        self.prediction = prediction
        
        
def check_status(request):
    status = 0
    if hasattr(request,'train_history') and request.train_history is not None and len(request.train_history):
        status |= param.STATUS_TRAIN
    if hasattr(request,'predict_history') and request.predict_history is not None and len(request.predict_history):
        status |= param.STATUS_PREDICT
    return status



def handle(request):
    rsp = Response(ModelInfo(False,None,None),None)

    if check_status(request) & param.STATUS_PREDICT:
        metrics = []
        for index, val in enumerate(request.predict_history):
            metrics.append(val.metric)
        out = net.predict(metrics,request.predict_history[0].type)
        rsp.model_info.trained = True
        rsp.model_info.metric_type = request.predict_history[0].type
        rsp.model_info.model_type = param.model_type
        rsp.prediction = out.tolist()[0]

    if check_status(request) & param.STATUS_TRAIN:
        metrics = []
        for index, val in enumerate(request.train_history):
            metrics.append(val.metric)
        train_loader = data_preparation.train_data_prepare(metrics)
        net.train(train_loader, request.train_history[0].type)
        rsp.model_info.trained = True
        rsp.model_info.metric_type = request.predict_history[0].type
        rsp.model_info.model_type = param.model_type

    return rsp


if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server_socket.bind(param.socket_address)
    request = json.loads(sys.stdin.read(), object_hook=Request)
    rsp = handle(request)

