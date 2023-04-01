import os
import signal
import sys
import json
import socket

import data_preparation
import net
import param


class Request(object):
    def __init__(self, d):
       self.__dict__ = d
        
        
class ModelInfo(object):
    def __init__(self,trained,model_type,metric_type):
        self.trained = trained
        self.model_type = model_type
        self.metric_type = metric_type
        
        
class Response(object):
    def __init__(self,trained,model_type,metric_type,prediction):
        self.trained = trained
        self.model_type = model_type
        self.metric_type = metric_type
        self.prediction = prediction
        
        
def check_status(request):
    status = 0
    if hasattr(request,'train_history') and request.train_history is not None and len(request.train_history):
        status |= param.STATUS_TRAIN
    if hasattr(request,'predict_history') and request.predict_history is not None and len(request.predict_history):
        status |= param.STATUS_PREDICT
    return status

def handle(request):
    rsp = Response(False,None,None,None)

    if check_status(request) & param.STATUS_PREDICT:
        metrics = []
        for index, val in enumerate(request.predict_history):
            metrics.append(val.metric)
        out = net.predict(metrics,request.predict_history[0].type)
        rsp.trained = True
        rsp.metric_type = request.predict_history[0].type
        rsp.model_type = param.model_type
        rsp.prediction = out.tolist()[0]

    if check_status(request) & param.STATUS_TRAIN:
        metrics = []
        for index, val in enumerate(request.train_history):
            metrics.append(val.metric)
        train_loader = data_preparation.train_data_prepare(metrics)
        net.train(train_loader, request.train_history[0].type)
        rsp.trained = True
        rsp.metric_type = request.predict_history[0].type
        rsp.model_type = param.model_type

    return rsp



class SocketServer:
    def __init__(self):
        # unix domain sockets
        socket_family = socket.AF_UNIX
        socket_type = socket.SOCK_STREAM

        self.sock = socket.socket(socket_family, socket_type)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(param.socket_address)
        self.sock.listen(1)
        print(f"listening on '{param.socket_address}'.")

        # register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def wait_and_deal_client_connect(self):
        while True:
            connection, client_address = self.sock.accept()
            data = b''
            while True:
                chunk = connection.recv(1024)
                if not chunk:
                    break
                data += chunk
                if data.endswith(b"\n"):
                    break
            print(f"recv data from client '{client_address}': {data.decode()}")
            resp = handle(Request(data.decode().removesuffix("\n")))
            connection.sendall(json.dumps(resp.__dict__).encode())
            connection.shutdown(socket.SHUT_WR)

    def _signal_handler(self, signum, frame):
        print(f"\nreceived signal {signum}, exiting...")
        self.__del__()
        sys.exit(0)

    def __del__(self):

        self.sock.close()
        os.system('rm -rf {}'.format(param.socket_address))

if __name__ == "__main__":
    socket_server_obj = SocketServer()
    socket_server_obj.wait_and_deal_client_connect()
