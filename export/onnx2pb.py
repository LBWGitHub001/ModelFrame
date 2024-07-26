import onnx
from onnx_tf.backend import prepare
import os

def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model


if __name__ == "__main__":
    onnx_input_path = './export/LeNet.onnx'
    pb_output_path = './export/LeNet.pb'

    onnx2pb(onnx_input_path, pb_output_path)
