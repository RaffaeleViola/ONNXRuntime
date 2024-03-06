import numpy as np
import onnx_rust2py as onnx


print("Running MNIST-7")
dep_graph = onnx.PyDepGraph("../src/mnist-7/model.onnx")
input_arr = onnx.py_parse_input_tensor("../src/mnist-7/test_data_set_0/input_0.pb")
out = dep_graph.py_run(input_arr)
result = onnx.py_parse_input_tensor("../src/mnist-7/test_data_set_0/output_0.pb")
print(f"Output: {out.as_vec()}")
print(f"Result: {result.as_vec()}")

print("Running Googlenet")
dep_graph = onnx.PyDepGraph("../src/googlenet/model.onnx")
input_arr = onnx.py_parse_input_tensor("../src/googlenet/test_data_set_0/input_0.pb")
out = dep_graph.py_run(input_arr)
result = onnx.py_parse_input_tensor("../src/googlenet/test_data_set_0/output_0.pb")
print(f"Output: {out.as_vec()}")
print(f"Result: {result.as_vec()}")

print("Running MNIST-7 with custom input")
dep_graph = onnx.PyDepGraph("../src/mnist-7/model.onnx")
input_arr = onnx.PyInput.from_numpy(np.full((1, 1, 28, 28), 0.7, dtype=np.float32))
out = dep_graph.py_run(input_arr)
result = onnx.py_parse_input_tensor("../src/mnist-7/test_data_set_0/output_0.pb")
print(f"Output: {out.as_vec()}")

print("Adding a new node")
dep_graph = onnx.PyDepGraph("../src/mnist-7/model.onnx")
operation = onnx.PySoftMax()
dep_graph.py_add_node("Softmax_213", operation, ["Plus214"])
input_arr = onnx.py_parse_input_tensor("../src/mnist-7/test_data_set_0/input_0.pb")
out = dep_graph.py_run(input_arr)
print(f"Output: {out.as_vec()}")

print("Removing the node previously created")
dep_graph.py_remove_node("Softmax_213")
out = dep_graph.py_run(input_arr)
print(f"Output: {out.as_vec()}")

print("Complex modifications of the network")
dep_graph.py_add_node("Softmax_213", operation, ["Times212"])
dep_graph.py_modify_node_dep("Plus214", "Times212", "Softmax_213")
out = dep_graph.py_run(input_arr)
print(f"Output: {out.as_vec()}")

print("Random input on the same network")
input_arr = onnx.PyInput.from_numpy(np.full((1, 1, 28, 28), 0.7, dtype=np.float32))
out = dep_graph.py_run(input_arr)
print(f"Output: {out.as_vec()}")


