cargo build --release
cd target\release
ren onnx_rust2py.dll onnx_rust2py.pyd
move onnx_rust2py.pyd ..\..\python_binding