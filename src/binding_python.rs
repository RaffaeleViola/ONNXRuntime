use pyo3::prelude::*;
use ndarray::{Array1, Dim, Shape};
use crate::operations::{Input, Output};
use crate::onnx_runtime::onnxruntime::{parse_input_tensor};
use std::path::Path;
use numpy::{PyArray, PyReadonlyArrayDyn};
use crate::onnx_runtime;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use pyo3::types::{PyList};
use crate::onnx_runtime::onnxruntime::Error as OnnxRuntimeError;
use crate::operations::add::Add;
use crate::operations::averagepool::AveragePool;
use crate::operations::concat::Concat;
use crate::operations::conv::Conv;
use crate::operations::dropout::Dropout;
use crate::operations::gemm::Gemm;
use crate::operations::local_response_normalization::LRN;
use crate::operations::matmul::MatMul;
use crate::operations::maxpool::MaxPool;
use crate::operations::relu::Relu;
use crate::operations::reshape::Reshape;
use crate::operations::soft_max::SoftMax;

impl From<OnnxRuntimeError> for PyErr {
    fn from(err: OnnxRuntimeError) -> Self {
        match err {
            OnnxRuntimeError::ProtoBufError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Failed to parse the model (protobuf error)")
            },
            OnnxRuntimeError::InputParsingError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Failed to parse input tensor")
            },
            OnnxRuntimeError::ShapeError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Invalid tensor shape")
            },
            OnnxRuntimeError::ConversionError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Data conversion error")
            },
        }
    }
}

enum PyOperation {
    PyAdd(PyAdd),
    PyAveragePool(PyAveragePool),
    PyConcat(PyConcat),
    PyConv(PyConv),
    PyDroput(PyDropout),
    PyGemm(PyGemm),
    PyLRN(PyLRN),
    PyMatMul(PyMatMul),
    PyMaxPool(PyMaxPool),
    PyRelu(PyRelu),
    PyReshape(PyReshape),
    PySoftMax(PySoftMax)
}


#[pyclass]
#[derive(Clone)]
struct PyAdd {
    pub add: Add,
}

#[pymethods]
impl PyAdd {
    #[new]
    fn new() -> PyResult<Self> {
        let add = Add::new();
        Ok(PyAdd {add})
    }
}

#[pyclass]
#[derive(Clone)]
struct PyAveragePool {
    pub average_pool: AveragePool,
}

#[pymethods]
impl PyAveragePool {
    #[new]
    fn new(kernel_shape: Option<(i32, i32)>, pads: Option<Vec<i32>>, strides: Option<Vec<i32>>) -> PyResult<Self> {
        let kernel_shape_rust = kernel_shape
            .map(|(k1, k2)| Shape::from(Dim([k1 as usize, k2 as usize])))
            .unwrap_or(Shape::from(Dim([1, 1])));

        let pads_rust = pads
            .map(|p| Array1::from(p))
            .unwrap_or(Array1::from_vec(vec![1, 1, 1, 1]));

        let strides_rust = strides
            .map(|s| Array1::from(s))
            .unwrap_or(Array1::from_vec(vec![1, 1]));

        let averagepool = AveragePool::new(Option::from(kernel_shape_rust), Option::from(pads_rust), Option::from(strides_rust));

        Ok(PyAveragePool { average_pool: averagepool })
    }
}

#[pyclass]
#[derive(Clone)]
struct PyConcat {
    pub concat: Concat,
}

#[pymethods]
impl PyConcat {
    #[new]
    fn new() -> PyResult<Self> {
        let concat = Concat::new();
        Ok(PyConcat {concat})
    }
}

#[pyclass]
#[derive(Clone)]
struct PyConv {
    pub conv: Conv,
}

#[pymethods]
impl PyConv {
    #[new]
    fn new(ap: Option<String>,
           dil: Option<&PyList>,
           group: Option<u32>,
           kernel_shape: Option<(i32, i32)>,
           pads: Option<&PyList>,
           strides: Option<&PyList>) -> PyResult<Self> {

        let autopad_rust = ap.unwrap_or("NOT_SET".to_string());

        let dilations_rust = dil
            .map(|d| Array1::from_iter(d.iter().map(|x| x.extract::<i32>().unwrap())))
            .unwrap_or(Array1::from_vec(vec![1, 1]));

        let group_rust = group.unwrap_or(1);

        let kernel_shape_rust = kernel_shape
            .map(|(k1, k2)| Shape::from(Dim([k1 as usize, k2 as usize])))
            .unwrap_or(Shape::from(Dim([1, 1])));

        let pads_rust = pads
            .map(|p| Array1::from_iter(p.iter().map(|x| x.extract::<i32>().unwrap())))
            .unwrap_or(Array1::from_vec(vec![0, 0, 0, 0]));

        let strides_rust = strides
            .map(|s| Array1::from_iter(s.iter().map(|x| x.extract::<i32>().unwrap())))
            .unwrap_or(Array1::from_vec(vec![1, 1]));

        let conv = Conv::new(
            Option::from(autopad_rust),
            Option::from(dilations_rust),
            Option::from(group_rust),
            Option::from(kernel_shape_rust),
            Option::from(pads_rust),
            Option::from(strides_rust)
        );

        Ok(PyConv { conv })
    }
}

#[pyclass]
#[derive(Clone)]
struct PyDropout {
    pub dropout: Dropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    fn new() -> PyResult<Self> {
        let dropout = Dropout::new();
        Ok(PyDropout {dropout})
    }
}

#[pyclass]
#[derive(Clone)]
struct PyGemm {
    pub gemm: Gemm,
}

#[pymethods]
impl PyGemm {
    #[new]
    fn new(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i32>,
        trans_b: Option<i32>,
    ) -> PyResult<Self> {
        let alpha_rust = alpha.unwrap_or(1.0);
        let beta_rust = beta.unwrap_or(1.0);
        let trans_a_rust = trans_a.unwrap_or(0);
        let trans_b_rust = trans_b.unwrap_or(0);

        let gemm = Gemm::new(Option::from(alpha_rust), Option::from(beta_rust), Option::from(trans_a_rust), Option::from(trans_b_rust));

        Ok(PyGemm{gemm})

    }
}

#[pyclass]
#[derive(Clone)]
struct PyLRN {
    pub lrn: LRN,
}

#[pymethods]
impl PyLRN {
    #[new]
    fn new(
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<f32>,
        size: Option<i64>,
    ) -> PyResult<Self> {
        let alpha_rust = alpha.unwrap();
        let beta_rust = beta.unwrap();
        let bias_rust = bias.unwrap();
        let size_rust = size.unwrap();

        let lrn = LRN::new(alpha_rust, beta_rust,bias_rust, size_rust);

        Ok(PyLRN{lrn})

    }
}

#[pyclass]
#[derive(Clone)]
struct PyMatMul {
    pub matmul: MatMul,
}

#[pymethods]
impl PyMatMul {
    #[new]
    fn new() -> PyResult<Self> {
        let matmul = MatMul::new();
        Ok(PyMatMul {matmul})
    }
}

#[pyclass]
#[derive(Clone)]
struct PyMaxPool {
    pub max_pool: MaxPool,
}

#[pymethods]
impl PyMaxPool {
    #[new]
    fn new(kernel_shape: Option<(i32, i32)>, pads: Option<Vec<i32>>, strides: Option<Vec<i32>>) -> PyResult<Self> {
        let kernel_shape_rust = kernel_shape
            .map(|(k1, k2)| Shape::from(Dim([k1 as usize, k2 as usize])))
            .unwrap_or(Shape::from(Dim([1, 1])));

        let pads_rust = pads
            .map(|p| Array1::from(p))
            .unwrap_or(Array1::from_vec(vec![1, 1, 1, 1]));

        let strides_rust = strides
            .map(|s| Array1::from(s))
            .unwrap_or(Array1::from_vec(vec![1, 1]));

        let maxpool = MaxPool::new(Option::from(kernel_shape_rust), Option::from(pads_rust), Option::from(strides_rust));

        Ok(PyMaxPool { max_pool: maxpool })
    }
}

#[pyclass]
#[derive(Clone)]
struct PyRelu {
    pub relu: Relu,
}

#[pymethods]
impl PyRelu {
    #[new]
    fn new() -> PyResult<Self> {
        let relu = Relu::new();
        Ok(PyRelu {relu})
    }
}

#[pyclass]
#[derive(Clone)]
struct PyReshape {
    pub reshape: Reshape,
}

#[pymethods]
impl PyReshape {
    #[new]
    fn new() -> PyResult<Self> {
        let reshape = Reshape::new();
        Ok(PyReshape {reshape})
    }
}

#[pyclass]
#[derive(Clone)]
struct PySoftMax {
    pub soft_max: SoftMax,
}

#[pymethods]
impl PySoftMax {
    #[new]
    fn new() -> PyResult<Self> {
        let soft_max = SoftMax::new();
        Ok(PySoftMax {soft_max})
    }
}

#[pyclass]
struct PyDepGraph {
    pub dep_graph: crate::graph::DepGraph,
}

#[pymethods]
impl PyDepGraph {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model_path = Path::new(model_path);
        let model_path_str = model_path.to_string_lossy().to_string();
        let dep_graph = onnx_runtime::onnxruntime::get_computational_graph(model_path_str);
        Ok(PyDepGraph { dep_graph })
    }

    fn py_run(&mut self, py_input: PyInput) -> PyResult<Option<PyOutput>> {
        match self.dep_graph.run(py_input.input) {
            Some(output) => {
                let py_output = PyOutput{ output };
                    Ok(Some(py_output))
            },
            None => Ok(None),
        }
    }

    fn py_add_node(&mut self, name: String, operation: &PyAny, deps: &PyList) -> PyResult<()> {

        let deps_vec: Vec<String> = deps.iter().map(|x| x.extract().unwrap()).collect();
        let operation_enum = if let Ok(py_add) = operation.extract::<PyAdd>() {
            PyOperation::PyAdd(py_add)
        } else if let Ok(py_avgpool) = operation.extract::<PyAveragePool>() {
            PyOperation::PyAveragePool(py_avgpool)
        } else if let Ok(py_concat) = operation.extract::<PyConcat>() {
            PyOperation::PyConcat(py_concat)
        } else if let Ok(py_conv) = operation.extract::<PyConv>() {
            PyOperation::PyConv(py_conv)
        } else if let Ok(py_dropout) = operation.extract::<PyDropout>() {
            PyOperation::PyDroput(py_dropout)
        } else if let Ok(py_gemm) = operation.extract::<PyGemm>() {
            PyOperation::PyGemm(py_gemm)
        } else if let Ok(py_lrn) = operation.extract::<PyLRN>() {
            PyOperation::PyLRN(py_lrn)
        } else if let Ok(py_matmul) = operation.extract::<PyMatMul>() {
            PyOperation::PyMatMul(py_matmul)
        } else if let Ok(py_maxpool) = operation.extract::<PyMaxPool>() {
            PyOperation::PyMaxPool(py_maxpool)
        } else if let Ok(py_relu) = operation.extract::<PyRelu>() {
            PyOperation::PyRelu(py_relu)
        } else if let Ok(py_reshape) = operation.extract::<PyReshape>() {
            PyOperation::PyReshape(py_reshape)
        } else if let Ok(py_softmax) = operation.extract::<PySoftMax>() {
            PyOperation::PySoftMax(py_softmax)
        } else {
            return Err(PyErr::new::<PyValueError, _>("Unsupported operation type"));
        };

        match operation_enum {
            PyOperation::PyAdd(py_add) => {
                self.dep_graph.add_node(name, Box::new(py_add.add.clone()), &deps_vec);
            },
            PyOperation::PyAveragePool(py_averagepool) => {
                self.dep_graph.add_node(name, Box::new(py_averagepool.average_pool.clone()), &deps_vec);
            },
            PyOperation::PyConcat(py_concat) => {
                self.dep_graph.add_node(name, Box::new(py_concat.concat.clone()), &deps_vec);
            },
            PyOperation::PyConv(py_conv) => {
                self.dep_graph.add_node(name, Box::new(py_conv.conv.clone()), &deps_vec);
            },
            PyOperation::PyDroput(py_dropout) => {
                self.dep_graph.add_node(name, Box::new(py_dropout.dropout.clone()), &deps_vec);
            },
            PyOperation::PyGemm(py_gemm) => {
                self.dep_graph.add_node(name, Box::new(py_gemm.gemm.clone()), &deps_vec);
            },
            PyOperation::PyLRN(py_lrn) => {
                self.dep_graph.add_node(name, Box::new(py_lrn.lrn.clone()), &deps_vec);
            },
            PyOperation::PyMatMul(py_matmul) => {
                self.dep_graph.add_node(name, Box::new(py_matmul.matmul.clone()), &deps_vec);
            },
            PyOperation::PyMaxPool(py_maxpool) => {
                self.dep_graph.add_node(name, Box::new(py_maxpool.max_pool.clone()), &deps_vec);
            },
            PyOperation::PyRelu(py_relu) => {
                self.dep_graph.add_node(name, Box::new(py_relu.relu.clone()), &deps_vec);
            },
            PyOperation::PyReshape(py_reshape) => {
                self.dep_graph.add_node(name, Box::new(py_reshape.reshape.clone()), &deps_vec);
            },
            PyOperation::PySoftMax(py_softmax) => {
                self.dep_graph.add_node(name, Box::new(py_softmax.soft_max.clone()), &deps_vec);
            }
        }
        Ok(())
    }

    fn py_remove_node(&mut self, name: String) -> PyResult<()> {
        match self.dep_graph.remove_node(name) {
            Ok(()) => Ok(()),
            Err(_err) => Err(PyValueError::new_err("Node not found")),
        }
    }

    fn py_modify_node_dep(&mut self, name: String, to_remove: Option<String>, to_add: Option<String>) -> PyResult<()> {
        match self.dep_graph.modify_node_dep(name, to_remove, to_add) {
            Ok(()) => Ok(()),
            Err(_err) => Err(PyValueError::new_err("Node not found")),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyInput {
    pub input: Input,
}

#[pymethods]
impl PyInput {
    pub fn as_vec(&self) -> PyResult<Vec<f32>> {
        match &self.input {
            Input::Tensor32(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor1(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor2(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor3(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor4(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::TensorD(tensor) => Ok(tensor.clone().into_raw_vec()),
            _ => Err(PyErr::new::<PyValueError, _>("Unsupported input type")),
        }
    }

    pub fn as_list(&self) -> PyResult<Vec<Vec<f32>>> {
        match &self.input {
            Input::Tensor4List(list) => {
                Ok(Input::Tensor4List(list.clone()).list_into_raw_vec()?)
            },
            Input::Tensor32Vec(list) => {
                Ok(Input::Tensor32Vec(list.clone()).list_into_raw_vec()?)
            },
            _ => Err(PyErr::new::<PyValueError, _>("Unsupported input type")),
        }
    }

    #[staticmethod]
    pub fn from_numpy(_py: Python, array: &PyAny) -> PyResult<Self> {
        if let Ok(numpy_array) = array.extract::<PyReadonlyArrayDyn<f32>>() {
            let input = Input::TensorD(numpy_array.to_owned_array());
            return Ok(PyInput { input });
        }

        if let Ok(py_list) = array.extract::<&PyList>() {
            let mut tensor32vec = Vec::new();
            for item in py_list.iter() {
                let arr: &PyArray<f32, _> = item.extract::<&PyArray<f32, _>>()?;
                let rust_array = arr.to_owned_array();
                tensor32vec.push(rust_array);
            }

            return Ok(PyInput { input: Input::Tensor32Vec(tensor32vec) });
        }

        Err(PyErr::new::<PyRuntimeError, _>("Input must be a numpy array or a list of numpy arrays"))
    }
}

#[pyfunction]
fn py_parse_input_tensor(path: String) -> PyResult<PyInput> {
    match parse_input_tensor(path) {
        Ok(input) => {
            Ok(PyInput { input })
        },
        Err(_err) => Err(PyValueError::new_err("Input parsing error")),
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyOutput {
    pub output: Output,
}

#[pymethods]
impl PyOutput {
    pub fn as_vec(&self) -> PyResult<Vec<f32>> {
        match &self.output {
            Output::Tensor32(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor1(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor2(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor3(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor4(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::TensorD(tensor) => Ok(tensor.clone().into_raw_vec()),
            //_ => Err(PyErr::new::<PyValueError, _>("Unsupported input type")),
        }
    }
}

#[pymodule]
fn onnx_rust2py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAdd>()?;
    m.add_class::<PyAveragePool>()?;
    m.add_class::<PyConcat>()?;
    m.add_class::<PyConv>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyGemm>()?;
    m.add_class::<PyLRN>()?;
    m.add_class::<PyMatMul>()?;
    m.add_class::<PyMaxPool>()?;
    m.add_class::<PyRelu>()?;
    m.add_class::<PyReshape>()?;
    m.add_class::<PySoftMax>()?;
    m.add_class::<PyDepGraph>()?;
    m.add_class::<PyInput>()?;
    m.add_class::<PyOutput>()?;
    m.add_function(wrap_pyfunction!(py_parse_input_tensor, m)?)?;
    Ok(())
}