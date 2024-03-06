use std::fmt::Debug;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, IxDyn};
use crate::onnx_runtime::onnxruntime::{Error};

pub mod add;
pub mod reshape;
pub mod soft_max;
pub mod dropout;
pub mod gemm;
pub mod concat;
pub mod maxpool;
pub mod start;
pub mod averagepool;
pub mod local_response_normalization;
pub mod relu;
pub mod matmul;
pub mod conv;
pub mod input;

pub trait Compute {
    fn compute(&mut self, inputs: Input) -> Output;
    fn op_type(&self) -> &'static str;
}


impl Debug for dyn Compute {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dyn Compute")
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Input {
    Tensor32(Array4<f32>),
    Tensor1(Array1<f32>),
    Tensor2(Array2<f32>),
    Tensor3(Array3<f32>),
    Tensor4(Array4<f32>),
    TensorD(ArrayD<f32>),
    Tensor32Vec(Vec<Array4<f32>>),
    Tensor4List(Vec<ArrayD<f32>>),
    Empty
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Output {
    Tensor32(Array4<f32>),
    Tensor1(Array1<f32>),
    Tensor2(Array2<f32>),
    Tensor3(Array3<f32>),
    Tensor4(Array4<f32>),
    TensorD(ArrayD<f32>)
}

impl Input {
    pub fn from_raw_vec(vec: Vec<f32>, shape: &[usize]) -> Result<Input, Error>{
        let res = match shape.len() {
            d if d >= 1 && d <= 4 => {
                ArrayD::from_shape_vec(IxDyn(shape), vec)
            },
            _ => return Err(Error::ShapeError)
        };
        return match res {
            Ok(val) => Ok(Input::TensorD(val)),
            Err(_e) => Err(Error::ConversionError)
        }
    }

    #[allow(dead_code)]
    pub fn into_raw_vec(self) -> Result<Vec<f32>, Error> {
        match self {
            Input::Tensor32(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor1(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor2(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor3(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor4(arr) => Ok(arr.into_raw_vec()),
            Input::TensorD(arr) => Ok(arr.into_raw_vec()),
            _ => Err(Error::ConversionError)
        }
    }

    #[allow(dead_code)]
    pub fn list_into_raw_vec(self) -> Result<Vec<Vec<f32>>, Error> {
        match self {
            Input::Tensor4List(vec) =>
                Ok(vec.into_iter().map(|val| val.into_raw_vec()).collect::<Vec<Vec<f32>>>()),
            Input::Tensor32Vec(vec) =>
                Ok(vec.into_iter().map(|val| val.into_raw_vec()).collect::<Vec<Vec<f32>>>()),
            _ => Err(Error::ConversionError)
        }
    }

}

impl Output {

    #[allow(dead_code)]
    pub fn from_raw_vec(vec: Vec<f32>, shape: &[usize]) -> Result<Output, Error>{
        let res = match shape.len() {
            d if d >= 1 && d <= 4 => {
                ArrayD::from_shape_vec(IxDyn(shape), vec)
            },
            _ => return Err(Error::ShapeError)
        };
        return match res {
            Ok(val) => Ok(Output::TensorD(val)),
            Err(_e) => Err(Error::ConversionError)
        }
    }

    pub fn into_raw_vec(self) -> Result<Vec<f32>, Error> {
        match self {
            Output::Tensor32(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor1(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor2(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor3(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor4(arr) => Ok(arr.into_raw_vec()),
            Output::TensorD(arr) => Ok(arr.into_raw_vec()),
            //_ => Err(Error::ConversionError)
        }
    }

}

