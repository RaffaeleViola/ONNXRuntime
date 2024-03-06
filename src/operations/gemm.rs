use ndarray::{Array2, IxDyn};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto};

#[derive(Clone, Debug)]
pub struct Gemm{
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: i32,
    pub trans_b: i32,
}

impl Gemm{

    #![allow(dead_code)]
    pub fn new(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i32>,
        trans_b: Option<i32>,
    ) -> Gemm{
        return Gemm{
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(1.0),
            trans_a: trans_a.unwrap_or(0),
            trans_b: trans_b.unwrap_or(0)
        }

    }


    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Gemm{
        let mut alpha = 1.0;
        let mut beta = 1.0;
        let mut trans_a = 0;
        let mut trans_b = 0;
        for attr in attributes.iter(){
            match attr.name.as_str(){
                "alpha" => {
                    alpha = attr.f;

                },
                "beta" => {
                    beta = attr.f;
                },
                "transA" => {
                    trans_a = attr.i as i32;
                },
                "transB" => {
                    trans_b = attr.i as i32;
                },
                _ => ()
            }

        }
        return Gemm{alpha, beta, trans_a, trans_b};
    }

}

impl Compute for Gemm{

    fn compute(&mut self, inputs: Input) -> Output {
        let mut arrays = match inputs {
            Input::Tensor4List(vec_array) => vec_array,
            _ => panic!("Input is not a vector")
        };
        let c = arrays.pop().unwrap();
        let mut b: Array2<f32> = arrays.pop().unwrap().into_dimensionality().unwrap();
        let mut a: Array2<f32>  = arrays.pop().unwrap().into_dimensionality().unwrap();
        if self.trans_a != 0 {
            a = a.reversed_axes();
        }
        if self.trans_b != 0 {
            b = b.reversed_axes();
        }
        let result  = self.alpha * a.dot(&b) + self.beta * c;
        let out_len  = Vec::from(result.shape());
        return Output::TensorD(result.into_shape(IxDyn(&out_len)).unwrap());
    }

    fn op_type(&self) -> &'static str {
        return "Gemm";
    }
}
