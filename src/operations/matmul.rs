use ndarray::{Array2, IxDyn};
use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct MatMul{
}

impl MatMul{

    #![allow(dead_code)]
    pub fn new() -> MatMul{
        return MatMul{}
    }

    pub fn parse_from_proto_node() -> MatMul{
        return MatMul{}
    }
}


#[allow(unreachable_code)]
impl Compute for MatMul {
    fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::Tensor4List(input) => {
                let output = input.into_iter()
                    .map(|vec| {
                        let vec_tmp: Array2<f32> = vec.into_dimensionality().unwrap();
                        return vec_tmp;
                    })
                    .reduce(move |v1, v2| (v1.dot(&v2))).unwrap();
                let out_len = Vec::from(output.shape());
                return Output::TensorD(output.into_shape(IxDyn(&out_len)).unwrap());
            },
            _ => panic!("Wrong input")
        }
    }

    fn op_type(&self) -> &'static str {
        return "MatMul";
    }
}

