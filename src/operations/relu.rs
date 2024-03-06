use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct Relu{
}

impl Relu{

    #![allow(dead_code)]
    pub fn new() -> Relu{
        return Relu{}
    }



    pub fn parse_from_proto_node() -> Relu{
        return Relu{}
    }
}

impl Compute for Relu {
    fn compute(&mut self, inputs: Input) -> Output {
        match inputs {
            Input::Tensor1(mut array) => {
                array.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                Output::Tensor1(array)
            },
            Input::Tensor2(mut array) => {
                array.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                Output::Tensor2(array)
            },
            Input::Tensor3(mut array) => {
                array.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                Output::Tensor3(array)
            },
            Input::Tensor32(mut array) => {
                array.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                Output::Tensor32(array)
            },
            Input::TensorD(mut array) => {
                array.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                Output::TensorD(array)
            },
            _ => panic!("Wrong input"),
        }
    }

    fn op_type(&self) -> &'static str {
        return "Relu";
    }
}