use ndarray::{IxDyn};
use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct Reshape{
}

impl Reshape{

    #![allow(dead_code)]
    pub fn new() -> Reshape{
        return Reshape{
        }
    }


    pub fn parse_from_proto_node() -> Reshape{
        return Reshape{};
    }

}

impl Compute for Reshape{

    fn compute(&mut self, inputs: Input) -> Output {
        let mut list =  match inputs {
            Input::Tensor4List(array) => array,
            _ => panic!("Wrong input reshape")
        };
        let shape = list.pop().unwrap().map(|val| (*val) as usize).into_raw_vec();
        let vec = list.pop().unwrap();
        let reshaped = vec.into_shape(IxDyn(&shape)).unwrap();
        return Output::TensorD(reshaped);
    }

    fn op_type(&self) -> &'static str {
        return "Reshape";
    }
}