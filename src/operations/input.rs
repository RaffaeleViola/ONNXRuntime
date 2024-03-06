use crate::operations::{Compute, Input, Output};

pub struct InputNode{
}

impl InputNode {
    pub fn new() -> Self {
        InputNode{}
    }
}

impl Compute for InputNode {
    fn compute(&mut self, inputs: Input) -> Output {
        let data  =match inputs {
            Input::TensorD(arr) => arr,
            _ => panic!("Wrong network input")
        };
        Output::TensorD(data)
    }

    fn op_type(&self) -> &'static str {
        return "Input";
    }
}