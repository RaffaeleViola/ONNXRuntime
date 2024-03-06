use ndarray::{ArrayD};
use crate::operations::{Compute, Input, Output};

pub struct Start {
    data: ArrayD<f32>
}

impl Start {
    pub fn new(data: ArrayD<f32>) -> Self {
        Start{data}
    }
}


impl Compute for Start{
    fn compute(&mut self, _inputs: Input) -> Output {
       Output::TensorD(self.data.clone())
    }
    fn op_type(&self) -> &'static str {
        return "Initializer";
    }
}