use ndarray::{IxDyn};
use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct Add{
}

impl Add{

    #[allow(dead_code)]
    pub fn new() -> Add{
        return Add{ }

    }



    pub fn parse_from_proto_node() -> Add{
        return Add{}
    }

}


#[allow(unreachable_code)]
impl Compute for Add{
    fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::Tensor4List(input) => {
                let output = input.into_iter()
                    .reduce(move |v1, v2| (v1 + v2)).unwrap();
                let out_len = Vec::from(output.shape());
                return Output::TensorD(output.into_shape(IxDyn(&out_len)).unwrap());
            },
            _ => panic!("Wrong input")
        }
    }

    fn op_type(&self) -> &'static str {
        return "Add";
    }
}
