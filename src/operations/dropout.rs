use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct Dropout{
}

impl Dropout{

    #![allow(dead_code)]
    pub fn new(
    ) -> Dropout{
        return Dropout{
        }

    }


    pub fn parse_from_proto_node() -> Dropout{
        return Dropout{};
    }

}

impl Compute for Dropout{

    fn compute(&mut self, inputs: Input) -> Output {
        let out = match inputs{
            Input::TensorD(array) => array,
            _ => panic!("Wrong input")
        };
        return Output::TensorD(out);
    }

    fn op_type(&self) -> &'static str {
        return "Dropout";
    }
}
