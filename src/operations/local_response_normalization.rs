use std::cmp::{max, min};
use ndarray::{Array4, IxDyn, s};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto};

#[derive(Clone, Debug)]
pub struct LRN{
    pub alpha: f32,
    pub beta: f32,
    pub bias: f32,
    pub size: i64
}

impl LRN {

    #![allow(dead_code)]
    pub fn new(alpha: f32, beta: f32, bias: f32, size: i64) -> LRN {
        LRN{alpha, beta, bias, size}
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> LRN {
        let mut alpha = 0.0;
        let mut beta = 0.0;
        let mut bias = 0.0;
        let mut size = 0;
        for attr in attributes.iter(){
            match attr.name.as_str(){
                "alpha" => {
                    alpha = attr.f;

                },
                "beta" => {
                    beta = attr.f;
                },
                "bias" => {
                    bias = attr.f;
                },
                "size" => {
                    size = attr.i;
                },
                _ => ()
            }

        }
        return LRN{alpha, beta, bias, size}
    }
}

impl Compute for LRN {
    fn compute(&mut self, input: Input) -> Output {

        let tensor: Array4<f32> = match input {
            Input::TensorD(array) => array.into_dimensionality().unwrap(),
            _ => panic!("wrong input type")
        };
        let input_len = tensor.shape();
        let mut square_sum = Array4::zeros(tensor.raw_dim());
        let (b, c, h, w) = (input_len[0], input_len[1], input_len[2], input_len[3]);
        let limit = (c) as i32;
        //compute square_sum
        for batch in 0..b{
            for channel in 0..c{
                let tmp = (self.size as f32 - 1.0)/2.0;
                let cur_channel = channel as i32;
                let start = max(0, cur_channel - tmp.floor() as i32) as usize;
                let end = min(limit, cur_channel + tmp.ceil() as i32 + 1) as usize;
                for i_h in 0..h{
                    for j_w in 0..w{
                        let sum_tmp = tensor.slice(s![batch,start..end,i_h, j_w])
                            .mapv(|val| val.powf(2.0))
                            .sum();
                        square_sum[[batch, channel, i_h, j_w]] = sum_tmp;
                    }
                }
            }
        }

        //y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        let tmp = (self.bias + (self.alpha / self.size as f32) * square_sum).mapv(|val| val.powf(self.beta));
        let normalized_values = tensor / tmp;
        let outlen = Vec::from(normalized_values.shape());
        return Output::TensorD(normalized_values.into_shape(IxDyn(&outlen)).unwrap());
    }

    fn op_type(&self) -> &'static str {
        return "LRN";
    }
}