use ndarray::{Array1, Array4, arr1, Shape, Dim, Dimension, s, Axis};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto};
use std::ops::Div;
use ndarray::parallel::prelude::*;

#[derive(Clone, Debug)]
pub struct Conv{
    pub autopad: String,
    pub dilations: Array1<i32>,
    pub group: u32,
    pub kernel_shape: Shape<Dim<[usize; 2]>>,
    pub pads: Array1<i32>,
    pub strides: Array1<i32>,
}

impl Conv {
    pub fn new(ap: Option<String>,
               dil: Option<Array1<i32>>,
               group: Option<u32>,
               kernel_shape: Option<Shape<Dim<[usize; 2]>>>,
               pads: Option<Array1<i32>>,
               strides: Option<Array1<i32>>, ) -> Conv {
        return Conv {
            autopad: ap.unwrap_or("NOT_SET".to_string()),
            dilations: dil.unwrap_or(arr1(&[1, 1])),
            group: group.unwrap_or(1),
            kernel_shape: kernel_shape.unwrap_or(Shape::from(Dim([1, 1]))),
            pads: pads.unwrap_or(arr1(&[0, 0, 0, 0])),
            strides: strides.unwrap_or(arr1(&[1, 1]))
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Conv {

        let mut conv_tmp = Conv::new(None, None, None,
                                     None, None, None);
        for attr in attributes.iter() {
            match attr.name.as_str() {
                "auto_pad" => {
                    let string_result: Result<String, _> = String::from_utf8(attr.s.clone());

                    // Check if the conversion was successful
                    match string_result {
                        Ok(string) => {
                            conv_tmp.autopad = string;
                        }
                        Err(e) => {
                            panic!("Error decoding Vec<u8>: {:?}", e);
                        }
                    }
                },
                "autopad" => {
                    let string_result: Result<String, _> = String::from_utf8(attr.s.clone());

                    // Check if the conversion was successful
                    match string_result {
                        Ok(string) => {
                            conv_tmp.autopad = string;
                        }
                        Err(e) => {
                            panic!("Error decoding Vec<u8>: {:?}", e);
                        }
                    }
                },
                "dilations" => {
                    // Step 1: Convert each element from i64 to i32
                    let converted_attr: Vec<i32> = attr.ints.clone().into_iter().map(|x| x as i32).collect();
                    // Step 2: Create an Array1<i32> from the converted Vec<i32>
                    let array_attr: Array1<i32> = arr1(&converted_attr);
                    conv_tmp.dilations = array_attr;
                },
                "group" => {
                    let converted_attr: u32 = attr.i.clone() as u32;
                    conv_tmp.group = converted_attr;
                },
                "kernel_shape" => {
                    //Convert from Vec<i64> to Shape<Dim<[usize; 2]>>
                    let mut kernel_vec: [usize; 2] = [0; 2];
                    let input = attr.ints.clone().into_iter().map(|val| val as usize).collect::<Vec<usize>>();
                    kernel_vec.copy_from_slice(&input);
                    conv_tmp.kernel_shape = Shape::from(Dim(kernel_vec));
                },
                "pads" => {
                    // Step 1: Convert each element from i64 to i32
                    let converted_attr: Vec<i32> = attr.ints.clone().into_iter().map(|x| x as i32).collect();
                    // Step 2: Create an Array1<i32> from the converted Vec<i32>
                    let array_attr: Array1<i32> = arr1(&converted_attr);
                    conv_tmp.pads = array_attr;
                },
                "strides" => {
                    // Step 1: Convert each element from i64 to i32
                    let converted_attr: Vec<i32> = attr.ints.clone().into_iter().map(|x| x as i32).collect();
                    // Step 2: Create an Array1<i32> from the converted Vec<i32>
                    let array_attr: Array1<i32> = arr1(&converted_attr);
                    conv_tmp.strides = array_attr;
                },

                _ => panic!("Attribute name not known")
            }
        }
        return conv_tmp;
    }
}
impl Compute for Conv{

    fn compute(&mut self, inputs: Input) -> Output {
        let autopad = self.autopad.clone();
        let dilations = self.dilations.clone();
        let kernel_shape = self.kernel_shape.clone();
        let pads: Array1<i32> = self.pads.clone();
        let strides = self.strides.clone();

        let vec = match inputs {
            Input::Tensor4List(vec_array) => vec_array,
            _ => panic!("Input is not a vector")
        };
        // let mut x = vec[0].clone();
        let x1 = &vec[0];
        let mut x: Array4<f32> = x1.clone().into_dimensionality().unwrap();
        let w2 = &vec[1];
        let w1 = w2.clone();

        // Get the size of each dimension
        let shape = x.shape();
        let ( b,  c,  h,  w) = match shape.len() {
            4 => {
                (shape[0].clone(),
                shape[1].clone(),
                shape[2].clone(),
                shape[3].clone())
            },
            _ => panic!("Unexpected number of dimensions in the tensor"),
        };


        // Retrieve weight dimensions

        let (m, _, _, _) = {
            // Get the size of each dimension
            let shape = w1.shape();
            match shape.len() {
                4 => (shape[0].clone(), shape[1].clone(), shape[2].clone(), shape[3].clone()),
                _ => panic!("Unexpected number of dimensions in the array"),
            }
        };

        let mut bias = Array1::<f32>::zeros(m);

        match vec.get(2) {
            Some(element) => bias = element.clone().into_shape((m,)).unwrap(),
            _=> bias = bias,
        };


        // Padding
        let mut left_h = pads[0] as usize;
        let mut left_w = pads[1] as usize;
        let mut right_h = pads[2] as usize;
        let mut right_w = pads[3] as usize;
        let kernel_size = kernel_shape.raw_dim().last_elem();
        let stride_h = strides[0] as usize;
        let stride_w = strides[1] as usize;

        // Calculate output dimensions based on autopad

        match autopad.as_str() {
            "SAME_UPPER" | "SAME_LOWER" => {
                let width_padding_difference = kernel_size/stride_w - 1;
                //I'd get the same value with height_padding_difference
                if width_padding_difference % 2 == 0 {
                    left_h = width_padding_difference.clone().div(2);
                    right_h = width_padding_difference.clone().div(2);
                    left_w = width_padding_difference.clone().div(2);
                    right_w = width_padding_difference.clone().div(2);
                }else{
                    if autopad == "SAME_LOWER" {
                        left_h = width_padding_difference.clone().div(2) + 1;
                        right_h = width_padding_difference.clone().div(2);
                        left_w = width_padding_difference.clone().div(2) + 1;
                        right_w = width_padding_difference.clone().div(2);
                    }else {
                        left_h = width_padding_difference.clone().div(2);
                        right_h = width_padding_difference.clone().div(2) + 1;
                        left_w = width_padding_difference.clone().div(2);
                        right_w = width_padding_difference.clone().div(2) + 1;
                    }
                }
            }
            "VALID" | "NOT_SET" => {

            },
            _ => panic!("Invalid autopad mode")
        };

        let oh = ((h + left_h + right_h - dilations[1] as usize * (kernel_size))/stride_h) + 1;
        let ow = ((w + left_w + right_w - dilations[1] as usize * (kernel_size))/stride_w) + 1;
        //Create padded image

        //create an image by taking into account the padding; this is the padded input, not the output
        let mut padded_image = Array4::<f32>::zeros((b, c, h + left_h + right_h, w + left_w + right_w));
        //generate a mutable copy of x
        let original_view = x.view_mut();
        //generate a view on padded_image by only considering the pixels without padding
        let mut padded_view = padded_image.slice_mut(s![.., .., left_h..left_h + h, left_w..left_w + w]);
        //now x is the original image + the padded values
        padded_view.assign(&original_view);
        x = padded_image;


        // Initialize output tensor
        let mut y = Array4::<f32>::zeros((b, m, oh, ow));

        let w_arr: Array4<f32> = w1.into_dimensionality().unwrap();
        for batch in 0..b {
            for h in 0..oh {
                for w in 0..ow {
                    let input_slice = x.slice(s![
                        batch,
                        ..,
                        h * stride_h..h * stride_h + kernel_size,
                        w * stride_w..w * stride_w + kernel_size
                    ]);

                    //let mut convolution_result = &input_slice * &w1;
                    let mut results = Vec::new();
                    w_arr.clone()
                        .axis_iter(Axis(0))
                        .into_par_iter()
                        .map(|v| (&v * &input_slice).sum())
                        .collect_into_vec(&mut results);

                    let mut convolution_result= Array1::from(results);
                    //convolution_result = convolution_result.sum_axis(Axis(1)).sum_axis(Axis(1)).sum_axis(Axis(1));
                    convolution_result = convolution_result + bias.clone();

                    // Assign the result to the output array
                    let mut slice_y = y.slice_mut(s![batch, .., h, w]);
                    slice_y.assign(&convolution_result);

                }
            }
        }
        Output::TensorD(y.into_dyn())
    }

    fn op_type(&self) -> &'static str {
        return "Conv";
    }
}
