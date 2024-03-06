use ndarray::{Array1, Array4, arr1, Shape, Dim, Dimension, IxDyn, s};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto};

#[derive(Clone, Debug)]
pub struct AveragePool{
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    strides: Array1<i32>,
}

impl AveragePool{

    #![allow(dead_code)]
    pub fn new(
        kernel_shape: Option<Shape<Dim<[usize; 2]>>>,
        pads: Option<ndarray::Array1<i32>>,
        strides: Option<ndarray::Array1<i32>>, ) -> AveragePool{
        return AveragePool{
            kernel_shape: kernel_shape.unwrap_or(Shape::from(Dim([1, 1]))),
            pads: pads.unwrap_or(arr1(&[1, 1, 1, 1])),
            strides: strides.unwrap_or(arr1(&[1, 1]))
        }

    }


    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> AveragePool{
        let mut kernel_shape= Shape::from(Dim([1, 1]));
        let mut kernel_vec: [usize; 2] = [0; 2];
        let mut pads = Default::default();
        let mut strides = Default::default();
        for attr in attributes.iter(){
            match attr.name.as_str(){
                "kernel_shape" => {
                    let tmp = attr.ints.iter().map(|val| *val as usize).collect::<Vec<usize>>();
                    kernel_vec.copy_from_slice(&tmp);
                    kernel_shape = Shape::from(Dim(kernel_vec));
                },
                "strides" => {
                    strides = arr1(&attr.ints.iter().map(|val| *val as i32).collect::<Vec<i32>>());
                },
                "pads" => {
                    pads = arr1(&attr.ints.iter().map(|val| *val as i32).collect::<Vec<i32>>());
                },
                _ => ()
            }

        }
        return AveragePool{kernel_shape, pads, strides }
    }

}

impl Compute for AveragePool {
    fn compute(&mut self, inputs: Input) -> Output {
        let out = match inputs{
            Input::TensorD(array) => array,
            _ => panic!("Wrong input")
        };

        let mut x: Array4<f32> = out.into_dimensionality().unwrap();

        // Padding
        let left_h = self.pads[0] as usize;
        let left_w = self.pads[1] as usize;
        let right_h = self.pads[2] as usize;
        let right_w = self.pads[3] as usize;
        let kernel_size = self.kernel_shape.raw_dim().last_elem();
        let stride_h = self.strides[0] as usize;
        let stride_w = self.strides[1] as usize;


        let output_dims = [x.shape()[0],
            x.shape()[1],
            ((x.shape()[2] - kernel_size + left_h + right_h)/stride_h + 1),
            ((x.shape()[3] - kernel_size + left_w + right_w)/stride_w + 1)];
        let mut result: Array4<f32> = Array4::from_elem(output_dims.clone(), 0.0);

        let (b, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);

        //Create padded image
        let mut padded_image = Array4::<f32>::zeros((b, c, h + left_h + right_h, w + left_w + right_w));
        let original_view = x.view_mut();
        let mut padded_view = padded_image.slice_mut(s![.., .., left_h..left_h + h, left_w..left_w + w]);
        padded_view.assign(&original_view);
        x = padded_image;


        //Input dims
        for batch in 0..b{
            for channel in 0..c{
                //outdim h
                for i in 0..output_dims[2]{
                    //outdim w
                    for j in 0..output_dims[3]{
                        //moving the kernel
                        let mut sum_num = 0.0;
                        for m in 0..kernel_size{
                            for n in 0..kernel_size {
                                sum_num += x[[batch, channel, (i * stride_h + m), (j * stride_w + n)]];
                            }
                        }
                        //after kernel assign the value
                        result[[batch, channel, i, j]] = sum_num / (kernel_size as f32 * kernel_size as f32);
                    }
                }
            }
        }
        return Output::TensorD(result.into_shape(IxDyn(&output_dims)).unwrap());
    }

    fn op_type(&self) -> &'static str {
        return "AveragePool";
    }
}