use ndarray::{Array4, ArrayView4, concatenate, Axis, IxDyn};
use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct Concat; // l'asse lungo il quale effettuare la concatenazione è sempre axis=1

impl Concat {

    #![allow(dead_code)]
    pub fn new() -> Concat {
        Concat
    }


    pub fn parse_from_proto_node() -> Concat {
        Concat
    }
}

impl Compute for Concat {
    fn compute(&mut self, input: Input) -> Output {

        let mut matrices: Vec<Array4<f32>> = Vec::new();

        match input{
            Input::Tensor4List(array) => {
                for input1 in array.iter(){
                    let element: Array4<f32> = input1.clone().into_dimensionality().unwrap();
                    matrices.push(element.into_dimensionality().unwrap());
                }
            },
            _ => panic!("wrong input type in the list"),
        }

        let first_shape = matrices[0].shape();

        for (_, tensor) in matrices.iter().enumerate() {
            let tensor_shape = tensor.shape();
            if tensor_shape[0] != first_shape[0]
                || tensor_shape[2] != first_shape[2]
                || tensor_shape[3] != first_shape[3]
            {
                panic!("mismatch input dimensions")
            }
        }

        // Applico il metodo view() a ciascun elemento della lista; questo serve per poter applicare
        // la funzione Concat agli elementi della lista
        let views: Vec<ArrayView4<f32>> = matrices.iter().map(|array| array.view()).collect();

        // Eseguo la concatenazione lungo l'asse 1
        let concatenated = concatenate(Axis(1), &views).unwrap();

        let out_len  = Vec::from(concatenated.shape());
        // Questa linea estrae le dimensioni della matrice concatenated e le converte in un vettore.

        return Output::TensorD(concatenated.into_shape(IxDyn(&out_len)).unwrap());

    }

    fn op_type(&self) -> &'static str {
        return "Concat";
    }
}