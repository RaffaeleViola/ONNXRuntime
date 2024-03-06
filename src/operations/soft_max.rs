use ndarray::{Array2, Axis, IxDyn};
use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct SoftMax;

impl SoftMax {

    #![allow(dead_code)]
    pub fn new() -> SoftMax {
        SoftMax
    }


    pub fn parse_from_proto_node() -> SoftMax {
        SoftMax
    }
}

impl Compute for SoftMax {
    fn compute(&mut self, input: Input) -> Output {

        let matrix: Array2<f32> = match input {
            Input::TensorD(array) => array.into_dimensionality().unwrap(),
            _ => panic!("wrong input type")
        }; //Questa parte del codice gestisce il passaggio dalla variante Input al tipo Array2<f32>
        let max_values = matrix.fold_axis(Axis(1),f32::NEG_INFINITY,
                                          |a, b| a.max(b.clone()));
        //Questa linea calcola il massimo valore in ciascuna riga della matrice matrix e restituisce un nuovo vettore max_values contenente i massimi valori.

        let len = max_values.len();
        let subtracted = matrix - max_values.into_shape((len, 1)).unwrap();
        //Qui vengono sottratti i massimi valori calcolati in precedenza da ciascun elemento della matrice matrix

        let exp_values = subtracted.mapv(|x| x.exp());
        //: Questa riga calcola il valore esponenziale di ciascun elemento della matrice subtracted.

        let sum_exp = exp_values.fold_axis(Axis(1), 0.0, |a, b| a + b);
        // Questa linea calcola la somma degli elementi esponenziali in ciascuna riga della matrice exp_values.

        let len = sum_exp.len();

        let softmax_values: Array2<f32> = exp_values / sum_exp.into_shape((len, 1)).unwrap();
        //Qui, viene calcolato il softmax dividendo ogni elemento dell'exp_values per la somma degli elementi sum_exp.
        // Questo restituisce una matrice di valori softmax.
        //In questa riga, exp_values è una matrice di valori esponenziali di dimensioni (n, m) e sum_exp è un vettore di somme di valori esponenziali di dimensione (n,). Il broadcasting avviene quando si esegue l'operazione di divisione tra exp_values e sum_exp.
        //
        // Nel broadcasting, Rust allinea automaticamente le dimensioni degli array in modo che
        // l'operazione possa essere eseguita senza errori. In questo caso, sum_exp è di dimensione (n,),
        // ma viene allineato in modo che possa essere diviso per exp_values che è di dimensione (n, m).
        // Il risultato di questa operazione di divisione sarà una matrice di dimensione (n, m) in
        // cui ogni elemento della riga i è stato diviso per sum_exp[i].

        let out_len  = Vec::from(softmax_values.shape());
        // Questa linea estrae le dimensioni della matrice softmax_values e le converte in un vettore.

        return Output::TensorD(softmax_values.into_shape(IxDyn(&out_len)).unwrap());

    }

    fn op_type(&self) -> &'static str {
        return "SoftMax";
    }
}
