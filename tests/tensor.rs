use neural_net::Tensor;

#[test]
fn new_tensor() {
    let mut tensor: Tensor<3> = Tensor::default();

    assert_eq!(tensor.sum(), 0.);
    assert_eq!(tensor.length_sqrd(), 0.);
    assert_eq!(tensor.length(), 0.);

    tensor.set(0, 2.);

    assert_eq!(tensor.sum(), 2.);
    assert_eq!(tensor.length_sqrd(), 4.);
    assert_eq!(tensor.length(), 2.);
}
