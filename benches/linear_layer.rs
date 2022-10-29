use criterion::{black_box, criterion_group, criterion_main, Criterion};

use neural_net::Tensor;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor Group");
    group.sample_size(10_000);

    let mut rng = rand::thread_rng();

    let lhs = Tensor::<150>::random_new(&mut rng);
    let rhs = Tensor::<150>::random_new(&mut rng);

    group.bench_function("Tensor product_sum", |b| {
        b.iter(|| assert_ne!(0., black_box(lhs.sum_product(&rhs))))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
