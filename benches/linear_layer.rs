use criterion::{black_box, criterion_group, criterion_main, Criterion};

use neural_net::Matrix;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Group");
    group.sample_size(10_000);

    let mut rng = rand::thread_rng();

    let lhs = Matrix::<150, 1>::random_new(&mut rng);
    let rhs = Matrix::<150, 1>::random_new(&mut rng);

    group.bench_function("Matrix product_sum", |b| {
        b.iter(|| assert_ne!(0., black_box(lhs.sum_product(&rhs))))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
