use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_render(c: &mut Criterion) {
    // TODO: add rendering benchmarks
    c.bench_function("placeholder", |b| b.iter(|| {}));
}

criterion_group!(benches, benchmark_render);
criterion_main!(benches);
