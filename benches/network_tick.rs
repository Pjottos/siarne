use siarne::Network;
use criterion::{criterion_group, criterion_main, Criterion, black_box};


fn run_bench(c: &mut Criterion) {
    // let mut net = Network::new(1024, 128);
    // c.bench_function("net 1024 128", |b| b.iter(|| net.tick()));

    let mut net = black_box(Network::new(4096, 1024, 0, 0).unwrap());
    c.bench_function("net 4096 1024", |b| b.iter(|| net.tick()));
}

criterion_group!(benches, run_bench);
criterion_main!(benches);
