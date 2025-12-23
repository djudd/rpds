use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rpds::HashTrieMap;
use std::hint::black_box;

fn rpds_hash_trie_map_choose(c: &mut Criterion) {
    let mut group = c.benchmark_group("rpds hash trie map choose");

    for size in [100, 1_000, 10_000, 100_000] {
        let mut map = HashTrieMap::new();
        for i in 0..size {
            map.insert_mut(i, -(i as isize));
        }

        group.bench_with_input(BenchmarkId::new("choose", size), &map, |b, map| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| {
                black_box(map.choose(&mut rng))
            });
        });
    }

    group.finish();
}

fn rpds_hash_trie_map_choose_multiple(c: &mut Criterion) {
    let mut group = c.benchmark_group("rpds hash trie map choose_multiple");

    let size = 100_000;
    let mut map = HashTrieMap::new();
    for i in 0..size {
        map.insert_mut(i, -(i as isize));
    }

    for amount in [10, 100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("choose_multiple", amount),
            &amount,
            |b, &amount| {
                let mut rng = StdRng::seed_from_u64(42);
                b.iter(|| {
                    for kv in map.choose_multiple(&mut rng, amount) {
                        black_box(kv);
                    }
                });
            },
        );
    }

    group.finish();
}

fn rpds_hash_trie_map_choose_vs_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("rpds hash trie map choose vs get");

    let size = 100_000;
    let mut map = HashTrieMap::new();
    for i in 0..size {
        map.insert_mut(i, -(i as isize));
    }

    group.bench_function("choose x1000", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| {
            for _ in 0..1000 {
                black_box(map.choose(&mut rng));
            }
        });
    });

    group.bench_function("get x1000", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| {
            for _ in 0..1000 {
                let key = rng.random_range(0..size);
                black_box(map.get(&key));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    rpds_hash_trie_map_choose,
    rpds_hash_trie_map_choose_multiple,
    rpds_hash_trie_map_choose_vs_get
);
criterion_main!(benches);
