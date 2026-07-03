//! Frame-encode benchmarks across the coder × depth × colorspace ×
//! slice-count matrix (see `benches/common/mod.rs` for the scenario
//! table and input synthesis).
//!
//! Times `encode_frame` alone (which dispatches to the Golomb-Rice /
//! range-coder / RGB drivers on the record); the produced bytes for
//! every scenario are pinned byte-exact by `tests/optimization_pins.rs`.
//!
//! Run with:
//!     cargo bench -p oxideav-ffv1 --bench encode

mod common;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_ffv1::{encode_frame, QuantizationTableSet};
use std::hint::black_box;

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    let qts: Vec<QuantizationTableSet> = vec![common::default_like_qts()];
    for s in common::SCENARIOS {
        let cr = common::make_cr(s.colorspace, s.coder_type, s.bits, s.num_h, s.num_v);
        let headers = common::make_headers(s.num_h, s.num_v);
        let frame = common::make_frame(s);

        group.throughput(Throughput::Bytes(common::raw_bytes(&frame)));
        group.bench_function(s.name, |b| {
            b.iter(|| {
                let bytes = encode_frame(black_box(&frame), &cr, &qts, &headers, true)
                    .expect("bench encode");
                black_box(bytes)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
