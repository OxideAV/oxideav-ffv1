//! Frame-decode benchmarks across the coder × depth × colorspace ×
//! slice-count matrix (see `benches/common/mod.rs` for the scenario
//! table and input synthesis).
//!
//! Each scenario encodes its input once in setup (via the crate's own
//! encoder — FFV1 decode inputs must be conforming streams, and the
//! encoder's output is pinned byte-exact by `tests/optimization_pins.rs`)
//! and then times `decode_frame` / `decode_frame_rgb` alone.
//!
//! Run with:
//!     cargo bench -p oxideav-ffv1 --bench decode

mod common;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_ffv1::{decode_frame, decode_frame_rgb, encode_frame, ColorspaceType};
use oxideav_ffv1::{FramePixelDimensions, QuantizationTableSet};
use std::hint::black_box;

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    let qts: Vec<QuantizationTableSet> = vec![common::default_like_qts()];
    for s in common::SCENARIOS {
        let cr = common::make_cr(s.colorspace, s.coder_type, s.bits, s.num_h, s.num_v);
        let headers = common::make_headers(s.num_h, s.num_v);
        let frame = common::make_frame(s);
        let bytes = encode_frame(&frame, &cr, &qts, &headers, true)
            .unwrap_or_else(|e| panic!("{}: setup encode failed: {e:?}", s.name));
        let dims = FramePixelDimensions::new(common::W, common::H).unwrap();

        group.throughput(Throughput::Bytes(common::raw_bytes(&frame)));
        group.bench_function(s.name, |b| {
            b.iter(|| {
                let decoded = match s.colorspace {
                    ColorspaceType::YCbCr => decode_frame(black_box(&bytes), &cr, &qts, dims, true),
                    ColorspaceType::Rgb => {
                        decode_frame_rgb(black_box(&bytes), &cr, &qts, dims, true)
                    }
                }
                .expect("bench decode");
                black_box(decoded)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
