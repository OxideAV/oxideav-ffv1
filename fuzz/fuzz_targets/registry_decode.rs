#![no_main]
//! `registry_decode` fuzz target — end-to-end `oxideav_core::Decoder`
//! trait panic-freedom.
//!
//! This is the realistic container attack surface: a muxer hands the
//! ffv1 decoder a [`oxideav_core::CodecParameters`] (RFC 9043 §4.3.3
//! extradata = the §4.2 Configuration Record, plus frame `width` /
//! `height`) and then feeds one or more coded [`oxideav_core::Packet`]s
//! through [`oxideav_core::Decoder::send_packet`] /
//! [`receive_frame`](oxideav_core::Decoder::receive_frame). The
//! attacker controls all three: the extradata bytes, the dimensions, and
//! every Packet payload.
//!
//! Two carriage shapes are exercised, selected by a header byte:
//!   * non-empty extradata — the v3 path (Configuration Record parse,
//!     then `decode_frame_with_carry` / `decode_frame_rgb_with_carry`).
//!   * empty extradata + dims — the versions-0/1 path (RFC 9043 §4.4
//!     inline Parameters parsed off the first keyframe Packet).
//!
//! Two Packets are fed so the §3.8.1.3 / §3.8.2.5 cross-Frame coder-state
//! carry (and, for v0/v1, the cached-record reuse on a non-keyframe) is
//! reached.
//!
//! Layout of the attacker buffer:
//!   byte 0      — width  (1..=MAX_DIM, after `% MAX_DIM + 1`)
//!   byte 1      — height (1..=MAX_DIM, after `% MAX_DIM + 1`)
//!   bytes 2..4  — little-endian extradata length, modulo remaining
//!   then        — `extradata_len` bytes of §4.3.3 extradata
//!   then        — the rest is the coded Packet payload (fed twice)
//!
//! The contract under test: no input shape may panic. A malformed stream
//! must surface a typed `oxideav_core::Error` (or decode), never an
//! out-of-bounds index, an arithmetic overflow, or an `unwrap` on a
//! value the attacker forced to `None` / `Err`.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, RuntimeContext, TimeBase};
use oxideav_ffv1::{register, CODEC_ID_STR};

/// Cap on each frame dimension; the decoders allocate plane buffers
/// proportional to `width * height`.
const MAX_DIM: u32 = 96;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let width = (u32::from(data[0]) % MAX_DIM) + 1;
    let height = (u32::from(data[1]) % MAX_DIM) + 1;
    let rest = &data[2..];

    let raw_len = usize::from(u16::from_le_bytes([rest[0], rest[1]]));
    let body = &rest[2..];
    let extradata_len = if body.is_empty() {
        0
    } else {
        raw_len % (body.len() + 1)
    };
    let (extradata, packet_payload) = body.split_at(extradata_len.min(body.len()));

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(width);
    params.height = Some(height);
    params.extradata = extradata.to_vec();

    let Some(mut dec) = ctx.codecs.first_decoder(&params).ok() else {
        return;
    };

    // Feed the same payload as two consecutive Packets so the cross-Frame
    // coder-state carry (and the v0/v1 cached-record reuse on the second,
    // non-keyframe Packet) is exercised. Success, `NeedMore`, `Eof`, and
    // every typed decode error are acceptable; only a panic is a finding.
    for pts in 0..2 {
        let pkt = Packet::new(pts, TimeBase::new(1, 1), packet_payload.to_vec());
        if dec.send_packet(&pkt).is_err() {
            break;
        }
        let _ = dec.receive_frame();
    }
});
