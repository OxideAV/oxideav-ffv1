//! FFV1 configuration record (RFC 9043 §4.2).
//!
//! The configuration record is a range-coded block describing stream-level
//! parameters (version, colorspace, chroma subsampling, slice grid, quant
//! tables, etc.) followed by a 32-bit CRC. FFV1 stores it in the container's
//! extradata. The encoder generates one; the decoder parses it.
//!
//! We implement the minimum shape needed for 8-bit or 10-bit YUV 4:2:0 /
//! 4:2:2 / 4:4:4 version 3 files: coder_type=1 (range, default states),
//! intra=1, ec=0 (CRC of the config record itself still emitted so FFmpeg
//! accepts the record).

use oxideav_core::{Error, Result};

use crate::crc::crc32_ieee;
use crate::range_coder::{RangeDecoder, RangeEncoder, StateTransition, DEFAULT_STATE_TRANSITION};

/// Parsed FFV1 configuration record — a superset of what this codec supports,
/// so we can round-trip foreign producers' records and fail cleanly.
#[derive(Clone, Debug, Default)]
pub struct ConfigRecord {
    pub version: u32,
    pub micro_version: u32,
    pub coder_type: u32,
    pub colorspace_type: u32, // 0 = YCbCr, 1 = RGB (JPEG 2000 RCT)
    pub bits_per_raw_sample: u32,
    pub chroma_planes: bool,
    pub log2_h_chroma_subsample: u32,
    pub log2_v_chroma_subsample: u32,
    pub extra_plane: bool, // alpha
    pub num_h_slices: u32,
    pub num_v_slices: u32,
    pub quant_table_set_count: u32,
    pub ec: u32,
    pub intra: u32,
    /// `state_transition_delta[1..256]`. Only populated when `coder_type == 2`
    /// (RFC 9043 §4.2.4). Applied on top of FFV1's default state transition
    /// table for every per-slice range coder the decoder instantiates.
    pub state_transition_delta: Option<[i16; 256]>,
}

impl ConfigRecord {
    /// Construct a fresh config record for our simplest supported shape: 8-bit
    /// YCbCr, 4:2:0 or 4:4:4, one slice, default range coder states, intra.
    pub fn new_simple(yuv444: bool) -> Self {
        Self::new_yuv(8, if yuv444 { 0 } else { 1 }, if yuv444 { 0 } else { 1 })
    }

    /// Construct a config record for YCbCr with the given bit depth and
    /// log2 chroma subsampling on each axis. `bits` is 8 or 10; other values
    /// are accepted but this crate does not currently encode them.
    pub fn new_yuv(bits: u32, log2_h: u32, log2_v: u32) -> Self {
        Self {
            version: 3,
            micro_version: 4,
            coder_type: 1, // range coder with default state transition
            colorspace_type: 0,
            bits_per_raw_sample: bits,
            chroma_planes: true,
            log2_h_chroma_subsample: log2_h,
            log2_v_chroma_subsample: log2_v,
            extra_plane: false,
            num_h_slices: 1,
            num_v_slices: 1,
            quant_table_set_count: 1,
            ec: 0,
            intra: 1,
            state_transition_delta: None,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut enc = RangeEncoder::new();
        // FFmpeg shares a single 32-byte state buffer across all symbol and
        // rac calls in the extradata — `put_rac(state, ...)` uses state[0]
        // (the same byte as `put_symbol`'s zero-bit marker). That subtle
        // detail matters for bit-for-bit compatibility.
        let mut state = [128u8; 32];
        enc.put_symbol_u(&mut state, self.version);
        if self.version >= 3 {
            enc.put_symbol_u(&mut state, self.micro_version);
        }
        enc.put_symbol_u(&mut state, self.coder_type);
        // state_transition_delta omitted (coder_type == 1).
        enc.put_symbol_u(&mut state, self.colorspace_type);
        if self.version >= 1 {
            enc.put_symbol_u(&mut state, self.bits_per_raw_sample);
        }
        enc.put_rac(&mut state[0], self.chroma_planes);
        enc.put_symbol_u(&mut state, self.log2_h_chroma_subsample);
        enc.put_symbol_u(&mut state, self.log2_v_chroma_subsample);
        enc.put_rac(&mut state[0], self.extra_plane);
        enc.put_symbol_u(&mut state, self.num_h_slices.saturating_sub(1));
        enc.put_symbol_u(&mut state, self.num_v_slices.saturating_sub(1));
        enc.put_symbol_u(&mut state, self.quant_table_set_count);
        // Emit the configured number of quantisation-table sets. Each set
        // has 5 sub-tables; we always emit FFmpeg's default sub-tables.
        for _ in 0..self.quant_table_set_count {
            emit_default_quant_table_set(&mut enc);
        }
        // initial_state_delta == false for each set (no custom states).
        for _ in 0..self.quant_table_set_count {
            enc.put_rac(&mut state[0], false);
        }
        if self.version >= 3 {
            enc.put_symbol_u(&mut state, self.ec);
            enc.put_symbol_u(&mut state, self.intra);
        }

        let mut bytes = enc.finish_for_extradata();
        // Append the CRC-32 (IEEE polynomial 0x04C11DB7) of the preceding
        // bytes in big-endian order. Chosen so that the CRC of the whole
        // record (including these 4 bytes) is zero — FFmpeg validates this.
        let crc = crc32_ieee(&bytes);
        bytes.extend_from_slice(&crc.to_be_bytes());
        bytes
    }

    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(Error::invalid("FFV1 config record too short"));
        }
        // Verify the trailing 32-bit CRC-32 parity: CRC of the whole record
        // (body + stored CRC bytes) must be zero. Reject on mismatch, which
        // is what FFmpeg does for a v3 extradata.
        if crc32_ieee(data) != 0 {
            return Err(Error::invalid("FFV1 config record CRC mismatch"));
        }
        let body = &data[..data.len() - 4];
        let mut dec = RangeDecoder::new(body);
        let mut state = [128u8; 32];

        let version = dec.get_symbol_u(&mut state);
        if version != 3 {
            return Err(Error::unsupported(format!("FFV1 version {version}")));
        }
        let micro_version = dec.get_symbol_u(&mut state);
        // `ac` / `coder_type`: 0 = Golomb-Rice, 1 = range-coder default-tab,
        // 2 = range-coder custom-tab.
        let coder_type = dec.get_symbol_u(&mut state);
        if coder_type > 2 {
            return Err(Error::unsupported(format!(
                "FFV1 unknown coder_type={coder_type}"
            )));
        }
        // coder_type == 0 (Golomb-Rice) is supported for decode; coder_type ==
        // 1 (range coder, default state table) is supported for both encode
        // and decode; coder_type == 2 (custom state transition table) is
        // accepted on decode by materialising the deltas below.
        let state_transition_delta = if coder_type > 1 {
            // RFC 9043 §4.2.4: state_transition_delta[1..256] is range-coded
            // as a signed symbol using the per-field initial state buffer.
            // The config-record range coder itself continues to use the
            // *default* state transition table; the deltas only take effect
            // for range coders that are constructed later (per slice).
            let mut deltas = [0i16; 256];
            for i in 1..256 {
                deltas[i] = dec.get_symbol(&mut state, true) as i16;
            }
            Some(deltas)
        } else {
            None
        };
        let colorspace_type = dec.get_symbol_u(&mut state);
        if colorspace_type > 1 {
            return Err(Error::unsupported(format!(
                "FFV1 unknown colorspace_type={colorspace_type}"
            )));
        }
        let bits_per_raw_sample = dec.get_symbol_u(&mut state);
        if bits_per_raw_sample == 0 {
            // FFmpeg quirk: version-3 extradata encodes 8-bit as 0.
        } else if bits_per_raw_sample != 8 && bits_per_raw_sample != 10 {
            return Err(Error::unsupported(format!(
                "FFV1 {bits_per_raw_sample}-bit samples"
            )));
        }
        let chroma_planes = dec.get_rac(&mut state[0]);
        let log2_h_chroma_subsample = dec.get_symbol_u(&mut state);
        let log2_v_chroma_subsample = dec.get_symbol_u(&mut state);
        let extra_plane = dec.get_rac(&mut state[0]);
        if extra_plane {
            return Err(Error::unsupported("FFV1 alpha plane"));
        }
        // RFC 9043 §4.2.5: RGB (colorspace_type == 1) requires chroma_planes == 1
        // and log2_{h,v}_chroma_subsample == 0. Anything else is outside the
        // specification.
        if colorspace_type == 1
            && (!chroma_planes || log2_h_chroma_subsample != 0 || log2_v_chroma_subsample != 0)
        {
            return Err(Error::invalid(
                "FFV1 RGB requires chroma_planes=1 and no chroma subsampling",
            ));
        }
        let num_h_slices = dec.get_symbol_u(&mut state) + 1;
        let num_v_slices = dec.get_symbol_u(&mut state) + 1;
        let quant_table_set_count = dec.get_symbol_u(&mut state);
        if quant_table_set_count == 0 || quant_table_set_count > 8 {
            return Err(Error::invalid("FFV1 bad quant_table_set_count"));
        }
        // Read each quant-table set from the record. Only set 0 is used by
        // streams with `context_model = 0` (FFmpeg's default, and the only
        // shape this crate decodes): every slice's `qt_idx[p]` field picks
        // the table set for plane `p`, but our slice header verification
        // rejects any non-zero index, so foreign sets 1..N are read past and
        // dropped. Set 0 must match our defaults (most notably: foreign
        // 10-bit streams ship `quant9_10bit`, which would otherwise decode
        // to garbage silently).
        let expected = crate::state::default_quant_tables();
        for idx in 0..quant_table_set_count {
            let got = read_quant_table_set(&mut dec)?;
            if idx == 0 && got != expected {
                return Err(Error::unsupported(
                    "FFV1: non-default quantisation tables (e.g. FFmpeg's 10-bit quant9_10bit) not implemented",
                ));
            }
        }
        // One `states_coded` bit per quant_table_set.
        for _ in 0..quant_table_set_count {
            let states_coded = dec.get_rac(&mut state[0]);
            if states_coded {
                return Err(Error::unsupported("FFV1 initial_state_delta"));
            }
        }
        let ec = dec.get_symbol_u(&mut state);
        let intra = dec.get_symbol_u(&mut state);
        let bits_per_raw_sample = if bits_per_raw_sample == 0 {
            8
        } else {
            bits_per_raw_sample
        };

        Ok(Self {
            version,
            micro_version,
            coder_type,
            colorspace_type,
            bits_per_raw_sample,
            chroma_planes,
            log2_h_chroma_subsample,
            log2_v_chroma_subsample,
            extra_plane,
            num_h_slices,
            num_v_slices,
            quant_table_set_count,
            ec,
            intra,
            state_transition_delta,
        })
    }

    pub fn is_yuv420(&self) -> bool {
        self.chroma_planes && self.log2_h_chroma_subsample == 1 && self.log2_v_chroma_subsample == 1
    }

    pub fn is_yuv422(&self) -> bool {
        self.chroma_planes && self.log2_h_chroma_subsample == 1 && self.log2_v_chroma_subsample == 0
    }

    pub fn is_yuv444(&self) -> bool {
        self.chroma_planes && self.log2_h_chroma_subsample == 0 && self.log2_v_chroma_subsample == 0
    }

    /// True for streams using the JPEG 2000 reversible colour transform
    /// (colorspace_type == 1). FFV1 places `Y` on plane 0 (holding the green
    /// component post-RCT) with `Cb`/`Cr` on planes 1/2 (holding blue and
    /// red differences). Chroma planes are required at full resolution.
    pub fn is_rgb(&self) -> bool {
        self.colorspace_type == 1
    }

    /// Resolve the range-coder state transition table for slices of this
    /// stream: default for `coder_type <= 1`, or the default patched with
    /// the config record's `state_transition_delta` entries for
    /// `coder_type == 2` (ffmpeg emits this shape for RGB streams).
    pub fn slice_state_transition(&self) -> StateTransition {
        match self.state_transition_delta {
            Some(deltas) => {
                let mut tbl = DEFAULT_STATE_TRANSITION;
                // `default_state_transition[i] + state_transition_delta[i]`,
                // interpreted modulo 256 per RFC 9043 Figure 22 — the result
                // is taken as an unsigned byte for the one_state table. We
                // only patch indices 1..256; index 0 is always left as 0.
                for i in 1..256 {
                    let combined = (tbl[i] as i32) + (deltas[i] as i32);
                    tbl[i] = (combined & 0xFF) as u8;
                }
                StateTransition::from_table(&tbl)
            }
            None => StateTransition::default_ffv1(),
        }
    }
}

/// Emit one quantisation-table set containing the FFmpeg default
/// (`ffv1_context=true`). Five tables, 256 entries each. RFC encodes each
/// *half* of each table as a run-length of equal values, using the
/// range-coded unsigned symbol encoding.
fn emit_default_quant_table_set(enc: &mut RangeEncoder) {
    let set = crate::state::default_quant_tables();
    for tbl in &set {
        // The first half is entries [0..128) but encoded starting at index 0
        // by reading runs. Each iteration: scan how many consecutive entries
        // from position `i` have the same value as `i`, emit `run - 1`, then
        // advance `i` by `run`.
        let mut i: usize = 0;
        let mut state = [128u8; 32];
        while i < 128 {
            let v = tbl[i];
            let mut j = i + 1;
            while j < 128 && tbl[j] == v {
                j += 1;
            }
            let run = (j - i) as u32;
            enc.put_symbol_u(&mut state, run - 1);
            i = j;
        }
    }
}

/// Materialise one quantisation-table set (five 256-entry sub-tables) from
/// its run-length encoding in a range-coded config record.
///
/// Each sub-table is symmetric around index 128. FFmpeg's `read_quant_table`
/// fills positions 0..128 with `scale * v`, where `v` starts at 0 and
/// increments once per run; `scale` starts at 1 for the first sub-table and
/// is the previous table's returned value (`2 * last_v - 1`, after the
/// terminating `v++`) for subsequent sub-tables.
fn read_quant_table_set(dec: &mut RangeDecoder<'_>) -> Result<crate::state::QuantTables> {
    let mut set = [[0i16; 256]; 5];
    let mut scale: i32 = 1;
    for tbl in set.iter_mut() {
        let mut state = [128u8; 32];
        let mut pos: usize = 0;
        let mut v: i32 = 0;
        while pos < 128 {
            let run = (dec.get_symbol_u(&mut state) + 1) as usize;
            if pos + run > 128 {
                return Err(Error::invalid("FFV1 quant table run overshoot"));
            }
            let value = (scale * v) as i16;
            for i in 0..run {
                tbl[pos + i] = value;
            }
            pos += run;
            v += 1;
        }
        // Mirror the negative half: table[256 - i] = -table[i] for i in 1..=127,
        // and table[128] = -table[127].
        for i in 1..128 {
            tbl[256 - i] = -tbl[i];
        }
        tbl[128] = -tbl[127];
        // Next table's scale: `2 * v - 1` using the final incremented `v`.
        scale *= 2 * v - 1;
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple_420() {
        let c = ConfigRecord::new_simple(false);
        let bytes = c.encode();
        let parsed = ConfigRecord::parse(&bytes).expect("parse");
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.coder_type, 1);
        assert_eq!(parsed.bits_per_raw_sample, 8);
        assert!(parsed.is_yuv420());
        assert_eq!(parsed.num_h_slices, 1);
        assert_eq!(parsed.num_v_slices, 1);
        assert_eq!(parsed.quant_table_set_count, 1);
    }

    #[test]
    fn roundtrip_simple_444() {
        let c = ConfigRecord::new_simple(true);
        let bytes = c.encode();
        let parsed = ConfigRecord::parse(&bytes).expect("parse");
        assert!(parsed.is_yuv444());
    }

    #[test]
    fn rejects_mismatched_quant_table() {
        // Build a valid config record, then corrupt one byte inside the
        // first quant-table set. The parser must report the mismatch rather
        // than accept the stream and silently produce wrong pixels.
        let c = ConfigRecord::new_simple(false);
        let mut bytes = c.encode();
        // The run-length-coded quant tables live after the first 12 config
        // fields and before the 4-byte CRC footer; flipping any byte inside
        // them breaks at least one run. Byte 14 is well inside that region.
        bytes[14] ^= 0x80;
        // After mutating, recompute and patch the trailing CRC so that only
        // the quant-table content differs (not the integrity check).
        let crc = crc32_ieee(&bytes[..bytes.len() - 4]);
        let n = bytes.len();
        bytes[n - 4..].copy_from_slice(&crc.to_be_bytes());
        let err = ConfigRecord::parse(&bytes).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_) | Error::InvalidData(_)),
            "expected Unsupported/InvalidData, got {err:?}"
        );
    }
}
