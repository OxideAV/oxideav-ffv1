#![no_main]
//! `config_record` fuzz target — §4.2 Configuration Record + §4.1
//! Quantization Table Set cascade parse panic-freedom.
//!
//! Drives attacker-controlled bytes straight through the crate's public
//! [`oxideav_ffv1::parse_configuration_record`] (RFC 9043 §4.2 Parameters
//! + §4.3.2 CRC trailer) and
//! [`oxideav_ffv1::parse_quantization_table_sets`] (the same Parameters
//! plus the §4.1 per-context Quantization Table Set cascade). Both run
//! the §3.8.1 range coder over attacker bytes, so every Parameter field
//! (§4.2.1 version, §4.2.3 coder_type, §4.2.5 colorspace, §4.2.6
//! chroma_planes, §4.2.8 / §4.2.9 subsample shifts, §4.2.7 bit depth) and
//! every quant-table delta is attacker-chosen.
//!
//! The contract under test: no input shape may panic. A malformed record
//! must surface a typed [`oxideav_ffv1::Error`], never an out-of-bounds
//! index, an arithmetic overflow, or an `unwrap` on a value the attacker
//! forced to `None` / `Err`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Both entries parse the §4.2 Parameters off the same range-coded
    // bytes; the cascade parser additionally walks the §4.1 quant-table
    // sets when the parsed version carries them. Success and every typed
    // error are acceptable; only a panic is a finding.
    let _ = oxideav_ffv1::parse_configuration_record(data);
    let _ = oxideav_ffv1::parse_quantization_table_sets(data);
});
