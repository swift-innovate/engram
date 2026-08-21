//! LE-f32 BLOB codec.
//!
//! Matches the TypeScript storage format: `Buffer.from(Float32Array.buffer)` —
//! little-endian f32, no header, dimension-agnostic. A 768-dim vector produces
//! a 3072-byte BLOB (768 × 4 bytes).

/// Encode a slice of f32 values as a little-endian byte BLOB.
pub fn encode_f32_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a little-endian f32 BLOB back to a `Vec<f32>`.
///
/// Returns `None` if `bytes.len()` is not a multiple of 4.
pub fn decode_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    // `as_chunks::<4>` over `chunks_exact(4)`: it yields `&[u8; 4]` directly,
    // so `from_le_bytes` needs no re-indexing, and clippy's
    // `chunks_exact_to_as_chunks` (new in Rust 1.98) requires it under
    // `-D warnings`. The discarded remainder is always empty — the length
    // check above guarantees a multiple of 4.
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}
