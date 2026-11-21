//! Slice-level fan-out under a caller-granted thread budget.
//!
//! RFC 9043 §4.5 Slices are independently entropy-coded (each owns its
//! range-coder / Golomb-Rice bit stream and its per-context state), and
//! §5 bounds a Slice's raster footprint above CIF "to ensure that fast
//! multithreaded decoding is possible". The frame drivers therefore run
//! one job per Slice.
//!
//! The policy is the fleet's: **serial unless the caller grants
//! threads**. A budget of `0` or `1` runs every job inline on the
//! calling thread — no thread is spawned, no synchronisation is touched.
//! A larger budget spawns `min(budget, jobs) - 1` scoped workers and the
//! calling thread takes a share too. Jobs are handed out through one
//! atomic counter; results are returned **in job-index order**, so a
//! driver that folds them in order produces byte-identical output at
//! every budget.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Run `job(0) .. job(count - 1)` and return the results in index order.
///
/// `threads <= 1` (or `count <= 1`) is a plain in-order loop on the
/// calling thread. Otherwise `min(threads, count)` workers (the caller
/// included) pull indexes from a shared counter. A panicking job
/// propagates the panic to the caller once the scope joins.
pub(crate) fn run_indexed<T, F>(count: usize, threads: usize, job: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let workers = threads.min(count);
    if workers <= 1 {
        return (0..count).map(job).collect();
    }
    let next = AtomicUsize::new(0);
    let drain = || {
        let mut produced: Vec<(usize, T)> = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                break;
            }
            produced.push((index, job(index)));
        }
        produced
    };
    let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (1..workers).map(|_| scope.spawn(drain)).collect();
        for (index, value) in drain() {
            slots[index] = Some(value);
        }
        for handle in handles {
            match handle.join() {
                Ok(produced) => {
                    for (index, value) in produced {
                        slots[index] = Some(value);
                    }
                }
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.expect("every job index is claimed exactly once"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::run_indexed;

    #[test]
    fn results_are_in_index_order_at_every_budget() {
        for threads in [0usize, 1, 2, 3, 8, 64] {
            let out = run_indexed(37, threads, |i| i * i);
            let want: Vec<usize> = (0..37).map(|i| i * i).collect();
            assert_eq!(out, want, "threads = {threads}");
        }
    }

    #[test]
    fn empty_and_single_job_sets() {
        assert!(run_indexed(0, 8, |i| i).is_empty());
        assert_eq!(run_indexed(1, 8, |i| i + 5), vec![5]);
    }
}
