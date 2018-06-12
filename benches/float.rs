#![feature(test)]

extern crate test;
use test::Bencher;

extern crate simple_rnd;
use simple_rnd::{ Rng, StdRng };

const RAND_BENCH_N: u64 = 1000;

macro_rules! range_float {
    ($fnn:ident, $ty:ty, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = StdRng::new_seeded(1);
            let range = rng.make_range($low, $high);
            b.iter(|| {
                let mut accum: $ty = Default::default();
                for _ in 0..::RAND_BENCH_N {
                    let val = rng.from_range(&range);
                    accum += val;
                }
                accum
            });
        }
    }
}

range_float!(test_range_f32_small, f32, -0.78f32, 10.0/3.0);
range_float!(test_range_f64_small, f64, -0.78f64, 10.0/3.0);
range_float!(test_range_f32_large, f32, 1.2f32, 123_456.1);
range_float!(test_range_f64_large, f64, 1.2f64, 123_456.1);
