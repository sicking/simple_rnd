extern crate num_bigint;
extern crate num_traits;

use std::{ self, f64, f32 };
use super::{ Rng, Rangeable, ZeroOneable };
use RangeImpl;

#[cfg(feature = "exact-floats")]
use self::num_bigint::{ BigUint, BigInt, Sign };
#[cfg(feature = "exact-floats")]
use self::num_traits::{ Signed, Zero, NumCast };
#[cfg(feature = "exact-floats")]
use std::cmp::min;

#[allow(unused_imports)] // rustc incorrectly thinks num_traits::One is unused
use self::num_traits::One;

trait UnsignedInt {
  fn bits(&self) -> usize;
}
impl UnsignedInt for u64 {
  fn bits(&self) -> usize {
    (64 - self.leading_zeros()) as usize
  }
}
impl UnsignedInt for u32 {
  fn bits(&self) -> usize {
    (32 - self.leading_zeros()) as usize
  }
}

fn bitmask<T, S>(bits: S) -> T
  where T: std::ops::Shl<S, Output=T> + std::ops::Sub<T, Output=T> + From<u8> {
  (T::from(1u8) << bits) - T::from(1u8)
}
fn mask<T, S>(val: T, bits: S) -> T
  where T: std::ops::Shl<S, Output=T> + std::ops::Sub<T, Output=T> + std::ops::BitAnd<T, Output=T> + From<u8> {
  val & bitmask(bits)
}

#[cfg(not(feature = "exact-floats"))]
pub struct RangeFloat<T> {
  start: T,
  end: T,
}

#[cfg(feature = "exact-floats")]
pub struct RangeFloat<T> {
  start: BigInt,
  end: BigInt,
  exponent: u16,
  _marker: std::marker::PhantomData<T>,
}

macro_rules! impl_float {
  ($fty: ty, $uty: ty, $mantissa_bits: expr, $exp_bits: expr, $gen_func: ident) => (
    impl Rangeable for $fty {
      type Output = $fty;
      type Range = RangeFloat<$fty>;

      fn zero() -> Self::Output {
        return 0.0
      }
      fn is_neg(&self) -> bool {
        return *self < 0.0;
      }
    }

    impl RangeImpl<$fty> for RangeFloat<$fty> {
      type Output = $fty;

      fn new_below(limit: $fty) -> Self {
        Self::new_range(0.0, limit)
      }

      #[cfg(not(feature = "exact-floats"))]
      fn new_range(start: $fty, end: $fty) -> Self {
        if start >= end {
          panic!("empty or inverted range");
        }
        let start_exp = mask(start.to_bits() >> $mantissa_bits, $exp_bits);
        let end_exp = mask(end.to_bits() >> $mantissa_bits, $exp_bits);
        if start_exp > bitmask::<$uty, _>($exp_bits) - 3 || end_exp > bitmask::<$uty, _>($exp_bits) - 3 {
          panic!("Overflow or NaN");
        }
        RangeFloat { start, end }
      }

      #[cfg(not(feature = "exact-floats"))]
      fn gen<'b, R: Rng + ?Sized>(&self, rng: &'b mut R) -> $fty {
        let scale = self.end - self.start;
        let offset = self.start - scale;
        let exp = bitmask::<$uty, _>($exp_bits - 1) << $mantissa_bits;
        loop {
          let frac = mask(rng.$gen_func(), $mantissa_bits);
          let res = <$fty>::from_bits(frac | exp) * scale + offset;
          // Check for rounding errors
          if res >= self.start && res < self.end {
            return res;
          }
        }
      }

      #[cfg(feature = "exact-floats")]
      fn new_range(start: $fty, end: $fty) -> Self {
        fn parse_float(val: $fty) -> (BigInt, u32) {
          assert!(val.is_finite());
          let bits = val.to_bits();
          let neg = (bits >> ($mantissa_bits + $exp_bits)) == 1;
          let mut exp = mask(bits >> $mantissa_bits, $exp_bits) as u32;
          let mut significand = mask(bits, $mantissa_bits);
          assert!(exp != bitmask($exp_bits));
          if exp > 0 {
            significand |= 1 as $uty << $mantissa_bits;
          } else {
            exp = 1;
          }

          if significand != 0 {
            let zeros = significand.trailing_zeros();
            significand >>= zeros;
            exp += zeros as u32;
          }
          (BigInt::from_biguint(if neg { Sign::Minus } else { Sign::Plus },
                                BigUint::from(significand)),
           exp)
        }

        if start >= end {
          panic!("empty or inverted range");
        }
        if !start.is_finite() || !end.is_finite() {
          panic!("Start or end is NaN or infinite")
        }

        let exponent;
        let (mut start_parsed, start_parsed_exp) = parse_float(start);
        let (mut end_parsed, end_parsed_exp) = parse_float(end);
        if start_parsed_exp > end_parsed_exp {
          start_parsed = &start_parsed << ((start_parsed_exp - end_parsed_exp) as usize);
          exponent = end_parsed_exp as u16;
        } else {
          end_parsed = &end_parsed << ((end_parsed_exp - start_parsed_exp) as usize);
          exponent = start_parsed_exp as u16;
        }
        RangeFloat { start: start_parsed, end: end_parsed, exponent, _marker: std::marker::PhantomData }
      }

      #[cfg(feature = "exact-floats")]
      fn gen<'b, R: Rng + ?Sized>(&self, rng: &'b mut R) -> $fty {
        let mut res_bigint = rng.range(&self.start, &self.end);
        let mut res_exp = self.exponent as $uty;

        let is_neg = res_bigint.is_negative();
        if is_neg {
          res_bigint = -res_bigint;
        }
        let mut res: $uty;
        let mut len = res_bigint.bits();
        const GOAL_LEN: usize = $mantissa_bits + 1;
        if len > GOAL_LEN {
          // Make sure that res_biguint doesn't have too many significan digits
          let mut res_biguint = res_bigint.to_biguint().unwrap();
          drop(res_bigint); // drop to prevent accidental use
          while len > GOAL_LEN {
            let extra = len - GOAL_LEN;
            let round_down = is_neg && (&res_biguint & bitmask::<BigUint, _>(extra)) != Zero::zero();
            res_biguint >>= extra;
            res_exp += extra as $uty;
            assert!(res_exp <= ((1 << $exp_bits) - 2));
            if round_down {
              // Round towards negative infinity.
              res_biguint += 1u32;
            }
            assert!(res_biguint.bits() == GOAL_LEN || (res_biguint.bits() == GOAL_LEN + 1 && round_down));
            len = res_biguint.bits();
          }
          res = NumCast::from(res_biguint).unwrap();
        } else {
          // Make sure that res has enough significant digits
          res = NumCast::from(res_bigint).unwrap();
          while len < GOAL_LEN && res_exp > 1 {
            let additional = min(GOAL_LEN - len, (res_exp - 1) as usize);
            let additional_bits = mask(rng.$gen_func(), additional);
            res <<= additional;
            if !is_neg {
              res |= additional_bits;
            } else {
              res -= additional_bits;
            }
            res_exp -= additional as $uty;
            len = res.bits();
          }

          // 10_0000_0000_0000 is an edgecase. When we subtract from this value we lose one bit
          // at the top and get fewer total bits. This means that we can fit in an extra bit at
          // the end, which if it's a zero will prevent rounding from getting us back to the
          // original value.
          // Make sure not to run this if the orignal bit-length was over 53 bits and we've already
          // rounded to get to this result.
          // Use !rng.chance() to fit better with testing code below, which expects that generating
          // high numbers from the rng, results in a value closer to the end of the range.
          if is_neg && res == (1 as $uty << $mantissa_bits) && res_exp > 1 &&
             !rng.chance(1, 2) {
            res = (2 as $uty << $mantissa_bits) - 1;
            res_exp -= 1;
          }
        }


        // Convert to f64
        assert!(res.bits() == GOAL_LEN || (res_exp == 1 && res.bits() < GOAL_LEN));
        if (res & (1 as $uty << $mantissa_bits)) == 0 {
          assert_eq!(res_exp, 1);
          res_exp = 0;
        }
        res &= !(1 as $uty << $mantissa_bits);
        res |= res_exp << $mantissa_bits;
        if is_neg {
          res |= 1 << ($mantissa_bits + $exp_bits);
        }
        <$fty>::from_bits(res)
      }
    }

    impl ZeroOneable for $fty {
      #[cfg(not(feature = "exact-floats"))]
      fn rng_zeroone<R: Rng + ?Sized>(rng: &mut R) -> $fty {
        let frac = rng.$gen_func() & ((1 << $mantissa_bits) - 1);
        let exp = ((1 as $uty << ($exp_bits - 1)) - 1) << $mantissa_bits;
        <$fty>::from_bits(frac | exp) - 1.0
      }


      #[cfg(feature = "exact-floats")]
      fn rng_zeroone<R: Rng + ?Sized>(rng: &mut R) -> $fty {
        const GOAL_LEN: usize = $mantissa_bits + 1;
        let mut exp = bitmask::<$uty, _>($exp_bits - 1) - 1;
        let mut res = rng.$gen_func() & ((1 as $uty << GOAL_LEN) - 1);
        let mut len = res.bits();
        while len < GOAL_LEN && exp > 1 {
          let additional = min(GOAL_LEN - len, (exp - 1) as usize);
          let additional_bits = mask(rng.$gen_func(), additional);
          res <<= additional;
          res |= additional_bits;
          exp -= additional as $uty;
          len = res.bits();
        }

        assert!(res.bits() == GOAL_LEN || (exp == 1 && res.bits() < GOAL_LEN));
        if (res & (1 as $uty << $mantissa_bits)) == 0 {
          assert_eq!(exp, 1);
          exp = 0;
        }
        res &= !(1 as $uty << $mantissa_bits);
        res |= exp << $mantissa_bits;
        <$fty>::from_bits(res)
      }
    }
  )
}

impl_float!(f64, u64, 52, 11, gen_u64);
impl_float!(f32, u32, 23, 8, gen_u32);

#[cfg(test)]
mod tests {
  use super::*;
  use super::super::StdRng;
  use std::panic::catch_unwind;

  #[test]
  fn test_float() {
    let mut rng = StdRng::new();
    for val in [0.000001f64, 1000000.0, 47.0, f64::from_bits(1), f64::from_bits(0x7fcf_ffff_ffff_ffff)].iter().cloned() {
      let r1 = rng.make_range_below(val);
      let r2 = rng.make_range(-val, val);
      let r3 = if val - 1.0 != val { Some(rng.make_range(val - 1.0, val)) } else { None };
      let r4 = if val - 1.0 != val { Some(rng.make_range(-val - 1.0, -val)) } else { None };
      for _ in 0..1000 {
        let x = rng.below(val);
        assert!(0.0 <= x && x < val);
        let x = rng.range(-val, val);
        assert!(-val <= x && x < val);
        let x = rng.from_range(&r1);
        assert!(0.0 <= x && x < val);
        let x = rng.from_range(&r2);
        assert!(-val <= x && x < val);
        if val - 1.0 != val {
          let x = rng.range(val - 1.0, val);
          assert!(val - 1.0 <= x && x < val);
          let x = rng.range(-val - 1.0, -val);
          assert!(-val - 1.0 <= x && x < -val);
          let x = rng.from_range(r3.as_ref().unwrap());
          assert!(val - 1.0 <= x && x < val);
          let x = rng.from_range(r4.as_ref().unwrap());
          assert!(-val - 1.0 <= x && x < -val);
        }
      }
      let x = rng.below(&val);
      assert!(0.0 <= x && x < val);
      let x = rng.range(&-val, &val);
      assert!(-val <= x && x < val);
    }

    assert!(catch_unwind(|| StdRng::new().below(f64::NAN)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(f64::INFINITY)).is_err());

    for val in [0.000001f32, 1000000.0, 47.0, f32::from_bits(1), f32::from_bits(0x7e7f_ffff)].iter().cloned() {
      let r1 = rng.make_range_below(val);
      let r2 = rng.make_range(-val, val);
      let r3 = if val - 1.0 != val { Some(rng.make_range(val - 1.0, val)) } else { None };
      let r4 = if val - 1.0 != val { Some(rng.make_range(-val - 1.0, -val)) } else { None };
      for _ in 0..1000 {
        let x = rng.below(val);
        assert!(0.0 <= x && x < val);
        let x = rng.range(-val, val);
        assert!(-val <= x && x < val);
        let x = rng.from_range(&r1);
        assert!(0.0 <= x && x < val);
        let x = rng.from_range(&r2);
        assert!(-val <= x && x < val);
        if val - 1.0 != val {
          let x = rng.range(val - 1.0, val);
          assert!(val - 1.0 <= x && x < val);
          let x = rng.range(-val - 1.0, -val);
          assert!(-val - 1.0 <= x && x < -val);
          let x = rng.from_range(r3.as_ref().unwrap());
          assert!(val - 1.0 <= x && x < val);
          let x = rng.from_range(r4.as_ref().unwrap());
          assert!(-val - 1.0 <= x && x < -val);
        }
      }
      let x = rng.below(&val);
      assert!(0.0 <= x && x < val);
      let x = rng.range(&-val, &val);
      assert!(-val <= x && x < val);
    }

    assert!(catch_unwind(|| StdRng::new().below(f32::NAN)).is_err());
    assert!(catch_unwind(|| StdRng::new().below(f32::INFINITY)).is_err());
  }

  #[cfg(feature = "exact-floats")]
  struct TestRng {
    vals: Vec<Option<u64>>,
    repeat: Option<u64>,
  }
  #[cfg(feature = "exact-floats")]
  impl Rng for TestRng {
    fn gen_u64(&mut self) -> u64 { self.test_gen_u64(0) }
    fn test_gen_u32(&mut self, limit: u32) -> u32 { self.test_gen_u64(limit as u64) as u32 }
    fn test_gen_u64(&mut self, limit: u64) -> u64 {
      match (self.repeat, self.vals.pop()) {
        (_, Some(Some(val))) => val,
        (_, Some(None)) => limit.wrapping_sub(1),
        (Some(repeat), None) => repeat,
        (None, None) => limit.wrapping_sub(1),
      }
    }

    fn gen_biguint(&mut self, _bits: usize) -> BigUint { panic!(""); }

    fn test_gen_biguint(&mut self, _bits: usize, limit: &BigUint) -> BigUint {
      match (self.repeat, self.vals.pop()) {
        (_, Some(Some(val))) => BigUint::from(val),
        (_, Some(None)) => limit - BigUint::one(),
        (Some(repeat), None) => BigUint::from(repeat),
        (None, None) => limit - BigUint::one(),
      }
    }
  }

  #[test]
  #[cfg(feature = "exact-floats")]
  fn test_exact_float() {
    let mut vals: Vec<f64> =
      [0i64,
       0x0000_0f00_0000_0000,
       0x0001_0000_0000_0000,
       0x0004_0000_0000_0000,
       0x0008_0000_0000_0000,
       0x0010_0000_0000_0000,
       0x0020_0000_0000_0000,
       0x0040_0000_0000_0000,
       0x0100_0000_0000_0000,
       0x00cd_ef12_3456_789a,
       0x0100_ffff_ffff_ffff,
       0x010f_ffff_ffff_ffff,
       0x0400_1234_5678_abcd,
       0x7fef_ffff_ffff_ffff,
       ].iter().cloned()
        .flat_map(|x| (-2i64..3i64).map(move |y| x + y))
        .map(|x| f64::from_bits(x as u64))
        .flat_map(|x| vec![x, -x].into_iter())
        .filter(|x| x.is_finite())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup();

    for a in vals.iter().cloned() {
      for b in vals.iter().cloned().filter(|&b| b > a) {

        let mut rng = TestRng { vals: vec![], repeat: Some(0) };
        assert_eq!(rng.range(a, b), a);

        let mut rng = TestRng { vals: vec![Some(1)], repeat: Some(0) };
        let res = rng.range(a, b);
        assert!(a <= res && res < b);

        let mut rng = TestRng { vals: vec![], repeat: None };
        let res = rng.range(a, b);
        if b > 0.0 {
          assert_eq!(res, f64::from_bits(b.to_bits() - 1));
        } else if b < 0.0 {
          assert_eq!(res, f64::from_bits(b.to_bits() + 1));
        } else {
          assert_eq!(res, f64::from_bits(0x8000_0000_0000_0001));
        }
      }
    }

    let mut vals: Vec<f32> =
      [0i32,
       0x000f_0000,
       0x0008_0000,
       0x0020_0000,
       0x0040_0000,
       0x0080_0000,
       0x0100_0000,
       0x0200_0000,
       0x0800_0000,
       0x5678_abcd,
       0x0807_ffff,
       0x087f_ffff,
       0x4012_3456,
       0x7f7f_ffff,
       ].iter().cloned()
        .flat_map(|x| (-2i32..3i32).map(move |y| x + y))
        .map(|x| f32::from_bits(x as u32))
        .flat_map(|x| vec![x, -x].into_iter())
        .filter(|x| x.is_finite())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup();

    for a in vals.iter().cloned() {
      for b in vals.iter().cloned().filter(|&b| b > a) {

        let mut rng = TestRng { vals: vec![], repeat: Some(0) };
        let res = rng.range(a, b);
        assert_eq!(res, a);

        let mut rng = TestRng { vals: vec![], repeat: Some(0) };
        assert_eq!(rng.range(a, b), a);

        let mut rng = TestRng { vals: vec![], repeat: None };
        let res = rng.range(a, b);
        if b > 0.0 {
          assert_eq!(res, f32::from_bits(b.to_bits() - 1));
        } else if b < 0.0 {
          assert_eq!(res, f32::from_bits(b.to_bits() + 1));
        } else {
          assert_eq!(res, f32::from_bits(0x8000_0001));
        }
      }
    }
  }

  #[test]
  #[cfg(feature = "exact-floats")]
  fn test_exact_zeroone() {
    let mut rng = TestRng { vals: vec![], repeat: Some(0) };
    assert_eq!(rng.zeroone::<f32>(), 0.0);
    let mut rng = TestRng { vals: vec![], repeat: Some(0xffff_ffff_ffff_ffff) };
    let res: f32 = rng.zeroone();
    assert!(0.9999 < res && res < 1.0);
    assert_eq!(res, f32::from_bits(0x3f7f_ffff));
    let res: f64 = rng.zeroone();
    assert!(0.9999 < res && res < 1.0);
    assert_eq!(res, f64::from_bits(0x3fef_ffff_ffff_ffff));

    let mut rng = TestRng { vals: vec![Some(1), Some(0)], repeat: Some(0) };
    assert_eq!(rng.zeroone::<f64>().to_bits(), 0x3950_0000_0000_0000);
    let mut rng = TestRng { vals: vec![Some(1), Some(0)], repeat: Some(0) };
    assert_eq!(rng.zeroone::<f32>().to_bits(), 0x2780_0000);
  }
}

