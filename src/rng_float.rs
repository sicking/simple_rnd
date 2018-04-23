extern crate num_bigint;
extern crate num_traits;

use self::num_bigint::{ BigUint, BigInt, Sign };
use self::num_traits::{ Signed, ToPrimitive, Zero };
use super::{ Rng, Rangeable };
use std::cmp::min;

#[allow(unused_imports)] // rustc incorrectly thinks num_traits::One is unused
use self::num_traits::One;


fn parse_float(val: f64) -> (BigInt, usize) {
  assert!(val.is_finite());
  let bits = val.to_bits();
  let neg = (bits >> 63) == 1;
  let mut exp = ((bits >> 52) & 0x7ff) as usize;
  let mut significand = bits & 0x000f_ffff_ffff_ffff;
  assert!(exp != 0x7ff);
  if exp > 0 {
    significand |= 0x0010_0000_0000_0000;
  } else {
    exp = 1;
  }

  if significand != 0 {
    let zeros = significand.trailing_zeros();
    significand >>= zeros;
    exp += zeros as usize;
  }
  (BigInt::from_biguint(if neg { Sign::Minus } else { Sign::Plus },
                        BigUint::from(significand)),
   exp)
}

trait UnsignedInt {
  fn bits(&self) -> usize;
}
impl UnsignedInt for u64 {
  fn bits(&self) -> usize {
    (64 - self.leading_zeros()) as usize
  }
}

impl Rangeable for f64 {
  type Output = f64;

  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: f64) -> f64 {
    Self::rng_range(rng, 0.0, limit)
  }

  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: f64, end: f64) -> f64 {
    if start >= end {
      panic!("empty or inverted range");
    }
    if !start.is_finite() || !end.is_finite() {
      panic!("Start or end is NaN or infinite")
    }

    let mut res_exp;
    let (mut start_parsed, start_parsed_exp) = parse_float(start);
    let (mut end_parsed, end_parsed_exp) = parse_float(end);
    if start_parsed_exp > end_parsed_exp {
      start_parsed = &start_parsed << (start_parsed_exp - end_parsed_exp);
      res_exp = end_parsed_exp;
    } else {
      end_parsed = &end_parsed << (end_parsed_exp - start_parsed_exp);
      res_exp = start_parsed_exp;
    }

    let mut res_bigint = rng.range(&start_parsed, &end_parsed);

    let is_neg = res_bigint.is_negative();
    if is_neg {
      res_bigint = -res_bigint;
    }
    let mut res: u64;
    let mut len = res_bigint.bits();
    if len > 53 {
      // Make sure that res_biguint doesn't have too many significan digits
      let mut res_biguint = res_bigint.to_biguint().unwrap();
      drop(res_bigint); // drop to prevent accidental use
      while len > 53 {
        let extra = len - 53;
        let round_down = is_neg && (&res_biguint & ((BigUint::from(1u32) << extra) - 1u32)) != Zero::zero();
        res_biguint >>= extra;
        res_exp += extra;
        assert!(res_exp <= 2046);
        if round_down {
          // Round towards negative infinity.
          res_biguint += 1u32;
        }
        assert!(res_biguint.bits() == 53 || (res_biguint.bits() == 54 && round_down));
        len = res_biguint.bits();
      }
      res = res_biguint.to_u64().unwrap();
    } else {
      // Make sure that res has enough significant digits
      res = res_bigint.to_u64().unwrap();
      while len < 53 && res_exp > 1 {
        let additional = min(53 - len, res_exp - 1);
        let additional_bits = rng.gen_u64() & ((1u64 << additional) - 1);
        res <<= additional;
        if !is_neg {
          res |= additional_bits;
        } else {
          res -= additional_bits;
        }
        res_exp -= additional;
        len = res.bits();
      }

      // Edgecase case is 10_0000_0000_0000. When we subtract from this value we lose one bit
      // at the top and get fewer total bits. This means that we can fit in an extra bit at
      // the end, which if it's a zero will prevent rounding from getting us back to the
      // original value.
      // Make sure not to run this if the orignal bit-length was over 53 bits and we've already
      // rounded to get to this result.
      // Use !rng.chance() to fit better with testing code below, which expects that generating
      // high numbers from the rng, results in a value closer to the end of the range.
      if is_neg && res == 0x0010_0000_0000_0000u64 && res_exp > 1 &&
         !rng.chance(1, 2) {
        res = 0x001f_ffff_ffff_ffffu64;
        res_exp -= 1;
      }
    }


    // Convert to u64 and then f64
    assert!(res.bits() == 53 || (res_exp == 1 && res.bits() < 53));
    if (res & (1 << 52)) == 0 {
      assert_eq!(res_exp, 1);
      res_exp = 0;
    }
    res &= !(1 << 52);
    res |= (res_exp as u64) << 52;
    if is_neg {
      res |= 1 << 63;
    }
    f64::from_bits(res)
  }

  fn zero() -> Self::Output {
    return 0.0
  }

  fn is_neg(&self) -> bool {
    return *self < 0.0;
  }
}

impl<'a> Rangeable for &'a f64 {
  type Output = f64;

  fn rng_below<R: Rng + ?Sized>(rng: &mut R, limit: &'a f64) -> f64 {
    <f64>::rng_below(rng, *limit)
  }

  fn rng_range<R: Rng + ?Sized>(rng: &mut R, start: &'a f64, end: &'a f64) -> f64 {
    <f64>::rng_range(rng, *start, *end)
  }

  fn zero() -> Self::Output {
    return 0.0;
  }

  fn is_neg(&self) -> bool {
    return **self < 0.0;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  struct TestRng {
    vals: Vec<Option<u64>>,
    repeat: Option<u64>,
  }
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
  }
}

