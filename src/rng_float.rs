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

    let mut signed_res = rng.range(&start_parsed, &end_parsed);

    let is_neg = signed_res.is_negative();
    if is_neg {
      signed_res = -signed_res;
    }
    let mut res = signed_res.to_biguint().unwrap();

    let mut len = res.bits();
    if len > 53 {
      // Make sure that res doesn't have too many significan digits
      while len > 53 {
        let extra = len - 53;
        let round_down = is_neg && (&res & ((BigUint::from(1u32) << extra) - 1u32)) != Zero::zero();
        res >>= extra;
        res_exp += extra;
        assert!(res_exp <= 2046);
        if round_down {
          // Round towards negative infinity.
          res += 1u32;
        }
        assert!(res.bits() == 53 || (res.bits() == 54 && round_down));
        len = res.bits();
      }
    } else {
      // Make sure that res has enough significant digits
      while len < 53 && res_exp > 1 {
        let additional = min(53 - len, res_exp - 1);
        let additional_bits = rng.gen_biguint(additional);
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
      // rounded to get to this result
      if is_neg && res == BigUint::from(0x0010_0000_0000_0000u64) && res_exp > 1 &&
          rng.gen_biguint(1) == BigUint::from(1u32) {
        res = BigUint::from(0x001f_ffff_ffff_ffffu64);
        res_exp -= 1;
      }
    }


    // Convert to u64 and then f64
    assert!(res.bits() == 53 || (res_exp == 1 && res.bits() < 53));
    let mut bits = res.to_u64().unwrap();
    if (bits & (1 << 52)) == 0 {
      assert_eq!(res_exp, 1);
      res_exp = 0;
    }
    bits &= !(1 << 52);
    bits |= (res_exp as u64) << 52;
    if is_neg {
      bits |= 1 << 63;
    }
    f64::from_bits(bits)
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
    bigints: Vec<Option<BigUint>>,
    bigint_repeat: Option<BigUint>,
  }
  impl Rng for TestRng {
    fn gen_u32(&mut self) -> u32 { panic!("") }
    fn gen_u64(&mut self) -> u64 { panic!("") }

    fn gen_biguint(&mut self, bits: usize) -> BigUint {
      self.test_gen_biguint(bits, &(BigUint::one() << bits))
    }

    fn test_gen_biguint(&mut self, _bits: usize, limit: &BigUint) -> BigUint {
      match (self.bigint_repeat.clone(), self.bigints.pop().clone()) {
        (_, Some(Some(bigint))) => bigint,
        (_, Some(None)) => limit - BigUint::one(),
        (Some(repeat), None) => repeat.clone(),
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

        let mut rng = TestRng { bigints: vec![], bigint_repeat: Some(Zero::zero()) };
        assert_eq!(rng.range(a, b), a);

        let mut rng = TestRng { bigints: vec![Some(BigUint::one())], bigint_repeat: Some(Zero::zero()) };
        let res = rng.range(a, b);
        assert!(a <= res && res < b);

        let mut rng = TestRng { bigints: vec![], bigint_repeat: None };
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

