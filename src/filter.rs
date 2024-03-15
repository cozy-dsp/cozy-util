#![allow(clippy::inline_always)]

//use nih_plug::debug::nih_debug_assert;
use num_complex::Complex32;
use std::f32::consts::{self, TAU};
use std::ops::{Add, Mul, Sub};
use std::simd::f32x2;

/// A simple biquad filter with functions for generating coefficients for an all-pass filter.
/// Stolen from NIH-Plug examples
///
/// Based on <https://en.wikipedia.org/wiki/Digital_biquad_filter#Transposed_direct_forms>.
///
/// The type parameter T  should be either an `f32` or a SIMD type.
#[derive(Clone, Copy, Debug)]
pub struct Biquad<T> {
    pub frequency: f32,
    pub coefficients: BiquadCoefficients<T>,
    s1: T,
    s2: T,
}

/// The coefficients `[b0, b1, b2, a1, a2]` for [`Biquad`]. These coefficients are all
/// prenormalized, i.e. they have been divided by `a0`.
///
/// The type parameter T  should be either an `f32` or a SIMD type.
#[derive(Clone, Copy, Debug)]
pub struct BiquadCoefficients<T> {
    sample_rate: f32,
    b0: T,
    b1: T,
    b2: T,
    a1: T,
    a2: T,
}

/// Either an `f32` or some SIMD vector type of `f32`s that can be used with our biquads.
pub trait SimdType:
    Mul<Output = Self> + Sub<Output = Self> + Add<Output = Self> + Copy + Sized
{
    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl<T: SimdType> Default for Biquad<T> {
    /// Before setting constants the filter should just act as an identity function.
    fn default() -> Self {
        Self {
            frequency: 0.0,
            coefficients: BiquadCoefficients::identity(),
            s1: T::from_f32(0.0),
            s2: T::from_f32(0.0),
        }
    }
}

impl<T: SimdType> Biquad<T> {
    /// Process a single sample.
    pub fn process(&mut self, sample: T) -> T {
        let result = self.coefficients.b0 * sample + self.s1;

        self.s1 = self.coefficients.b1 * sample - self.coefficients.a1 * result + self.s2;
        self.s2 = self.coefficients.b2 * sample - self.coefficients.a2 * result;

        result
    }
}

impl<T: SimdType> Default for BiquadCoefficients<T> {
    /// Before setting constants the filter should just act as an identity function.
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: SimdType> BiquadCoefficients<T> {
    /// Convert scalar coefficients into the correct vector type.
    pub fn from_f32s(scalar: BiquadCoefficients<f32>) -> Self {
        Self {
            sample_rate: scalar.sample_rate,
            b0: T::from_f32(scalar.b0),
            b1: T::from_f32(scalar.b1),
            b2: T::from_f32(scalar.b2),
            a1: T::from_f32(scalar.a1),
            a2: T::from_f32(scalar.a2),
        }
    }

    /// Filter coefficients that would cause the sound to be passed through as is.
    pub fn identity() -> Self {
        Self::from_f32s(BiquadCoefficients {
            sample_rate: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        })
    }

    /// Compute the coefficients for a lowpass filter.
    ///
    /// Based on <http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html>.
    pub fn lowpass(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = ((1.0 - cos_omega0) / 2.0) / a0;
        let b1 = (1.0 - cos_omega0) / a0;
        let b2 = b0;
        let a1 = (-2.0 * cos_omega0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    /// Compute the coefficients for a hipass filter.
    ///
    /// Based on <http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html>.
    pub fn hipass(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = ((1.0 - cos_omega0) / 2.0) / a0;
        let b1 = -(1.0 + cos_omega0) / a0;
        let b2 = b0;
        let a1 = (-2.0 * cos_omega0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    /// Compute the coefficients for a bandpass filter.
    /// The skirt gain is constant, the peak gain is Q
    ///
    /// Based on <http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html>.
    pub fn bandpass_peak(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = (q * alpha) / a0;
        let b1 = 0.;
        let b2 = (-q * alpha) / a0;
        let a1 = (-2.0 * cos_omega0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    pub fn bandpass(sample_rate: f32, frequency: f32, band_width: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 * ((2.0_f32.ln() / 2.0) * band_width * (omega0 / sin_omega0)).sinh();

        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        let b1 = 0.0;
        let b2 = -alpha / a0;
        let a1 = (-2.0 * cos_omega0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    pub fn notch(sample_rate: f32, frequency: f32, band_width: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 * ((2.0_f32.ln() / 2.0) * band_width * (omega0 / sin_omega0)).sinh();

        let a0 = 1.0 + alpha;
        let b0 = a0.recip();
        let b1 = (-2.0 * cos_omega0) / a0;
        let b2 = a0.recip();
        let a1 = (-2.0 * cos_omega0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    pub fn allpass(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = (1.0 - alpha) / a0;
        let b1 = (-2.0 * cos_omega0) / a0;
        // b2 is 1.0 + alpha, so we can optimize that away to 1
        let b2 = 1.0;
        let a1 = b1;
        let a2 = b0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    /// Compute the coefficients for a peaking bell filter.
    ///
    /// Based on <http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html>.
    pub fn peaking_eq(sample_rate: f32, frequency: f32, db_gain: f32, q: f32) -> Self {
        //nih_debug_assert!(sample_rate > 0.0);
        //nih_debug_assert!(frequency > 0.0);
        //nih_debug_assert!(frequency < sample_rate / 2.0);
        //nih_debug_assert!(q > 0.0);

        let a = 10_f32.powf(db_gain / 40.0);
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = sin_omega0 / (2.0 * q);

        // We'll prenormalize everything with a0
        let a0 = 1.0 + alpha / a;
        let b0 = alpha.mul_add(a, 1.0) / a0;
        let b1 = (-2.0 * cos_omega0) / a0;
        let b2 = alpha.mul_add(-a, 1.0) / a0;
        let a1 = (-2.0 * cos_omega0) / a0;
        let a2 = (1.0 - alpha / a) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    pub fn low_shelf(sample_rate: f32, frequency: f32, db_gain: f32, slope: f32) -> Self {
        let a = 10_f32.powf(db_gain / 40.0);
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = (sin_omega0 / 2.0) * ((a + a.recip()) * (slope.recip() - 1.0) + 2.0).sqrt();

        let a_plus_one = a + 1.0;
        let a_minus_one = a - 1.0;
        let sqrt_a = a.sqrt();

        let a0 = a_plus_one + a_minus_one * cos_omega0 + 2.0 * sqrt_a * alpha;
        let b0 = (a * (a_plus_one - a_minus_one * cos_omega0 + 2.0 * sqrt_a * alpha)) / a0;
        let b1 = (2.0 * a * (a_minus_one - a_plus_one * cos_omega0)) / a0;
        let b2 = (a * (a_plus_one - a_minus_one * cos_omega0 - 2.0 * sqrt_a * alpha)) / a0;
        let a1 = (-2.0 * (a_minus_one + a_plus_one * cos_omega0)) / a0;
        let a2 = (a_plus_one + a_minus_one * cos_omega0 - 2.0 * sqrt_a * alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    pub fn high_shelf(sample_rate: f32, frequency: f32, db_gain: f32, slope: f32) -> Self {
        let a = 10_f32.powf(db_gain / 40.0);
        let omega0 = consts::TAU * (frequency / sample_rate);
        let (sin_omega0, cos_omega0) = omega0.sin_cos();
        let alpha = (sin_omega0 / 2.0) * ((a + a.recip()) * (slope.recip() - 1.0) + 2.0).sqrt();

        let a_plus_one = a + 1.0;
        let a_minus_one = a - 1.0;
        let sqrt_a = a.sqrt();

        let a0 = a_plus_one - a_minus_one * cos_omega0 + 2.0 * sqrt_a * alpha;
        let b0 = (a * (a_plus_one + a_minus_one * cos_omega0 + 2.0 * sqrt_a * alpha)) / a0;
        let b1 = (-2.0 * a * (a_minus_one + a_plus_one * cos_omega0)) / a0;
        let b2 = (a * (a_plus_one + a_minus_one * cos_omega0 - 2.0 * sqrt_a * alpha)) / a0;
        let a1 = (2.0 * (a_minus_one - a_plus_one * cos_omega0)) / a0;
        let a2 = (a_plus_one - a_minus_one * cos_omega0 - 2.0 * sqrt_a * alpha) / a0;

        Self::from_f32s(BiquadCoefficients { sample_rate, b0, b1, b2, a1, a2 })
    }

    /// Get the frequency response of this filter at the given frequency.
    /// The complex output gives the amplitude in the real component, and the phase offset in the imaginary component.
    /// If you want just the scalar response, use [`Complex32::norm`]
    pub fn frequency_response(&self, freq: f32) -> Complex32 {
        self.transfer_function((Complex32::i() * TAU * (freq / self.sample_rate)).exp())
    }

    /// The transfer function for the biquad. No, I don't know how it works.
    pub fn transfer_function(&self, z: Complex32) -> Complex32 {
        let pow1 = z.powi(-1);
        let pow2 = z.powi(-2);
        (self.b0.to_f32() + self.b1.to_f32() * pow1 + self.b2.to_f32() * pow2)
            / (1.0 + self.a1.to_f32() * pow1 + self.a2.to_f32() * pow2)
    }
}

impl SimdType for f32 {
    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

impl SimdType for f32x2 {
    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        Self::splat(value)
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.as_array()[0]
    }
}
