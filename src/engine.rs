use std::{
    cell::RefCell,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    rc::Rc,
};

use inner::ValInner;
use num_traits::Pow;

mod inner;

#[derive(Debug, Clone)]
pub struct Val {
    pub inner: Rc<RefCell<ValInner>>,
}

impl Val {
    pub fn new(data: f32) -> Self {
        Self {
            inner: ValInner::rc(data, None),
        }
    }

    pub fn grad(&self) -> f32 {
        self.inner.as_ref().borrow().grad
    }

    pub fn add_data(&self, grad: f32) {
        self.inner.as_ref().borrow_mut().data += grad;
    }

    pub fn data(&self) -> f32 {
        self.inner.as_ref().borrow().data
    }

    pub fn tanh(&self) -> Self {
        Self {
            inner: ValInner::tanh(self.inner.clone()),
        }
    }

    pub fn relu(&self) -> Self {
        Self {
            inner: ValInner::relu(self.inner.clone()),
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            inner: ValInner::exp(self.inner.clone()),
        }
    }

    pub fn backward(&self) {
        self.inner.as_ref().borrow_mut().backward();
    }

    pub fn zero_grad(&self) {
        self.inner.as_ref().borrow_mut().grad = 0.0;
    }
}

impl<T> From<T> for Val
where
    T: Into<f32>,
{
    fn from(value: T) -> Self {
        Self::new(value.into())
    }
}

impl From<&Val> for f32 {
    fn from(val: &Val) -> f32 {
        val.data()
    }
}

impl<T: Into<Val>> Add<T> for Val {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        Self {
            inner: ValInner::add(self.inner, rhs.inner),
        }
    }
}

impl<T: Into<Val>> Add<T> for &Val {
    type Output = Val;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        Val {
            inner: ValInner::add(Rc::clone(&self.inner), rhs.inner),
        }
    }
}

impl Add<&Val> for f32 {
    type Output = Val;

    fn add(self, rhs: &Val) -> Self::Output {
        let lhs: Val = self.into();
        lhs + rhs.clone()
    }
}

impl<T: Into<Val>> Sub<T> for Val {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self {
            inner: ValInner::sub(self.inner, rhs.into().inner),
        }
    }
}

impl<T: Into<Val>> Sub<T> for &Val {
    type Output = Val;

    fn sub(self, rhs: T) -> Self::Output {
        Val {
            inner: ValInner::sub(Rc::clone(&self.inner), rhs.into().inner),
        }
    }
}

impl Sub<Val> for f32 {
    type Output = Val;

    fn sub(self, rhs: Val) -> Self::Output {
        let lhs: Val = self.into();
        lhs - rhs
    }
}

impl<T: Into<Val>> Mul<T> for Val {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            inner: ValInner::mul(self.inner, rhs.into().inner),
        }
    }
}

impl Mul for &Val {
    type Output = Val;

    fn mul(self, rhs: Self) -> Self::Output {
        Val {
            inner: ValInner::mul(self.inner.clone(), rhs.inner.clone()),
        }
    }
}

impl Mul<Val> for f32 {
    type Output = Val;

    fn mul(self, rhs: Val) -> Self::Output {
        let lhs: Val = self.into();
        lhs * rhs
    }
}

impl Mul<&Val> for f32 {
    type Output = Val;

    fn mul(self, rhs: &Val) -> Self::Output {
        let lhs: Val = self.into();
        lhs * rhs.clone()
    }
}

impl Mul<f32> for &Val {
    type Output = Val;

    fn mul(self, rhs: f32) -> Self::Output {
        Val {
            inner: ValInner::mul(self.inner.clone(), ValInner::rc(rhs, None)),
        }
    }
}

impl<T: Into<Val>> Div<T> for Val {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {
            inner: ValInner::div(self.inner, rhs.into().inner),
        }
    }
}

impl<T: Into<Val>> Div<T> for &Val {
    type Output = Val;

    fn div(self, rhs: T) -> Self::Output {
        Val {
            inner: ValInner::div(self.inner.clone(), rhs.into().inner.clone()),
        }
    }
}

impl Div<Val> for f32 {
    type Output = Val;

    fn div(self, rhs: Val) -> Self::Output {
        let lhs: Val = self.into();
        lhs / rhs
    }
}

impl Neg for Val {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            inner: ValInner::neg(self.inner),
        }
    }
}

impl Neg for &Val {
    type Output = Val;

    fn neg(self) -> Self::Output {
        Val {
            inner: ValInner::neg(self.inner.clone()),
        }
    }
}

impl<T: Into<f32>> Pow<T> for Val {
    type Output = Self;

    fn pow(self, exponent: T) -> Self::Output {
        Val {
            inner: ValInner::pow(self.inner, exponent.into()),
        }
    }
}

impl<T: Into<f32>> Pow<T> for &Val {
    type Output = Val;

    fn pow(self, exponent: T) -> Self::Output {
        Val {
            inner: ValInner::pow(self.inner.clone(), exponent.into()),
        }
    }
}

impl<T: Into<Val>> AddAssign<T> for Val {
    fn add_assign(&mut self, rhs: T) {
        self.add_data(rhs.into().data());
    }
}

impl Add<Val> for f32 {
    type Output = Val;

    fn add(self, rhs: Val) -> Self::Output {
        Val {
            inner: ValInner::add(ValInner::rc(self, None), rhs.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression() {
        let a = Val::new(2.0);
        let b = Val::new(2.0);
        // b(2.0, 3.0)
        //              + c(4.0, 3.0)
        // a(2.0, 3.0)                  *  x(12.0, 1.0)
        //                _(3.0, 4.0)
        let c = b.clone() + a.clone();
        let x = c.clone() * 3.0;

        x.backward();

        println!("{:?}", x);
        assert_eq!(12.0, x.inner.as_ref().borrow().data);
        assert_eq!(1.0, x.inner.as_ref().borrow().grad);
        assert_eq!(4.0, c.inner.as_ref().borrow().data);
        assert_eq!(3.0, c.inner.as_ref().borrow().grad);
        assert_eq!(2.0, a.inner.as_ref().borrow().data);
        assert_eq!(3.0, a.inner.as_ref().borrow().grad);
        assert_eq!(2.0, b.inner.as_ref().borrow().data);
        assert_eq!(3.0, b.inner.as_ref().borrow().grad);
    }

    #[test]
    fn test_add_itself() {
        let a = Val::new(3.0);
        let mut b = &a * &a;
        println!("B: {:?}", b);
        b += &a;
        println!("B: {:?}", b);
        b.backward();
        assert_eq!(12.0, b.data());
    }

    #[test]
    fn test_micrograd_example() {
        let a = Val::new(-4.0);
        let b = Val::new(2.0);
        let mut c = a.clone() + b.clone();
        let mut d = a.clone() * b.clone() + (b.clone()).pow(3.0);
        c += c.clone() + 1.0;
        c += 1.0 + c.clone() + -(a.clone());
        d += d.clone() * 2.0 + (b.clone() + a.clone()).relu();
        d += 3.0 * d.clone() + (b.clone() - a.clone()).relu();
        let e = c - d;
        let f = e.pow(2.0);
        let mut g = f.clone() / 2.0;
        g += 10.0 / f;
        assert_eq!(24.0, g.data().floor());
        g.backward();
        // assert_eq!(139.0, a.grad().floor());
        // assert_eq!(646.0, b.grad().floor());
    }
}

