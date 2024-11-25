use std::{
    cell::RefCell,
    fmt::Display,
    rc::Rc,
};

use num_traits::Pow;

#[derive(Debug, PartialEq)]
pub struct ValInner {
    pub data: f32,
    pub grad: f32,
    pub op: Option<Op>,
}

impl ValInner {
    fn new<T>(data: T) -> Self
    where
        T: Into<f32>,
    {
        Self {
            data: data.into(),
            grad: 0.0,
            op: None,
        }
    }

    pub fn rc<T>(data: T, op: Option<Op>) -> Rc<RefCell<Self>>
    where
        T: Into<ValInner>,
    {
        let mut data = data.into();
        data.op = op;
        Rc::new(RefCell::new(data))
    }

    pub fn add(left: Rc<RefCell<Self>>, right: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let data = left.as_ref().borrow().data + right.as_ref().borrow().data;
        Self::rc(data, Some(Op::Add { left, right }))
    }

    pub fn neg(value: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        Self::mul(value, Self::rc(-1.0, None))
    }

    pub fn sub(left: Rc<RefCell<Self>>, right: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        Self::add(left, Self::neg(right))
    }

    pub fn mul(left: Rc<RefCell<Self>>, right: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let data = left.as_ref().borrow().data * right.as_ref().borrow().data;
        Self::rc(data, Some(Op::Mul { left, right }))
    }

    pub fn div(left: Rc<RefCell<Self>>, right: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        Self::mul(left, Self::pow(right, -1.0))
    }

    pub fn pow(base: Rc<RefCell<Self>>, exponent: f32) -> Rc<RefCell<Self>> {
        let data = base.as_ref().borrow().data.pow(exponent);
        Self::rc(data, Some(Op::Pow { base, exponent }))
    }

    pub fn tanh(value: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let tanh = value.as_ref().borrow().data.tanh();
        Self::rc(tanh, Some(Op::Tanh { val: value, tanh }))
    }

    pub fn relu(value: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let data = value.as_ref().borrow().data;
        let data = if data < 0.0 { 0.0 } else { data };
        Self::rc(
            data,
            Some(Op::ReLU {
                relu: data,
                prev: value,
            }),
        )
    }

    pub fn exp(value: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let exp = value.as_ref().borrow().data.exp();
        Self::rc(exp, Some(Op::Exp { val: value, exp }))
    }

    pub fn backward(&mut self) {
        self.grad = 1.0;
        self.backward_inner();
    }

    fn backward_inner(&mut self) {
        if let Some(op) = &mut self.op {
            op.backward(self.grad);
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Op {
    Add {
        left: Rc<RefCell<ValInner>>,
        right: Rc<RefCell<ValInner>>,
    },
    Mul {
        left: Rc<RefCell<ValInner>>,
        right: Rc<RefCell<ValInner>>,
    },
    Tanh {
        val: Rc<RefCell<ValInner>>,
        tanh: f32,
    },
    Exp {
        val: Rc<RefCell<ValInner>>,
        exp: f32,
    },
    Pow {
        base: Rc<RefCell<ValInner>>,
        exponent: f32,
    },
    ReLU {
        relu: f32,
        prev: Rc<RefCell<ValInner>>,
    },
}

impl Op {
    pub fn backward(&mut self, grad: f32) {
        match self {
            Op::Add { left, right } => {
                if Rc::ptr_eq(left, right) {
                    let mut value = left.as_ref().borrow_mut();
                    value.grad += grad;
                    value.grad += grad;
                    value.backward_inner();
                    value.backward_inner();
                } else {
                    let mut left_ref = left.as_ref().borrow_mut();
                    let mut right_ref = right.as_ref().borrow_mut();
                    left_ref.grad += grad;
                    right_ref.grad += grad;
                    left_ref.backward_inner();
                    right_ref.backward_inner();
                };
            }
            Op::Mul { left, right } => {
                if Rc::ptr_eq(left, right) {
                    let mut value = left.as_ref().borrow_mut();
                    value.grad += 2.0 * value.data * grad;
                    value.backward_inner();
                    value.backward_inner();
                } else {
                    let mut left_ref = left.as_ref().borrow_mut();
                    let mut right_ref = right.as_ref().borrow_mut();
                    left_ref.grad += right_ref.data * grad;
                    right_ref.grad += left_ref.data * grad;
                    left_ref.backward_inner();
                    right_ref.backward_inner();
                }
            }
            Op::Tanh { val, tanh } => {
                let mut value = val.as_ref().borrow_mut();
                value.grad += (1.0 - tanh.powi(2)) * grad;
                value.backward_inner();
            }
            Op::Exp { val, exp } => {
                let mut value = val.as_ref().borrow_mut();
                value.grad += *exp * grad;
                value.backward_inner();
            }
            Op::Pow { base, exponent } => {
                let mut value = base.as_ref().borrow_mut();
                value.grad += *exponent * (value.data.pow(*exponent - 1.0)) * grad;
                value.backward_inner();
            }
            Op::ReLU { relu, prev } => {
                let mut value = prev.as_ref().borrow_mut();
                value.grad += *relu * grad;
                value.backward_inner();
            }
        }
    }
}

impl From<f32> for ValInner {
    fn from(data: f32) -> Self {
        ValInner::new(data)
    }
}

impl Display for ValInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ValInner(data: {}, grad: {})", self.data, self.grad)
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add { left, right } => write!(
                f,
                "{} + {}",
                left.as_ref().borrow(),
                right.as_ref().borrow()
            ),
            Op::Mul { left, right } => write!(
                f,
                "{} * {}",
                left.as_ref().borrow(),
                right.as_ref().borrow()
            ),
            Op::Pow { base, exponent } => {
                write!(f, "{} ** {}", base.as_ref().borrow_mut().data, exponent)
            }
            Op::Tanh { val, tanh: _ } => write!(f, "tanh({})", val.as_ref().borrow().data),
            Op::Exp { val, exp: _ } => write!(f, "e ** {}", val.as_ref().borrow().data),
            Op::ReLU { relu, prev: _ } => write!(f, "{}", relu),
        }
    }
}
