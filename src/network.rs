use rand::Rng;

#[derive(Clone, Debug)]
pub struct Matrix {
    x: usize,
    y: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn new(x: usize, y: usize, data: Vec<f32>) -> Self {
        Self { x, y, data }
    }

    fn zeros(x: usize, y: usize) -> Self {
        Self {
            x,
            y,
            data: vec![0.0; x * y],
        }
    }

    fn random(x: usize, y: usize, l: f32, u: f32) -> Self {
        let mut r = rand::thread_rng();
        Self {
            x,
            y,
            data: (0..x * y).map(|_| r.gen_range(l..=u)).collect(),
        }
    }

    fn matadd(&self, other: &Self) -> Self {
        assert!(other.y == 1 || other.y == self.y);
        assert_eq!(other.x, self.x);
        let mut out = Self::zeros(self.x, self.y);

        for i in 0..self.y {
            for j in 0..self.x {
                let b = if other.y == 1 {
                    other.data[j]
                } else {
                    other.data[i * self.x + j]
                };

                out.data[i * self.x + j] = self.data[i * self.x + j] + b;
            }
        }
        out
    }

    fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.x, other.y);
        let mut out = Self::zeros(other.x, self.y);

        for i in 0..self.y {
            for k in 0..self.x {
                let a = self.data[i * self.x + k];

                for j in 0..out.x {
                    out.data[i * out.x + j] += a * other.data[k * other.x + j];
                }
            }
        }
        out
    }

    fn relu(&self) -> Self {
        let mut out = self.clone();
        for val in out.data.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }
        out
    }

    fn tanh(&self) -> Self {
        let mut out = self.clone();
        for val in out.data.iter_mut() {
            *val = val.tanh();
        }
        out
    }
}

#[derive(Debug)]
enum Layer {
    Linear {
        ws: Matrix,
        bs: Matrix,
        gw: Matrix,
        gb: Matrix,
        mw: Matrix,
        mb: Matrix,
    },
    ReLU,
    Tanh,
}

impl Layer {
    fn new_linear(n_in: usize, n_out: usize) -> Self {
        let x = (6.0 / (n_in as f32 + n_out as f32)).sqrt();
        Layer::Linear {
            ws: Matrix::random(n_in, n_out, -x, x),
            bs: Matrix::zeros(n_out, 1),
            gw: Matrix::zeros(n_in, n_out),
            gb: Matrix::zeros(n_out, 1),
            mw: Matrix::zeros(n_in, n_out),
            mb: Matrix::zeros(n_out, 1),
        }
    }

    fn forward(&self, inputs: Matrix) -> Matrix {
        match self {
            Layer::Linear {
                ws,
                bs,
                gw: _,
                gb: _,
                mw: _,
                mb: _,
            } => inputs.matmul(ws).matadd(bs),
            Layer::ReLU => inputs.relu(),
            Layer::Tanh => inputs.tanh(),
        }
    }
}

#[derive(Debug)]
pub struct ActivationCache {
    body: Vec<Matrix>,
    p_head: Vec<Matrix>,
    v_head: Vec<Matrix>,
}

#[derive(Debug)]
pub struct Network {
    body: Vec<Layer>,
    p_head: Vec<Layer>,
    v_head: Vec<Layer>,
}

impl Network {
    pub fn new(
        n_inputs: usize,
        n_hidden: usize,
        n_outputs: usize,
        n_layers: usize,
        n_p_layers: usize,
        n_v_layers: usize,
    ) -> Self {
        let mut n = Self {
            body: vec![Layer::new_linear(n_inputs, n_hidden), Layer::ReLU],
            p_head: Vec::new(),
            v_head: Vec::new(),
        };

        for _ in 1..n_layers {
            n.body.push(Layer::new_linear(n_hidden, n_hidden));
            n.body.push(Layer::ReLU);
        }

        for _ in 1..n_p_layers {
            n.p_head.push(Layer::new_linear(n_hidden, n_hidden));
            n.p_head.push(Layer::ReLU);
        }
        n.p_head.push(Layer::new_linear(n_hidden, n_outputs));

        for _ in 1..n_v_layers {
            n.v_head.push(Layer::new_linear(n_hidden, n_hidden));
            n.v_head.push(Layer::ReLU);
        }
        n.v_head.push(Layer::new_linear(n_hidden, 1));
        n.v_head.push(Layer::Tanh);

        n
    }

    pub fn forward(&self, inputs: Matrix) -> (Matrix, f32, ActivationCache) {
        // NOTE: matrices are passed by value. reference
        // may allow for later optimizations

        let mut a = inputs.clone();
        let mut a_cache = ActivationCache {
            body: Vec::new(),
            p_head: Vec::new(),
            v_head: Vec::new(),
        };
        a_cache.body.push(inputs.clone());

        for layer in &self.body {
            a = layer.forward(a);
            a_cache.body.push(a.clone());
        }
        let b_out = a.clone();
        a_cache.p_head.push(b_out.clone());
        a_cache.v_head.push(b_out.clone());

        for layer in &self.p_head {
            a = layer.forward(a);
            a_cache.p_head.push(a.clone());
        }
        let p_out = a.clone();

        a = b_out;
        for layer in &self.v_head {
            a = layer.forward(a);
            a_cache.v_head.push(a.clone());
        }

        (p_out, a.data[0], a_cache)
    }

    pub fn backward() {}
}
