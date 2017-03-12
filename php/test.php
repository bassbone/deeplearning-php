<?php

require_once('../vendor/autoload.php');

$a = [[1, 2], [3, 4]];
$b = [[5, 6], [7, 8]];

// var_dump($dl->sub($a, $b));
// var_dump($dl->softmax([0.3, 2.9, 4.0]));
// var_dump($dl->cross_entropy_error(
//     [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
//     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
// ));
// var_dump(DL::numerical_gradiant(function($x){
//     return $x[0] * $x[0] + $x[1] * $x[1];
// }, [3, 4]));
var_dump(DL::transposition([1,2,3]));
exit;

class DL {

    static function sigmoid(Array $x) {
        return array_map(function($tmp) {
            return 1 / (1 + exp(-1 * $tmp));
        }, $x);
    }

    static function relu(Array $x) {
        return array_map(function($tmp) {
            return max(0, $tmp);
        }, $x);
    }

    static function softmax(Array $x) {
        $c = max($x);

        $exp_x = [];
        foreach($x as $tmp) {
            $exp_x[] = exp($tmp - $c);            
        }

        $sum_exp_x = array_sum($exp_x);

        $y = [];
        foreach($exp_x as $tmp) {
            $y[] = $tmp / $sum_exp_x;
        }

        return $y;
    }

    static function mean_squared_error(Array $y, Array $t) {
        return 0.5 * array_sum(array_map(function($tmp) {
            return pow($tmp, 2);
        }, self::sub($y, $t)));
    }

    static function cross_entropy_error(Array $x, Array $t) {
        $delta = 1e-7;
        return -1 * array_sum(self::mul($t, self::loga(self::add_scalar($x, $delta))));
    }

    static function numerical_diff($f, $x) {
        $h = 1e-4;
        return ($f($x + $h) - $f($x - $h)) / (2 * $h);
    }

    static function numerical_gradiant($f, $x) {
        $h = 1e-4;
        $grad = [];
        for ($i = 0; $i < count($x); $i++) {
            $tmp_val = $x[$i];
            
            $x[$i] = $tmp_val + $h;
            $fxh1 = $f($x);

            $x[$i] = $tmp_val - $h;
            $fxh2 = $f($x);

            $grad[] = ($fxh1 - $fxh2) / (2 * $h);
            $x[$i] = $tmp_val;
        }
        return $grad;
    }

    static function add_scalar(Array $x, $c) {
        $y = [];
        foreach ($x as $tmp) {
            $y[] = $tmp + $c;
        }
        return $y;
    }

    static function mul(Array $x, Array $t) {
        $y = [];
        for ($i = 0; $i < count($x); $i++) {
            $y[$i] = $x[$i] * $t[$i];
        }
        return $y;
    }

    static function loga(Array $x) {
        return array_map(function($tmp) {
            return log($tmp);
        }, $x);
    }

    static function add(Array $a, Array $b) {
        if (count($a) !== count($b)) {
            return false;
        }
        $c = [];
        $row = count($a);
        for ($i = 0; $i < $row; $i++) {
            if (is_array($a[$i])) {
                $c[$i] = self::add($a[$i], $b[$i]);
            } else {
                $c[$i] = $a[$i] + $b[$i];
            }
        }
        return $c;
    }

    static function sub(Array $a, Array $b) {
        if (count($a) !== count($b)) {
            return false;
        }
        $c = [];
        $row = count($a);
        for ($i = 0; $i < $row; $i++) {
            if (is_array($a[$i])) {
                $c[$i] = self::sub($a[$i], $b[$i]);
            } else {
                $c[$i] = $a[$i] - $b[$i];
            }
        }
        return $c;
    }

    static function div_scalar(Array $x, $c) {
        $y = [];
        foreach ($x as $tmp) {
            $y[] = $tmp / $c;
        }
        return $y;
    }

    static function dot(Array $a, Array $b) {
        $c = [];
        $row = count($a);
        $col = count($b[0]);
        for ($i = 0; $i < $row; $i++) {
            for ($j = 0; $j < $col; $j++) {
                $c[$i][$j] = 0;
                for ($k = 0; $k < count($a[0]); $k++) {
                    $c[$i][$j] += $a[$i][$k] * $b[$k][$j];
                }
            }
        }
        return $c;
    }

    static function transposition(Array $x) {
        $y = [];
        if (is_array($x[0])) {
            for ($i = 0; $i < count($x[0]); $i++) {
                $tmp_row = [];
                for ($j = 0; $j < count($x); $j++) {
                    $tmp_row[] = $x[$j][$i];
                }
                $y[] = $tmp_row;
            }
        } else {
            foreach ($x as $tmp) {
                $y[] = [$tmp];
            }
        }
        return $y;
    }

    static function sum(Array $x, $axis) {
        $y = [];
        for ($i = 0; $i < count($x[$axis]); $i++) {
            $tmp_sum = 0;
            for ($j = 0; $j < count($x); $j++) {
                $tmp_sum += $x[$j][$i];
            }
            $y[] = $tmp_sum;
        }
        return $y;
    }

    static function shape($x) {
        if (is_array($x[0])) {
            return [count($x), count($x[0])];
        } else {
            return [1, count($x)];
        }
    }
}

class Relu {

    private $mask;

    function __construct() {
        $this->mask = [];
    }

    function forward($x) {
        $this->mask = array_map(function($tmp) {
            return ($tmp <= 0);
        }, $x);
        
        return array_map(function($tmp) {
            return ($tmp <= 0) ? 0 : $tmp;
        }, $x);
    }

    function backward($x) {
        return array_map(function($tmp) {
            return ($tmp <= 0) ? 0 : $tmp;
        }, $x);
    }
}

class Sigmoid {

    private $out;

    function __construct() {
        $this->out = [];
    }

    function forward($x) {
        $this->out = array_map(function($tmp) {
            return 1 / (1 + exp(-1 * $tmp));
        }, $x);
        return $this->out;
    }

    function backward($x) {
        $y = [];
        for($i = 0; $i < count($x); $i++) {
            $y[] = $x[$i] * (1 - $this->out[$i]) * $this->out[$i];
        }
        return $y;
    }
}

class Affine {

    private $w;
    private $b;
    private $x;
    public $dw;
    public $db;

    function __construct($w, $b) {
        $this->w = $w;
        $this->b = $b;
        $this->x = $this->dw = $this->db = [];
    }

    function forward($x) {
        $this->x = $x;
        return DL::add(DL::dot($x, $this->w), $b);
    }

    function backward($dout) {
        $dx = DL::dot($dout, DL::transposition($this->w));
        $this->dw = DL::dot(DL::transposition($this->x), $dout);
        $this->db = DL::sum($dout, 0);

        return $dx;
    }
}

class SoftmaxWithLoss {

    private $loss;
    private $y;
    private $t;

    function __construct() {
        $this->loss = $this->y = $this->t = [];
    }

    function forward($x, $t) {
        $this->t = $t;
        $this->y = DL::softmax($x);
        $this->loss = DL::cross_entropy_error($this->y, $this->t);
        return $this->loss;
    }

    function backward($dout = 1) {
        $batch_size = (DL::shape($this->t))[0];
        return DL::div_scalar(DL::sub($this->y, $this->t), $batch_size);
    }
}

class GaussianRandom {

    static function uniform_rand() {
        return mt_rand() / mt_getrandmax();
    }

    static function get() {
        $x = self::uniform_rand();
        $y = self::uniform_rand();
        return sqrt(-2 * log($x)) * cos(2 * M_PI * $y);
    }

    static function get_matrix($col, $row, $c) {
        $matrix = [];
        for ($i = 0; $i < $col; $i++) {
            $tmp_col = [];
            for ($j = 0; $j < $row; $j++) {
                $tmp_col[] = self::get() * $c;
            }
            $matrix[] = $tmp_col;
        }
        return $matrix;
    }
}

class TwoLayerNet {

    private $params;
    private $layers;
    private $last_layer;

    function __construct($input_size, $hidden_size, $output_size, $weight_init_std = 0.01) {

        $this->params = [];
        $this->params['w1'] = GaussianRandom::get_matrix($input_size, $hidden_size, $weight_init_std);
        $this->params['b1'] = array_pad([], $hidden_size, 0);
        $this->params['w1'] = GaussianRandom::get_matrix($hidden_size, $output_size, $weight_init_std);
        $this->params['b2'] = array_pad([], $output_size, 0);

        $this->layers = [];
        $this->layers['Affine1'] = new Affine($this->params['w1'], $this->params['b1']);
        $this->layers['Relu1'] = new Relu();
        $this->layers['Affine2'] = new Affine($this->params['w2'], $this->params['b2']);
        $this->last_layer = new SoftmaxWithLoss();
    }

    function predict($x, $t) {
        foreach ($this->layers as $layer) {
            $x = $layer->forward($x);
        }
        return $x;
    }

    function loss($x, $t) {
        $y = $this->predict($x);
        return $this->last_layer->forward($y, $t);
    }

    function gradient($x, $t) {
        $this->loss($x, $t);

        $dout = 1;
        $dout = $this->last_layer->backward($dout);

        $layers = array_reverse($this->layers);
        foreach ($layer as $layers) {
            $dout = $layer->backward($dout);
        }

        $grads = [];
        $grads['w1'] = (($this->layers)['Affine1'])->dw;
        $grads['b1'] = (($this->layers)['Affine1'])->db;
        $grads['w2'] = (($this->layers)['Affine2'])->dw;
        $grads['b2'] = (($this->layers)['Affine2'])->db;

        return $grads;
    }
}