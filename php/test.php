<?php

require_once('../vendor/autoload.php');

$a = [[1, 2], [3, 4]];
$b = [[5, 6], [7, 8]];

$dl = new DL();

// var_dump($dl->sub($a, $b));
// var_dump($dl->softmax([0.3, 2.9, 4.0]));
// var_dump($dl->cross_entropy_error(
//     [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
//     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
// ));
var_dump($dl->numerical_gradiant(function($x){
    return $x[0] * $x[0] + $x[1] * $x[1];
}, [3, 4]));
exit;

class DL {

    function sigmoid(Array $x) {
        return array_map(function($tmp) {
            return 1 / (1 + exp(-1 * $tmp));
        }, $x);
    }

    function relu(Array $x) {
        return array_map(function($tmp) {
            return max(0, $tmp);
        }, $x);
    }

    function softmax(Array $x) {
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

    function mean_squared_error(Array $y, Array $t) {
        return 0.5 * array_sum(array_map(function($tmp) {
            return pow($tmp, 2);
        }, $this->sub($y, $t)));
    }

    function cross_entropy_error(Array $x, Array $t) {
        $delta = 1e-7;
        return -1 * array_sum($this->mul($t, $this->loga($this->add_scalar($x, $delta))));
    }

    function numerical_diff($f, $x) {
        $h = 1e-4;
        return ($f($x + $h) - $f($x - $h)) / (2 * $h);
    }

    function numerical_gradiant($f, $x) {
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

    function add_scalar(Array $x, $c) {
        $y = [];
        foreach ($x as $tmp) {
            $y[] = $tmp + $c;
        }
        return $y;
    }

    function mul(Array $x, Array $t) {
        $y = [];
        for ($i = 0; $i < count($x); $i++) {
            $y[$i] = $x[$i] * $t[$i];
        }
        return $y;
    }

    function loga(Array $x) {
        return array_map(function($tmp) {
            return log($tmp);
        }, $x);
    }

    function add(Array $a, Array $b) {
        if (count($a) !== count($b)) {
            return false;
        }
        $c = [];
        $row = count($a);
        for ($i = 0; $i < $row; $i++) {
            if (is_array($a[$i])) {
                $c[$i] = $this->add($a[$i], $b[$i]);
            } else {
                $c[$i] = $a[$i] + $b[$i];
            }
        }
        return $c;
    }

    function sub(Array $a, Array $b) {
        if (count($a) !== count($b)) {
            return false;
        }
        $c = [];
        $row = count($a);
        for ($i = 0; $i < $row; $i++) {
            if (is_array($a[$i])) {
                $c[$i] = $this->sub($a[$i], $b[$i]);
            } else {
                $c[$i] = $a[$i] - $b[$i];
            }
        }
        return $c;
    }

    function dot(Array $a, Array $b) {
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
}
