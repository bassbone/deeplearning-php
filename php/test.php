<?php

require_once('../vendor/autoload.php');

$a = [[1, 2], [3, 4]];
$b = [[5, 6], [7, 8]];

$dl = new DL();

var_dump($dl->add($a, $b));
var_dump($dl->softmax([0.3, 2.9, 4.0]));
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
