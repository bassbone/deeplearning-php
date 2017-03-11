<?php

require_once('../vendor/autoload.php');

use NumPHP\Core\NumArray;

$circuit = new Circuit();

echo $circuit->func_xor(0, 0).PHP_EOL;
echo $circuit->func_xor(1, 0).PHP_EOL;
echo $circuit->func_xor(0, 1).PHP_EOL;
echo $circuit->func_xor(1, 1).PHP_EOL;

class Circuit {

    function func_xor($x1, $x2) {
        $s1 = $this->func_nand($x1, $x2);
        $s2 = $this->func_or($x1, $x2);
        $y = $this->func_and($s1, $s2);
        return $y;
    }

    function func_and($x1, $x2) {
        $x = new NumArray([$x1, $x2]);
        $w = new NumArray([0.5, 0.5]);
        $b = -0.7;
        $tmp = $w->mult($x)->sum()->getData() + $b;
        if ($tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    function func_nand($x1, $x2) {
        $x = new NumArray([$x1, $x2]);
        $w = new NumArray([-0.5, -0.5]);
        $b = 0.7;
        $tmp = $w->mult($x)->sum()->getData() + $b;
        if ($tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    function func_or($x1, $x2) {
        $x = new NumArray([$x1, $x2]);
        $w = new NumArray([0.5, 0.5]);
        $b = -0.2;
        $tmp = $w->mult($x)->sum()->getData() + $b;
        if ($tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }
}
