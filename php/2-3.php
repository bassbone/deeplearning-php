<?php

require_once('../vendor/autoload.php');

use NumPHP\Core\NumArray;

echo func_or(0, 0).PHP_EOL;
echo func_or(1, 0).PHP_EOL;
echo func_or(0, 1).PHP_EOL;
echo func_or(1, 1).PHP_EOL;

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
