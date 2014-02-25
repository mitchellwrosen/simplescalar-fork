#!/bin/bash

if [[ $# == 0 ]]; then
   echo "arg missing"
   exit -1
fi

function dowork {
   echo -n "$1: "
   $1 2>&1 1>/dev/null | grep bpred_dir_rate | awk '{print $2}'
}

echo "### TAKEN"
dowork "./sim-bpred -bpred taken $1"
echo ""

echo "### NOTTAKEN"
dowork "./sim-bpred -bpred nottaken $1"
echo ""

echo "### BIMOD"
for i in {1..16}; do
   two_exp_i=$((2**$i))
   dowork "./sim-bpred -bpred bimod -bpred:bimod $two_exp_i $1"
done
echo ""

for multiplier in 1 2 4; do
   echo "### 2LEV - GAp, M=${multiplier}*2^W"
   for w in {1..16}; do
      m=$(($multiplier * (2**$w)))
      dowork "./sim-bpred -bpred 2lev -bpred:2lev 1 $m $w 0 $1"
   done
   echo ""
done

for w in {1..16}; do
   echo "### 2LEV - PAg, W=$w"
   for exp in {1..16}; do
      n=$((2**$exp))
      two_exp_w=$((2**$w))
      dowork "./sim-bpred -bpred 2lev -bpred:2lev $n $two_exp_w $w 0 $1"
   done
   echo ""
done

echo "### 2LEV - PAp"
for exp in {1..3}; do
   for w in {1..8}; do
      n=$((2**$exp))
      m=$((2**($w+$n)))
      dowork "./sim-bpred -bpred 2lev -bpred:2lev $n $m $w 0 $1"
   done
done
echo ""

echo "### 2LEV - gshare"
for w in {1..16}; do
   two_exp_w=$((2**$w))
   dowork "./sim-bpred -bpred 2lev -bpred:2lev 1 $two_exp_w $w 1 $1"
done
echo ""
