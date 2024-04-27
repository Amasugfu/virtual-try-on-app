#!/bin/bash

list=(
    "m2" 
    "msys2" 
    "pywin" 
    "twisted-iocpsupport" 
    "powershell" 
    "winpty" 
    "menuinst" 
    "icc_rt" 
    "vs2015" 
    " vc=" 
)

for i in "${list[@]}" 
do
    regex=${regex:+$regex|}$i
done

grep -E "$regex" $1 > environment_linux.yml