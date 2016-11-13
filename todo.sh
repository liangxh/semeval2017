#! /usr/bin/env bash

for f in $(find . | grep '\.py$')
    do c=$(grep -c "TODO" $f)
    if [ $c != '0' ]
    then
        echo $f '-' $c 'TODOs'
    fi
done

