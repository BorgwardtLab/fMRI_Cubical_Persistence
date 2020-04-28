for a in *.json; do
  for b in *.json; do
    if [ "$a" != "$b" ]; then
      diff -s $a $b
    fi
  done
done
