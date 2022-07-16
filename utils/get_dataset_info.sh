#!/bin/sh

cd $LOG_DIR

print_info() {
  echo "Files:"
  ls $1
  
  echo
  echo "Total Size Compressed:"
  cat $1 | wc -c | numfmt --to=si
  
  echo
  echo "Total Size Uncompressed:"
  zcat $1 | wc -c | numfmt --to=si
  
  echo
  echo "Total Lines:"
  zcat $1 | wc -l
  echo 
  echo
}

echo "SLURM DATA"
print_info 'sacct-*-*'

echo "/var/log/messages"
print_info 'messages-*'

echo "/var/log/secure"
print_info 'secure-*'
