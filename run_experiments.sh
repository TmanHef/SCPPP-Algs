#!/usr/bin/bash

domains=( blocks logistics depot rovers )
for domain in "${domains[@]}"
do
  if [ "$domain" = "blocks" ]
  then
    problems=( 2-2-blue-red 3-3-blue-red 4-3-blue-red 5-2-blue-red 6-2-blue-red )
  elif [ "$domain" = "logistics" ];
  then
    problems=( 1-3 2-3 2-4 2-5 3-4 )
  elif [ "$domain" = "depot" ];
  then
    problems=( 4-3 5-3 5-4 7-5 8-5 )
  elif [ "$domain" = "rovers" ];
  then
    problems=( 2-2 3-2 3-3 4-3 3-4 )
  else
    echo "NO PROBLEMS FOR $domain DOMAIN"
    problems=()
  fi

  for problem in "${problems[@]}"
  do
    echo "planning $domain-$problem with DRTDP"
    python main.py --domain "$domain" --problem "$problem" --drtdp
    echo "planning $domain-$problem with PS-RTDP"
    python main.py --domain "$domain" --problem "$problem"
  done

echo "finished with $domain domain"
done

echo "DONE PLANNING!"