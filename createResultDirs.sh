#!/usr/bin/bash

mkdir "result_graphs"
domains=( blocks logistics depot rovers )
for domain in "${domains[@]}"
do
  mkdir result_graphs/$domain
  mkdir result_graphs/$domain/evaluations
  mkdir result_graphs/$domain/expansions
  mkdir result_graphs/$domain/goal_reach_time
  mkdir result_graphs/$domain/messages_count
  mkdir result_graphs/$domain/overall
done