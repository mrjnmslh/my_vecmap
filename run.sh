#!/usr/bin/env bash
DATA_DIR=/data
src=fa
tgt=en
#random_list=$(python3 -c 'import random; random.seed(0); print(" ".join([random.randint(0, 1000) for _ in range(10)]))') # random seeds
random_list=$(python -c "import random; random.seed(0); print(' '.join([str(random.randint(0, 1000)) for _ in range(10)]))") # random seeds
#random=`awk -v awk_var="$i" 'BEGIN{srand(awk_var);print int(rand()*10000%1000)}'`
random_list=(497)
echo $random_list
for s in ${random_list[@]}
do
echo $s
#python3 map_embeddings.py --unsupervised $DATA_DIR/wiki.$src.vec $DATA_DIR/wiki.$tgt.vec output/wiki-mapped-$src-$tgt-$s.$src output/wiki-mapped-$src-$tgt-$s.$tgt --cuda --log train_log_$src-$tgt.txt -v  --validation data/dictionaries/$src-$tgt.5000-6500.txt --unsupervised_vocab 4000 --seed $s --device_id 3
python map_embeddings.py --unsupervised $DATA_DIR/wiki.fa.vec $DATA_DIR/wiki.en.vec output/src.mapped.txt output/trg.mapped.txt --cuda --log output/train_log.txt -v  --validation data/dictionaries/en-fa.test.txt --unsupervised_vocab 4000 --seed $s --device_id 1
done
