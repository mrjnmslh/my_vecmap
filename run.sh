#!/usr/bin/env bash
DATA_DIR=/data/embeddings
src=de
tgt=en
#random_list=$(python3 -c 'import random; random.seed(0); print(" ".join([random.randint(0, 1000) for _ in range(10)]))') # random seeds
random_list=$(python3 -c "import random; random.seed(0); print(' '.join([str(random.randint(0, 1000)) for _ in range(10)]))") # random seeds
#random=`awk -v awk_var="$i" 'BEGIN{srand(awk_var);print int(rand()*10000%1000)}'`
random_list=(497)
echo $random_list
for s in ${random_list[@]}
do
echo $s
#python3 map_embeddings.py --unsupervised $DATA_DIR/wiki.$src.vec $DATA_DIR/wiki.$tgt.vec output/wiki-mapped-$src-$tgt-$s.$src output/wiki-mapped-$src-$tgt-$s.$tgt --cuda --log train_log_$src-$tgt.txt -v  --validation data/dictionaries/$src-$tgt.5000-6500.txt --unsupervised_vocab 4000 --seed $s --device_id 3
python3 map_embeddings.py --unsupervised $DATA_DIR/$src.emb.txt $DATA_DIR/$tgt.emb.txt output/wacky-mapped-$src-$tgt-$s.$src output/wacky-mapped-$src-$tgt-$s.$tgt --cuda --log output/wacky-train_log_$src-$tgt.txt -v  --validation data/dictionaries/$src-$tgt.test.txt --unsupervised_vocab 4000 --seed $s --device_id 1
done
