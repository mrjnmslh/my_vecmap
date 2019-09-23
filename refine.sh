#!/usr/bin/env bash
SRC_LANG="en"
TGT_LANG="zh"
python unsupercised.py --unsupervised /disk/xfbai/corpus/monolingual/wiki.$SRC_LANG.vec /disk/xfbai/corpus/monolingual/wiki.$TGT_LANG.vec su.$SRC_LANG-$TGT_LANG.$SRC_LANG su.$SRC_LANG-$TGT_LANG.$TGT_LANG  --cuda --log train_log.txt -v -d ../MyCycleBLI/src/$SRC_LANG-$TGT_LANG/best/$SRC_LANG-$TGT_LANG.dict --validation ../MyCycleBLI/data/bilingual_dicts/$SRC_LANG-$TGT_LANG.5000-6500.txt
python unsupercised.py --unsupervised /home/hlcao/xfbai/embeddings/en.emb.txt /home/hlcao/xfbai/embeddings/it.emb.txt Res.en-it.en Res.en-it.it --cuda --log train_log_en2it.txt -v -d ../BiAAE/src/dict/en-it.dict --validation ../BiAAE/data/bilingual_dicts/en-it.5000-6500.txt --device_id 1
