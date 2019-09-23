#!/usr/bin/env bash
SRC_LANG="en"
TGT_LANG="zh"
#python unsupercised.py --unsupervised /disk/xfbai/corpus/monolingual/wiki.$SRC_LANG.vec /disk/xfbai/corpus/monolingual/wiki.$TGT_LANG.vec Su.$SRC_LANG-$TGT_LANG.$SRC_LANG Su.$SRC_LANG-$TGT_LANG.$TGT_LANG  --cuda --log train_log.txt -v -d ../MyCycleBLI/src/tune2/$SRC_LANG-$TGT_LANG/best/$SRC_LANG-$TGT_LANG.dict --validation ../MyCycleBLI/data/bilingual_dicts/$SRC_LANG-$TGT_LANG.5000-6500.txt
python unsupercised.py --unsupervised /disk/xfbai/corpus/monolingual/wiki.$SRC_LANG.vec /disk/xfbai/corpus/monolingual/wiki.$TGT_LANG.vec Su.$SRC_LANG-$TGT_LANG.$SRC_LANG Su.$SRC_LANG-$TGT_LANG.$TGT_LANG  --cuda --log train_log.txt -v -d ../MyCycleBLI/src/tune2/$SRC_LANG-$TGT_LANG/best/$SRC_LANG-$TGT_LANG.dict --validation ../MyCycleBLI/data/bilingual_dicts/$SRC_LANG-$TGT_LANG.5000-6500.txt
#python unsupercised.py --unsupervised /disk/xfbai/corpus/monolingual/wiki.$SRC_LANG.vec /disk/xfbai/corpus/monolingual/wiki.$TGT_LANG.vec su.$SRC_LANG-$TGT_LANG.$SRC_LANG su.$SRC_LANG-$TGT_LANG.$TGT_LANG  --cuda --log train_log.txt -v -d ../MyCycleBLI/src/tune2/$SRC_LANG-$TGT_LANG/best/$SRC_LANG-$TGT_LANG.dict 

