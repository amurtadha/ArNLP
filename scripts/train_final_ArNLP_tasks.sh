

cd ../
for task in AJGT ASTD LABR ASAD SemEval17 HateSpeech Offensive Adult ArSAS_Sentiment
#for m in  mabert camel mdbert arabert arbert mbert labse
do
HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1  python train_final.py \
  --num_epoch 6\
  --baseline_plm $2\
  --dataset $task\
  --pretrained_bert_name $3
done
