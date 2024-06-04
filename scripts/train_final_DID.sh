

cd ../
for m in  mabert camel mdbert arabert arbert mbert labse
do
HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1  python train_TADI.py \
  --pretrained_bert_name $m \
  --dataset $2
done
