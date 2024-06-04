

cd ../

HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1  python train_identicators.py   --pretrained_bert_name $2
