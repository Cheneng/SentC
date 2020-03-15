python3 train_tfs.py \
    --data_dir='./data/train_pairs' \
    --dict_path='./checkpoint/dict_20000.pkl' \
    --embedding_path_random='./model/save_embedding_97and3.ckpt' \
    --save_model_path='./checkpoint/Transformer_lr{}_b{}_head{}_layer{}_ff{}/' \
    --batch_size=200 \
    --head=10 \
    --layers=2 \
    --decoder_layers=2 \
    --dim_ffd=100 \
    --lr=3e-4 \
    --epoch=100 \
    --save_text_path='save_text_no_mask'

