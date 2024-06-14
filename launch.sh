# Encoder only - Test 
python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64 --output_dir models/OadTR_encoder_only --device cuda:1 --test True --resume models/enc_only/checkpoint0004.pth

# Encoder only - Train
python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64 --output_dir models/debug --device cuda:0 --numclass 297 --batch_size 128 --add_labels False
