
CODE="code_phi/main.py"
out_dir="out_eval_v2" # not used
data_dir="data_cherry"
data="cherry"

model='cl4phi'
model_pretrained='out_cherry_v3/model-cl4phi-data-cherry-granu-species-epoch-150-aug-0.0:0.5-seed-123.pth'

python3 $CODE --model $model --data $data --model_pretrained $model_pretrained --out_dir $out_dir --data_dir $data_dir
