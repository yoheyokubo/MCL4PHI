
CODE="code_phi/main.py"
out_dir="out_eval_v2" # not used
data_dir="data_cherry"
data="cherry"

model='cl4phi'
model_pretrained='out_cherry/model-cl4phi-data-cherry-granu-species-epoch-150-aug-0.0_0.5_0.9_0.99_0.999-seed-123.pth'

python3 $CODE --model $model --data $data --model_pretrained $model_pretrained --out_dir $out_dir --data_dir $data_dir
