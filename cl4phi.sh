
CODE="code_phi/main.py"

model="cl4phi"
epoch=150
out_dir="out_cherry_v5"
data_dir="data_cherry"
data="cherry"
granularity="species"
aug="0.0:0.5:0.9:0.99:0.999" # set "0.0" for cl4phi
seed=123

python3 $CODE --model $model --data $data --aug $aug --epoch $epoch --out_dir $out_dir --data_dir $data_dir --seed $seed --granularity $granularity --seed $seed
