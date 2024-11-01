CODE="code_phi/main.py"

model="cherry"
epoch=250
out_dir="out_cherry"
data_dir="data_cherry"
data="cherry"
granularity="species"
period=1
seed=127

python3 $CODE --model $model --data $data --epoch $epoch --out_dir $out_dir --data_dir $data_dir --seed $seed --granularity $granularity --seed $seed --cherry_period $period
