CODE="code_phi/main.py"

model="pblks"
out_dir="out_cherry_v3"
data_dir="data_cherry"
data="cherry"
granularity="species"
seed=123

python3 $CODE --model $model --data $data --out_dir $out_dir --data_dir $data_dir --seed $seed --granularity $granularity --seed $seed
