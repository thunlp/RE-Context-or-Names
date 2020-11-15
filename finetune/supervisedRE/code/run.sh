array=(42 43 44 45 46)
ckpt="ckpt_cp"
for seed in ${array[@]}
do
	bash train.sh 1 $seed $ckpt 0.01 20
done
