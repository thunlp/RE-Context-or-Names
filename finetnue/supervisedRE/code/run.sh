array=(42 43 44 45 46)
ckpt="None"
for seed in ${array[@]}
do
	bash train.sh 1 $seed $ckpt 1 6
done
