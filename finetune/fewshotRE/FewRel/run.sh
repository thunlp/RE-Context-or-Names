array=(42)
path="paht/to/ckpt"
for seed in ${array[@]}
do
	bash train.sh 7 $seed $path
done
