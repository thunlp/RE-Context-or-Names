array=(42)
path="None"
for seed in ${array[@]}
do
	bash train.sh 5 $seed $path
done
