array=(42)
#path="None"
#path="/data3/private/penghao/REPretrain/ckpt/ckpt_ex_mtb_256/ckpt_of_step_30000"
#path="/data3/private/penghao/REPretrain/ckpt/ckpt_ex_distant_2048/ckpt_of_step_7000"
path="/data3/private/penghao/EMNLP2020/REAnalysis/ckpt/ckpt_cp/ckpt_of_step_3500"
for seed in ${array[@]}
do
	bash train.sh 7 $seed $path
done
