# Pre-process data
cd code 
python prepare_data.py --dataset cp 
python prepare_data.py --dataset mtb 

# Sample train set
python utils.py --dataset tacred