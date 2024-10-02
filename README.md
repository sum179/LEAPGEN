## This is the repository for LEAPGen

## Example of simulation scripts:
python main_leapgen.py imr_leapgen --length 30 --epochs 10 --num_tasks 10 --lr 0.05 --output_dir ./output/imr_10t

python main_leapgen.py cifar100_leapgen --length 30 --epochs 10 --num_tasks 10 --lr 0.01 --output_dir ./output/cifar_10t

python main_leapgen.py cub_leapgen --length 30 --epochs 20 --num_tasks 10 --lr 0.005 --output_dir ./output/cub_10t
