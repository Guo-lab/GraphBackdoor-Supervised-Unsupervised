dataset="MUTAG"
graphtype='ER'

gpu=0
target=1

p=0.8
frac=0.05

echo ${dataset}
echo ${graphtype}
echo ${frac}
echo ${p}

for triggersize in 4

do
  echo $triggersize
  python3 main_backdoor_supervised.py --device $gpu --dataset ${dataset} --target ${target} --seed=7 --fold_idx=0\
	--graphtype ${graphtype}  --frac ${frac} --prob $p --triggersize $triggersize --K=3  --degree_as_tag\
	--filenamebd myresults/${dataset}_${graphtype}_prob_${p}_frac_${frac}_trig_${triggersize}
done



