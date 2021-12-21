##$ -S /bin/bash
###$ -q mazda
##$ -pe mazda 32
####$ -l ngpus=1
##$ -N conformer_gen_CPU
##$ -M xiaqiancheng@nibs.ac.cn
##$ -o /pubhome/qcxia02/git-repo/AI-CONF/GeoMol/out
##$ -e /pubhome/qcxia02/git-repo/AI-CONF/GeoMol/error
##$ -cwd
### $ -now y

conda activate GeoMol-cuda11x

'''
Usage:
# python get_results.py --testpath $testpath --core 1 --gen test_GeoMol_20.pickle --rdkit test_rdkit_20.pickle --no 1

# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-11-15-37
testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati/pre-train
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-19-20-5
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati/train_5epoch
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-11-25
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-13-00
core=8
pklname=test_GeoMol_50.pickle
# threshold=0.5
rdkerrtxt=rdkit_err_smiles_25.txt
maxmatches=100000
qsub_anywhere.py -c "source get_results.sh $testpath $core $pklname $threshold $rdkerrtxt $maxmatches" -q k233 -n $core -j . -N c$core-$pklname-t$threshold-m$maxmatches --qsub_now
'''

testpath=$1
core=$2
pklname=$3
threshold=$4
rdkerrtxt=$5
maxmatches=$6
# python get_results.py --testpath $testpath --core $core --file $pklname --threshold $threshold --rdkerrtxt $rdkerrtxt --maxmatches $maxmatches
# python get_results.py --testpath $testpath --core $core --file $pklname --threshold $threshold --removeH --rdkerrtxt $rdkerrtxt --maxmatches $maxmatches

python get_results.py --testpath $testpath --core $core --file $pklname --threshold 0.5 --rdkerrtxt $rdkerrtxt --maxmatches $maxmatches
python get_results.py --testpath $testpath --core $core --file $pklname --threshold 0.5 --removeH --rdkerrtxt $rdkerrtxt --maxmatches $maxmatches
python get_results.py --testpath $testpath --core $core --file $pklname --threshold 1.0 --rdkerrtxt $rdkerrtxt --maxmatches $maxmatches
python get_results.py --testpath $testpath --core $core --file $pklname --threshold 1.0 --removeH --rdkerrtxt $rdkerrtxt --maxmatches $maxmatches


