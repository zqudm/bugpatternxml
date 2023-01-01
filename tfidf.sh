if [ -z ${1} ]; then
    echo "bash tfidf.sh [dataset]"
    exit
fi

dataset=$1
data_dir=./${dataset}
output_dir=./${data_dir}/tfidf
model_dir=./${dataset}/model
tr_code=${data_dir}/X.trn.txt
ts_code=${data_dir}/X.tst.txt
conf=${data_dir}/config.json

if [ -e ${data_dir} ]
then
    echo "data dir exists ok"
else 
   echo "data dir not exists"
   exit -1 
fi

if [ -e ${output_dir} ]
then
    echo "output dir exists ok"
else
   echo "output dir not exists"
   exit -1 
fi

if [ -e ${tr_code} ]
then
    echo "train code exists ok"
else
   echo "train code  not exists"
   exit -1 
fi

if [ -e ${ts_code} ]
then
    echo "test code exists ok"
else
   echo "test code  not exists"
   exit -1 
fi

if [ -e ${conf} ]
then
    echo "conf exists ok"
else
   echo "conf  not exists"
   exit -1 
fi


python3 -m pecos.utils.featurization.text.preprocess build --text-pos 0 --input-text-path ${tr_code} --vectorizer-config-path ${conf}  --output-model-folder ${model_dir}

python3 -m pecos.utils.featurization.text.preprocess run \
  --text-pos 0 \ 
  --input-preprocessor-folder ${model_dir} \
  --input-text-path ${tr_code} \
  --output-inst-path ${output_dir}/X.trn.npz

python3 -m pecos.utils.featurization.text.preprocess run \
  --text-pos 0 \ 
  --input-preprocessor-folder ${model_dir} \
  --input-text-path ${ts_code} \
  --output-inst-path ${output_dir}/X.tst.npz

if [ ! -f ${output_dir}/X.trn.npz ]; then
    echo "X.trn.npz not found!"
fi

if [ ! -f ${output_dir}/X.tst.npz ]; then
    echo "X.tst.npz not found!"
fi


