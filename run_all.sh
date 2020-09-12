echo 'amlsim_sample xgboost data'
python amlsim_process.py data/transactions_sample.csv amlsim_sample
echo 'amlsim_sample xgboost train'
python amlsim_xgb.py -d amlsim_sample -s > output/xgb_amlsim_sample_cv.txt

echo 'amlsim_sample catboost data'
python amlsim_process.py data/transactions_sample.csv amlsim_sample
echo 'amlsim_sample catboost train'
python amlsim_cat.py -d amlsim_sample -s > output/cat_amlsim_sample_cv.txt

echo 'amlsim xgboost data'
python amlsim_process.py data/transactions.csv amlsim
echo 'amlsim xgboost train'
python amlsim_xgb.py -d amlsim -s > output/xgb_amlsim_cv.txt

echo 'amlsim catboost data'
python amlsim_process.py data/transactions.csv amlsim
echo 'amlsim catboost train'
python amlsim_cat.py -d amlsim > output/cat_amlsim_cv.txt

echo 'amlsim_10k xgboost data'
python amlsim_process.py data/transactions_10k.csv amlsim_10k
echo 'amlsim_10k xgboost train'
python amlsim_xgb.py -d amlsim_10k -s > output/xgb_amlsim_10k_cv.txt

echo 'amlsim_10k catboost data'
python amlsim_process.py data/transactions_10k.csv amlsim_10k
echo 'amlsim_10k catboost train'
python amlsim_cat.py -d amlsim_10k > output/cat_amlsim_10k_cv.txt

echo 'amlsim_oversample xgboost data'
python amlsim_process.py data/transactions_oversample.csv amlsim_oversample
echo 'amlsim_oversample xgboost train'
python amlsim_xgb.py -d amlsim_oversample -s > output/xgb_amlsim_oversample_cv.txt

echo 'amlsim_oversample catboost data'
python amlsim_process.py data/transactions_oversample.csv amlsim_oversample
echo 'amlsim_oversample catboost train'
python amlsim_cat.py -d amlsim_oversample > output/cat_amlsim_oversample_cv.txt

