python amlsim_process.py data/transactions.csv amlsim
python amlsim_xgb.py -d amlsim -s > output_xgb_amlsim_cv.txt

python amlsim_process.py data/transactions_sample.csv amlsim_sample
python amlsim_xgb.py -d amlsim_sample -s > output_xgb_amlsim_sample_cv.txt

python amlsim_process.py data/transactions_10k.csv amlsim_10k
python amlsim_xgb.py -d amlsim_10k -s > output_xgb_amlsim_10k_cv.txt

python amlsim_process.py data/transactions_oversample.csv amlsim_oversample
python amlsim_xgb.py -d amlsim_oversample -s > output_xgb_amlsim_oversample_cv.txt
