Setup instructions:

1) Create a folder with name, for example "X"
2) Copy the python code from src folder to this new folder "X"
4) Download the data files from google drive link provided in the data folder's txt file
5) Save these downloaded data files in a new folder named "dataset" within folder "X"




Necessary built-in libraries to be installed:
1) Tensorflow -1
2) tqdm




Code run Command:
MR: 
python3 ProjectCode.py --task mr --model Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task mr --model DS_Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task mr --model DC_Bi_LSTM True --resume_training True --has_devset False

SST1:
python3 ProjectCode.py --task sst1 --model Bi_LSTM True --resume_training True --has_devset True
python3 ProjectCode.py --task sst1 --model DS_Bi_LSTM True --resume_training True --has_devset True
python3 ProjectCode.py --task sst1 --model DC_Bi_LSTM True --resume_training True --has_devset True

SST2:
python3 ProjectCode.py --task sst2 --model Bi_LSTM True --resume_training True --has_devset True
python3 ProjectCode.py --task sst2 --model DS_Bi_LSTM True --resume_training True --has_devset True
python3 ProjectCode.py --task sst2 --model DC_Bi_LSTM True --resume_training True --has_devset True

Subj:
python3 ProjectCode.py --task subj --model Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task subj --model DS_Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task subj --model DC_Bi_LSTM True --resume_training True --has_devset False

TREC:
python3 ProjectCode.py --task trec --model Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task trec --model DS_Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task trec --model DC_Bi_LSTM True --resume_training True --has_devset False

CR:
python3 ProjectCode.py --task cr --model Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task cr --model DS_Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task cr --model DC_Bi_LSTM True --resume_training True --has_devset False

MPQA:
python3 ProjectCode.py --task mpqa --model Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task mpqa --model DS_Bi_LSTM True --resume_training True --has_devset False
python3 ProjectCode.py --task mpqa --model DC_Bi_LSTM True --resume_training True --has_devset False