## bar-ai-main

# conda env
<!-- conda env remove -n face-api -->
conda update --all -y
conda create -n  face-api  python=3.10 -y
conda activate face-api
conda install -c conda-forge jupyter -y

<!-- conda install -c conda-forge jupyterlab -y
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge openh264 -y -->

python -m pip install --upgrade pip

<!-- cuda -->
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
<!-- intel gpu -->
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

<!-- windows -->
pip install ultralytics --user


人臉dataset
https://susanqq.github.io/UTKFace/


embedding 參考
https://github.com/timesler/facenet-pytorch/tree/master

<!-- 年紀性別 age-gender，目前用這個 -->
https://github.com/tae898/age-gender

# 目前使用
https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/tree/master?tab=readme-ov-file


<!-- 情緒 -->
# 目前使用
https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images.git


HSEmotion (High-Speed face Emotion recognition) library
https://github.com/av-savchenko/face-emotion-recognition