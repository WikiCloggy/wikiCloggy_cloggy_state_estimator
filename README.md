# wikiCloggy_cloggy_state_estimator
강아지 상태를 추정해주는 딥러닝 모델  
**이 코드는 밑바닥부터 시작하는 딥러닝 책을 보면서 작성하였습니다.**

**I wrote this code while looking at the book 'Deep Learning from Scratch'.**
# Requirements
Python3 and later  
numpy  
opencv

# Usage
Class cloggyNet is the neural network designed for trainig the cloggy states.  
Class cloggy_state_estimator is the trained cloggyNet and it can predict a cloggy state.  
If you look at the demo.py, you can get information about how to predict cloggy state with cloggy_state_estimator.

# Training
If you get more state skeleton data, than just run cloggyNet_trainer.py.  
If you want to add more state, add more state to label in label_maker.py and run. Then, add data to created label directories.

If you want to make skeleton data, than use cloggy_dataset_maker in <https://github.com/WikiCloggy/cloggy_dataset_maker>!
