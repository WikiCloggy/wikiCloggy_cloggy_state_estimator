import pickle
import os

label = ['exciting', 'stomachache', 'butt_scooting', 'stressed', 'very_aggressive']
file = open('./data/label.txt', 'wb')
pickle.dump(label, file)
file.close()

def makeDirectory(path):
    try:
        os.mkdir(path)
    except:
        dir_name = os.path.split(path)[1]
        print(dir_name + " directory is already exist.")

root_path = os.getcwd()
training_data_path = os.path.join(root_path, 'training_data')
makeDirectory(training_data_path)

testing_data_path = os.path.join(root_path, 'testing_data')
makeDirectory(testing_data_path)

for i in range(len(label)):
    training_label_path = os.path.join(training_data_path, label[i])
    makeDirectory(training_label_path)
    testing_label_path = os.path.join(testing_data_path, label[i])
    makeDirectory((testing_label_path))
