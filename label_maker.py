import pickle
import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))

def setup_label(command_list, label_list):
    if len(command_list) > 1:
        for i in range(1, len(command_list)):
            if command_list[i] == '-add' or command_list[i] == '-a':
                i += 1
                while command_list[i][0] != '-':
                    try:
                        index = label_list.index(command_list[i])
                    except:
                        label_list.append(command_list[i])
                    i += 1
                    if i == len(command_list):
                        break

if __name__ == '__main__':
    label_path = os.path.join(root_path, 'data/label.txt')
    try:
        file = open(label_path, 'rb')
        label = pickle.load(file)
        file.close()
    except:
        init_label = ['exciting', 'stomachache', 'butt_scooting', 'stressed', 'very_aggressive']
        label = init_label

    setup_label(sys.argv, label)

    file = open(label_path, 'wb')
    pickle.dump(label, file)
    file.close()

    def makeDirectory(path):
        try:
            os.mkdir(path)
        except:
            dir_name = os.path.split(path)[1]
            print(dir_name + " directory is already exist.")

    training_data_path = os.path.join(root_path, 'training_data')
    makeDirectory(training_data_path)

    testing_data_path = os.path.join(root_path, 'testing_data')
    makeDirectory(testing_data_path)

    for i in range(len(label)):
        training_label_path = os.path.join(training_data_path, label[i])
        makeDirectory(training_label_path)
        testing_label_path = os.path.join(testing_data_path, label[i])
        makeDirectory((testing_label_path))
